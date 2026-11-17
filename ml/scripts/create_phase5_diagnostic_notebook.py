"""Genera el notebook nuevo 07_fase5_diagnostico_gradcam_coste_incertidumbre.ipynb.

Nunca modifica notebooks anteriores (02-06, 04b, 04c) ni contiene evaluación de test.
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = '07_fase5_diagnostico_gradcam_coste_incertidumbre.ipynb'


def main():
    package = json.loads((ROOT / 'dist/phase5_diagnostic_package_info.json').read_text(encoding='utf-8'))
    cells = []

    def add(kind, source):
        cell = dict(
            cell_type=kind,
            id=f'phase5diag-{len(cells)}',
            metadata={},
            source=textwrap.dedent(source).strip().splitlines(keepends=True),
        )
        if kind == 'code':
            compile(''.join(cell['source']), f"{NAME}:{cell['id']}", 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''
    # 07 · Fase 5 (Parte 1): Diagnóstico de `validation`, explicabilidad, coste e incertidumbre

    **Alcance exclusivo: conjunto `validation` (`1.241` imágenes, `1.237` grupos `group_id`).**
    - Runtime GPU nuevo en Google Colab. Se restauran `best.pt` verificados; **no se entrena**.
    - Los conjuntos `test` interno y `reserved_external` siguen **cerrados y reservados**.
    - No existe ninguna celda ni opción para abrir `test` o congelar decisiones aquí.
    - Conservar ejecutados en Drive los notebooks `02`–`06`, `04b` y `04c`.
    ''')

    add('markdown', '''
    ## Paso 1 — Montar Drive, verificar ZIP aislado y crear carpeta de resultados nueva
    ''')

    add('code', '''
    import gc, hashlib, importlib, json, stat, subprocess, sys, uuid, zipfile
    from datetime import datetime, timezone
    from pathlib import Path, PurePosixPath
    from google.colab import drive

    drive.mount('/content/drive')
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    ARCHIVE = DRIVE_BASE / 'tbx11k_fase5_diagnostico_colab.zip'
    EXPECTED_PACKAGE_SHA256 = '__PACKAGE_SHA256__'

    def file_sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    if file_sha256(ARCHIVE) != EXPECTED_PACKAGE_SHA256:
        raise ValueError('ZIP distinto de esta entrega del notebook 07. Verifique la versión en Drive.')

    RUN_ID = 'phase5_diag_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    PROJECT_ROOT = Path('/content/tbx11k_phase5_diag') / RUN_ID
    PROJECT_ROOT.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('ZIP con entradas duplicadas.')
        for item in archive.infolist():
            rel = PurePosixPath(item.filename)
            target = (PROJECT_ROOT / item.filename).resolve()
            if (rel.is_absolute() or '..' in rel.parts or ':' in item.filename
                    or chr(92) in item.filename or stat.S_ISLNK(item.external_attr >> 16)
                    or PROJECT_ROOT.resolve() not in target.parents):
                raise ValueError('Ruta ZIP insegura: ' + item.filename)
        checksums = json.loads(archive.read('phase5_diagnostic_checksums.json'))
        if set(names) != set(checksums) | {'phase5_diagnostic_checksums.json'}:
            raise ValueError('Cobertura de checksums incorrecta.')
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Checksum alterado: ' + name)
        archive.extractall(PROJECT_ROOT)

    for name in list(sys.modules):
        if name in {'tbxdata', 'tbxtrain', '_tbx_official_vmamba', '_tbx_official_scan'} or name.startswith(('tbxdata.', 'tbxtrain.')):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

    from tbxtrain.protocol import atomic_json
    OUT = DRIVE_BASE / 'results/phase5_diag' / RUN_ID
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / 'session.json', dict(
        run_id=RUN_ID,
        package_sha256=EXPECTED_PACKAGE_SHA256,
        scope='validation_only',
        decisions_frozen=False,
        test_evaluated=False,
        external_evaluated=False,
        phase5_complete=False,
    ))
    print('Proyecto extraído en:', PROJECT_ROOT)
    print('Salidas nuevas en:', OUT)
    '''.replace('__PACKAGE_SHA256__', package['sha256']))

    add('markdown', '''
    ## Paso 2 — Dependencias protegidas y entorno GPU reproducible
    ''')

    add('code', '''
    import importlib.metadata as metadata
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Seleccione un runtime con GPU CUDA en Google Colab.')

    protected = {}
    for pkg in ['torch', 'torchvision', 'torchaudio']:
        try:
            protected[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    constraints = PROJECT_ROOT / 'runtime_constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\\n' for k, v in protected.items()), encoding='utf-8')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
        '-c', str(constraints), '-r', str(PROJECT_ROOT / 'requirements-phase5.txt'),
    ])
    if any(metadata.version(k) != v for k, v in protected.items()):
        raise RuntimeError('Se alteró torch/torchvision/torchaudio. Reinicie el runtime.')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image as IPyImage, display

    environment = dict(
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        packages={d.metadata['Name']: d.version for d in metadata.distributions()},
    )
    atomic_json(OUT / 'environment.json', environment)
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('Python:', sys.version.split()[0], '| torch:', torch.__version__, '| CUDA:', torch.version.cuda,
          '| GPU:', torch.cuda.get_device_name(0))
    ''')

    add('markdown', '''
    ## Paso 3 — Verificación de los 5 clasificadores, detector `YOLO11n` y dataset `validation`
    ''')

    add('code', '''
    from tbxdata.colab import stage_dataset
    from tbxdata.common import safe_path, sha256, write_csv
    from tbxdata.dataset import TBXDataset
    from tbxdata.validate import validate_manifests
    from tbxtrain.phase5 import CLASSES, validation_rows
    from tbxtrain.phase5_diagnostics import (
        POINTER_ARCHS,
        load_diagnostic_config,
        restore_any_classifier,
        verify_any_classifier,
        verify_detector,
    )

    CONFIG = load_diagnostic_config(PROJECT_ROOT)
    ROWS = validation_rows(PROJECT_ROOT)
    if len(ROWS) != 1241:
        raise ValueError(f'Se esperaban 1241 filas en validation, hay {len(ROWS)}.')

    PREDICTIONS_BY_ARCH = {}
    PROBS_BY_ARCH = {}
    VERIFICATION_ROWS = []
    for spec in CONFIG['classifiers']:
        summary, ordered, y, p, metrics = verify_any_classifier(PROJECT_ROOT, DRIVE_BASE, spec, ROWS, check_checkpoint=True)
        PREDICTIONS_BY_ARCH[spec['architecture']] = ordered
        PROBS_BY_ARCH[spec['architecture']] = p
        VERIFICATION_ROWS.append(dict(
            architecture=spec['architecture'],
            display_name=spec['display_name'],
            run_id=spec['run_id'],
            best_epoch=summary['best_epoch'],
            accuracy=metrics['accuracy'],
            macro_f1=metrics['macro_f1'],
            tb_recall=metrics['per_class']['TB']['recall'],
            tb_precision=metrics['per_class']['TB']['precision'],
            tb_specificity=metrics['per_class']['TB']['specificity'],
            tb_roc_auc=metrics['per_class']['TB']['roc_auc'],
            tb_ap=metrics['per_class']['TB']['average_precision'],
            checkpoint_sha256=spec['sha256'],
        ))

    DET_SPEC = CONFIG['detector']
    DET_WEIGHTS, DET_SHA256, DET_SUMMARY = verify_detector(DRIVE_BASE, DET_SPEC)

    source = DRIVE_BASE / 'TBX11K.zip'
    if not source.exists():
        source = DRIVE_BASE / 'TBX11K'
    DATASET_ROOT = stage_dataset(source, Path('/content/tbx11k_phase5_data_cache'), True)
    transfer = validate_manifests(PROJECT_ROOT, DATASET_ROOT, all_image_hashes=True)
    atomic_json(OUT / 'transfer_integrity.json', transfer)
    atomic_json(OUT / 'artifact_verification.json', dict(
        classifiers=VERIFICATION_ROWS,
        detector=dict(run_id=DET_SPEC['run_id'], sha256=DET_SHA256, summary=DET_SUMMARY),
        test_evaluated=False,
        external_evaluated=False,
    ))
    write_csv(OUT / 'validation_5models_verified.csv', VERIFICATION_ROWS)
    VAL_DATASET = TBXDataset(DATASET_ROOT, PROJECT_ROOT / 'manifests/validation.csv',
                             PROJECT_ROOT / 'configs/data.json', training=False)
    display(pd.DataFrame(VERIFICATION_ROWS).sort_values('macro_f1', ascending=False))
    print('Integridad dataset:', transfer['status'], '| Imágenes verificadas:', transfer['all_image_hashes_verified'])
    ''')

    add('markdown', '''
    ## Paso 4 — Selective scan `oflex` y smoke de carga e inferencia para los 5 clasificadores
    ''')

    add('code', '''
    if importlib.util.find_spec('selective_scan_cuda_oflex') is None:
        cmd = [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/setup_vmamba_colab.py'),
               '--project', str(PROJECT_ROOT), '--drive-base', str(DRIVE_BASE)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (OUT / 'vmamba_setup.log').open('w', encoding='utf-8') as log:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        if proc.wait():
            raise RuntimeError('Falló instalación de selective_scan_cuda_oflex.')
        importlib.invalidate_caches()

    from tbxtrain.architectures import scan_backend
    backend = scan_backend()
    if backend != 'oflex':
        raise RuntimeError(f'Se requiere backend oflex; detectado: {backend}')
    atomic_json(OUT / 'scan_backend.json', dict(backend=backend))

    smoke_indices = [next(i for i, r in enumerate(ROWS) if r['class_name'] == c) for c in CLASSES]
    smoke_records = []
    for spec in CONFIG['classifiers']:
        model = restore_any_classifier(PROJECT_ROOT, DRIVE_BASE, spec, device='cuda', require_oflex=True)
        with torch.inference_mode():
            x = torch.stack([torch.from_numpy(VAL_DATASET[i][0]) for i in smoke_indices]).cuda()
            probs = model(x).softmax(1).cpu().numpy()
        ref = PROBS_BY_ARCH[spec['architecture']][smoke_indices]
        delta = float(np.abs(probs - ref).max())
        passed = bool(np.allclose(probs, ref, atol=1e-3, rtol=1e-3))
        smoke_records.append(dict(
            architecture=spec['architecture'],
            passed=passed,
            max_probability_delta=delta,
            paths=[ROWS[i]['relative_path'] for i in smoke_indices],
        ))
        del model, x
        gc.collect()
        torch.cuda.empty_cache()
        if not passed:
            raise RuntimeError(f'Smoke fallido en {spec["architecture"]}: delta={delta}')

    atomic_json(OUT / 'load_smoke_5models.json', smoke_records)
    display(pd.DataFrame(smoke_records))
    ''')

    add('markdown', '''
    ## Paso 5 — Diagnóstico completo de `YOLO11n` en `validation` y concordancia exploratoria (3 punteros)
    ''')

    add('code', '''
    from ultralytics import YOLO
    from tbxtrain.phase5_diagnostics import (
        concordance_grid_analysis,
        detector_detailed_metrics,
        export_detector_diagnostic_figures,
    )

    if sha256(DET_WEIGHTS) != DET_SHA256:
        raise ValueError('Peso del detector modificado tras verificación.')
    detector = YOLO(str(DET_WEIGHTS))
    if detector.names != {0: 'tb_lesion'}:
        raise ValueError(f'Clases inesperadas en detector: {detector.names}')

    det_opts = dict(DET_SPEC['inference_options'], device=0, save=False, verbose=False)
    DET_PREDICTIONS = {}
    for offset in range(0, len(ROWS), 8):
        batch = ROWS[offset:offset + 8]
        results = detector.predict(
            source=[str(safe_path(DATASET_ROOT, r['relative_path'])) for r in batch],
            **det_opts,
        )
        if len(results) != len(batch):
            raise ValueError('Batch incompleto del detector.')
        for row, res in zip(batch, results):
            if Path(res.path).resolve() != safe_path(DATASET_ROOT, row['relative_path']):
                raise ValueError('Orden de imágenes del detector inconsistente.')
            DET_PREDICTIONS[row['relative_path']] = [
                dict(xyxy=xy, confidence=float(score))
                for xy, score in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist())
            ]
        if offset % 240 == 0:
            print('Detector validation:', min(offset + 8, len(ROWS)), '/', len(ROWS), flush=True)

    del detector
    gc.collect()
    torch.cuda.empty_cache()

    atomic_json(OUT / 'detector_validation_predictions.json', DET_PREDICTIONS)
    atomic_json(OUT / 'detector_inference_options.json', det_opts)

    det_table = detector_detailed_metrics(ROWS, DET_PREDICTIONS, DET_SPEC['thresholds'], match_iou=0.5)
    export_detector_diagnostic_figures(det_table, OUT)
    display(pd.DataFrame(det_table)[[
        'threshold', 'matched_boxes', 'false_positive_boxes', 'lesion_recall',
        'box_precision', 'image_sensitivity_any_box', 'image_sensitivity_localized',
        'fp_boxes_per_image', 'negative_image_alert_rate', 'health_alert_rate', 'sick_alert_rate'
    ]])
    display(IPyImage(filename=str(OUT / 'detector_diagnostic.png'), width=1000))

    grid_summary, default_concordance = concordance_grid_analysis(
        ROWS,
        PREDICTIONS_BY_ARCH,
        DET_PREDICTIONS,
        CONFIG['exploratory_grid']['classifier_thresholds'],
        CONFIG['exploratory_grid']['detector_thresholds'],
        pointer_archs=POINTER_ARCHS,
        default_pair=tuple(CONFIG['exploratory_grid']['default_pair']),
    )
    write_csv(OUT / 'concordance_grid_summary.csv', grid_summary)
    write_csv(OUT / 'concordance_default_validation.csv', default_concordance)
    display(pd.DataFrame(grid_summary))
    ''')

    add('markdown', '''
    ## Paso 6 — Revisión visual determinista de errores, discrepancias y cajas
    ''')

    add('code', '''
    from tbxtrain.phase5_diagnostics import render_visual_review_panels, select_visual_review_cases

    VISUAL_CASES = select_visual_review_cases(
        ROWS,
        PREDICTIONS_BY_ARCH,
        DET_PREDICTIONS,
        pointer_archs=POINTER_ARCHS,
        cls_threshold=0.5,
        det_threshold=0.25,
        max_det_only_non_tb=12,
        sample_tp_per_subtype=3,
    )
    write_csv(OUT / 'visual_review_cases.csv', VISUAL_CASES)
    panel_files = render_visual_review_panels(
        DATASET_ROOT,
        VISUAL_CASES,
        OUT / 'visual_review_panels',
        pointer_archs=POINTER_ARCHS,
        per_page=6,
    )
    print('Casos seleccionados para revisión visual:', len(VISUAL_CASES), '| Páginas:', len(panel_files))
    for fname in panel_files[:3]:
        display(IPyImage(filename=str(OUT / 'visual_review_panels' / fname), width=1000))
    ''')

    add('markdown', '''
    ## Paso 7 — Explicabilidad verificada (Grad-CAM, Attention Rollout, CrossFusion y sanity checks)
    ''')

    add('code', '''
    from tbxtrain.phase5_explainability import (
        render_explainability_panels,
        run_verified_explainability_suite,
        summarize_explainability_checks,
    )

    # Selección determinista de 12 casos representativos de validation para explicabilidad verificada.
    row_by_path = {r['relative_path']: r for r in ROWS}
    idx_by_path = {r['relative_path']: i for i, r in enumerate(ROWS)}

    def pick_by_cat(cat, limit):
        return [c['relative_path'] for c in VISUAL_CASES if cat in c['categories'].split(';')][:limit]

    explain_paths = []
    for p in (pick_by_cat('tb_tp_representative', 4)
              + pick_by_cat('tb_fn', 3)
              + pick_by_cat('tb_fp', 3)):
        if p not in explain_paths:
            explain_paths.append(p)
    # Añadir 1 sick (no-TB) y 1 health de control.
    for cls_target in ('sick', 'health'):
        p_ctrl = next(r['relative_path'] for r in ROWS if r['class_name'] == cls_target and r['relative_path'] not in explain_paths)
        explain_paths.append(p_ctrl)

    EXPLAIN_SAMPLES = []
    for p in explain_paths:
        tensor, _ = VAL_DATASET[idx_by_path[p]]
        r = row_by_path[p]
        EXPLAIN_SAMPLES.append({
            'relative_path': p,
            'class_name': r['class_name'],
            'tensor': tensor,
            'boxes_json': r['boxes_json'],
        })

    all_explain_records = []
    visuals_by_arch = {}
    for spec in CONFIG['classifiers']:
        arch = spec['architecture']
        print('Ejecutando explicabilidad verificada para:', arch, flush=True)
        model = restore_any_classifier(PROJECT_ROOT, DRIVE_BASE, spec, device='cuda', require_oflex=True)
        recs, vis = run_verified_explainability_suite(
            model,
            arch,
            EXPLAIN_SAMPLES,
            device='cuda',
            image_size=512,
            target_class=2,
            contrast_class=0,
            run_randomization_check=True,
        )
        all_explain_records.extend(recs)
        visuals_by_arch[arch] = vis
        del model
        gc.collect()
        torch.cuda.empty_cache()

    explain_flat = summarize_explainability_checks(all_explain_records, OUT)
    explain_df = pd.DataFrame(explain_flat)
    if not (explain_df['all_finite'].all() and explain_df['all_nonzero_grad'].all()):
        raise RuntimeError('Falló chequeo numérico de finitud o no nulidad en Grad-CAM.')

    explain_panel_files = render_explainability_panels(
        DATASET_ROOT,
        visuals_by_arch,
        OUT / 'explainability_panels',
        image_size=512,
    )
    display(explain_df.groupby('architecture')[
        ['min_grad_norm', 'class_contrast_pearson', 'class_contrast_l1',
         'head_randomization_pearson', 'head_randomization_l1', 'gt_enrichment']
    ].mean(numeric_only=True))
    for fname in explain_panel_files[:4]:
        display(IPyImage(filename=str(OUT / 'explainability_panels' / fname), width=1050))
    ''')

    add('markdown', '''
    ## Paso 8 — Coste computacional comparable en GPU (parámetros, VRAM pico y latencia)
    ''')

    add('code', '''
    from tbxtrain.phase5_diagnostics import (
        benchmark_classifier_cost,
        benchmark_detector_cost,
        count_parameters,
    )

    cost_rows = []
    cost_payload = {'gpu': torch.cuda.get_device_name(0), 'classifiers': {}, 'detector': {}}
    for spec in CONFIG['classifiers']:
        arch = spec['architecture']
        ckpt_path = DRIVE_BASE / 'checkpoints' / spec['run_id'] / 'best.pt'
        model = restore_any_classifier(PROJECT_ROOT, DRIVE_BASE, spec, device='cuda', require_oflex=True)
        p_info = count_parameters(model, ckpt_path)
        bench = benchmark_classifier_cost(model, device='cuda', batch_sizes=(1, 8), warmup=20, iterations=100, image_size=512)
        cost_payload['classifiers'][arch] = {**p_info, 'batches': bench}
        cost_rows.append(dict(
            component='classifier',
            name=arch,
            display_name=spec['display_name'],
            **p_info,
            b1_mean_ms=bench['batch_1']['mean_per_image_ms'],
            b1_p95_ms=bench['batch_1']['p95_batch_ms'],
            b1_vram_mb=bench['batch_1']['peak_vram_mb'],
            b8_img_s=bench['batch_8']['throughput_img_s'],
            b8_vram_mb=bench['batch_8']['peak_vram_mb'],
        ))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    sample_paths = [safe_path(DATASET_ROOT, r['relative_path']) for r in ROWS[:8]]
    det_cost = benchmark_detector_cost(
        DET_WEIGHTS,
        sample_paths,
        dict(DET_SPEC['inference_options'], device=0),
        batch_sizes=(1, 8),
        warmup=10,
        iterations=40,
    )
    cost_payload['detector'] = det_cost
    cost_rows.append(dict(
        component='detector',
        name='yolo11n',
        display_name='YOLO11n',
        total_params=det_cost['total_params'],
        protocol_trainable_params=det_cost['total_params'],
        checkpoint_mb=det_cost['checkpoint_mb'],
        b1_mean_ms=det_cost['batches']['batch_1']['mean_per_image_ms'],
        b1_p95_ms=det_cost['batches']['batch_1']['p95_batch_ms'],
        b1_vram_mb=det_cost['batches']['batch_1']['peak_vram_mb'],
        b8_img_s=det_cost['batches']['batch_8']['throughput_img_s'],
        b8_vram_mb=det_cost['batches']['batch_8']['peak_vram_mb'],
    ))

    # Estimación aditiva de pipeline secuencial (clasificador puntero + detector YOLO11n).
    for arch in POINTER_ARCHS:
        c = cost_payload['classifiers'][arch]
        cost_rows.append(dict(
            component='pipeline_sequential',
            name=f'{arch}+yolo11n',
            display_name=f'{arch} + YOLO11n',
            total_params=c['total_params'] + det_cost['total_params'],
            protocol_trainable_params=(c['protocol_trainable_params'] or 0) + det_cost['total_params'],
            checkpoint_mb=c['checkpoint_mb'] + det_cost['checkpoint_mb'],
            b1_mean_ms=c['batches']['batch_1']['mean_per_image_ms'] + det_cost['batches']['batch_1']['mean_per_image_ms'],
            b1_p95_ms=None,
            b1_vram_mb=max(c['batches']['batch_1']['peak_vram_mb'], det_cost['batches']['batch_1']['peak_vram_mb']),
            b8_img_s=1000.0 / (c['batches']['batch_8']['mean_per_image_ms'] + det_cost['batches']['batch_8']['mean_per_image_ms']),
            b8_vram_mb=max(c['batches']['batch_8']['peak_vram_mb'], det_cost['batches']['batch_8']['peak_vram_mb']),
        ))

    write_csv(OUT / 'computational_cost.csv', cost_rows)
    atomic_json(OUT / 'computational_cost.json', cost_payload)
    display(pd.DataFrame(cost_rows))
    ''')

    add('markdown', '''
    ## Paso 9 — Incertidumbre por bootstrap estratificado y agrupado por `group_id`
    ''')

    add('code', '''
    from tbxtrain.phase5_diagnostics import detector_group_bootstrap, group_bootstrap_metrics

    boot_cfg = CONFIG['bootstrap']
    cls_boot = group_bootstrap_metrics(
        ROWS,
        PREDICTIONS_BY_ARCH,
        thresholds=(0.1, 0.3, 0.5),
        pointer_archs=POINTER_ARCHS,
        n_resamples=boot_cfg['n_resamples'],
        seed=boot_cfg['seed'],
        ci_level=boot_cfg['ci_level'],
    )
    det_boot = detector_group_bootstrap(
        ROWS,
        DET_PREDICTIONS,
        thresholds=(0.1, 0.25, 0.5),
        match_iou=0.5,
        n_resamples=boot_cfg['n_resamples'],
        seed=boot_cfg['seed'],
        ci_level=boot_cfg['ci_level'],
    )

    cls_ci_rows = []
    for arch, payload in cls_boot['models'].items():
        for metric_name, st in payload['metrics'].items():
            cls_ci_rows.append(dict(architecture=arch, scope='argmax', metric=metric_name, **st))
        for t_key, t_payload in payload['thresholds'].items():
            for metric_name in ('sensitivity', 'specificity', 'precision'):
                cls_ci_rows.append(dict(architecture=arch, scope=f'p_TB>={t_key}', metric=metric_name, **t_payload[metric_name]))

    delta_rows = []
    for pair_name, metrics_map in cls_boot['paired_deltas'].items():
        for metric_name, st in metrics_map.items():
            delta_rows.append(dict(pair=pair_name, metric=metric_name, **st))

    det_ci_rows = []
    for t_key, metrics_map in det_boot['thresholds'].items():
        for metric_name, st in metrics_map.items():
            det_ci_rows.append(dict(threshold=t_key, metric=metric_name, **st))

    write_csv(OUT / 'classifier_bootstrap_ci.csv', cls_ci_rows)
    write_csv(OUT / 'classifier_paired_deltas_ci.csv', delta_rows)
    write_csv(OUT / 'detector_bootstrap_ci.csv', det_ci_rows)
    atomic_json(OUT / 'bootstrap_uncertainty.json', dict(classifiers=cls_boot, detector=det_boot))

    display(pd.DataFrame(cls_ci_rows)[pd.DataFrame(cls_ci_rows)['metric'].isin(['accuracy', 'macro_f1', 'tb_recall', 'tb_roc_auc', 'sensitivity'])])
    display(pd.DataFrame(delta_rows))
    display(pd.DataFrame(det_ci_rows))
    ''')

    add('markdown', '''
    ## Paso 10 — Paquete compacto de cierre de Parte 1 (`validation` únicamente)
    ''')

    add('code', '''
    summary_bundle = {
        'run_id': RUN_ID,
        'scope': 'validation_only',
        'decisions_frozen': False,
        'test_evaluated': False,
        'external_evaluated': False,
        'phase5_complete': False,
        'gpu': environment['gpu'],
        'scan_backend': backend,
        'smoke_all_passed': all(r['passed'] for r in smoke_records),
        'explainability_all_finite_nonzero': bool(explain_df['all_finite'].all() and explain_df['all_nonzero_grad'].all()),
        'visual_review_cases_count': len(VISUAL_CASES),
        'detector_thresholds': det_table,
        'concordance_grid': grid_summary,
        'computational_cost': cost_rows,
        'bootstrap_summary': dict(classifiers=cls_boot, detector=det_boot),
    }
    atomic_json(OUT / 'part1_summary_bundle.json', summary_bundle)
    atomic_json(OUT / 'diagnostic_status.json', dict(
        status='phase5_part1_validation_diagnostic_completed',
        run_id=RUN_ID,
        decisions_frozen=False,
        test_evaluated=False,
        external_evaluated=False,
        phase5_complete=False,
        ready_for_freeze_decision=True,
    ))
    print('Parte 1 completada correctamente en:', OUT)
    print('Guarde este notebook ejecutado (Ctrl+S) y comparta la carpeta OUT para congelar contrato y pasar a Parte 2 (test único).')
    ''')

    notebook = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'cells': cells,
        'metadata': {
            'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU',
            'colab': {'name': NAME, 'provenance': []},
        },
    }
    target = ROOT / 'notebooks' / NAME
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Notebook generado:', target)


if __name__ == '__main__':
    main()

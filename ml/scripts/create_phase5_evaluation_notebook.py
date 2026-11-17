"""Genera el notebook 08_fase5_evaluacion_final_test_y_externos.ipynb.

Ejecuta la evaluación única bajo contrato congelado sobre manifests/test.csv,
descarga/verifica Montgomery y Shenzhen para evaluación externa con mapeo
binario explícito y exporta el paquete congelado de despliegue.
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = '08_fase5_evaluacion_final_test_y_externos.ipynb'


def main():
    package = json.loads((ROOT / 'dist/phase5_evaluation_package_info.json').read_text(encoding='utf-8'))
    cells = []

    def add(kind, source):
        cell = dict(
            cell_type=kind,
            id=f'phase5eval-{len(cells)}',
            metadata={},
            source=textwrap.dedent(source).strip().splitlines(keepends=True),
        )
        if kind == 'code':
            compile(''.join(cell['source']), f"{NAME}:{cell['id']}", 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''
    # 08 · Fase 5 (Parte 2): Contrato congelado, evaluación única en `test` y externos

    - **Contrato congelado**: `ViT-B/16` (`p_TB >= 0.30`) + `YOLO11n` (`conf >= 0.25`) + concordancia aditiva en 4 estados.
    - **Evaluación única de `test`**: `evaluate_frozen_test` exige bloqueo de ejecución única (`frozen_test_lock.json`) y nunca reajusta umbrales ni pesos.
    - **Evaluación externa**: `Montgomery` (`138`) y `Shenzhen` (`662`), verificados por SHA-256 oficial NLM y auditados con `0` duplicados frente a `train`/`validation`.
    - **Exportación**: genera `deployment_bundle/` en Drive con contrato y pesos verificados para `api/` y `web/`.
    ''')

    add('markdown', '''
    ## Paso 1 — Montar Drive, verificar paquete aislado y preparar salida final
    ''')

    add('code', '''
    import gc, hashlib, importlib, json, stat, subprocess, sys, uuid, zipfile
    from datetime import datetime, timezone
    from pathlib import Path, PurePosixPath
    from google.colab import drive

    drive.mount('/content/drive')
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    ARCHIVE = DRIVE_BASE / 'tbx11k_fase5_evaluacion_colab.zip'
    EXPECTED_PACKAGE_SHA256 = '__PACKAGE_SHA256__'

    def file_sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    if file_sha256(ARCHIVE) != EXPECTED_PACKAGE_SHA256:
        raise ValueError('ZIP distinto de esta entrega del notebook 08.')

    RUN_ID = 'phase5_final_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    PROJECT_ROOT = Path('/content/tbx11k_phase5_final') / RUN_ID
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
        checksums = json.loads(archive.read('phase5_evaluation_checksums.json'))
        if set(names) != set(checksums) | {'phase5_evaluation_checksums.json'}:
            raise ValueError('Cobertura de checksums incorrecta.')
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Checksum alterado: ' + name)
        archive.extractall(PROJECT_ROOT)

    for name in list(sys.modules):
        if name in {'tbxdata', 'tbxtrain'} or name.startswith(('tbxdata.', 'tbxtrain.')):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

    from tbxtrain.protocol import atomic_json
    FINAL_ROOT = DRIVE_BASE / 'results/phase5_final'
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    OUT = FINAL_ROOT / RUN_ID
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / 'session.json', dict(
        run_id=RUN_ID,
        package_sha256=EXPECTED_PACKAGE_SHA256,
        decisions_frozen=True,
        test_evaluated=False,
        external_evaluated=False,
    ))
    print('Salidas finales:', OUT)
    '''.replace('__PACKAGE_SHA256__', package['sha256']))

    add('markdown', '''
    ## Paso 2 — Dependencias protegidas y entorno GPU
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

    import pandas as pd
    from IPython.display import Image as IPyImage, display

    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    atomic_json(OUT / 'environment.json', dict(
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
    ))
    print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
    ''')

    add('markdown', '''
    ## Paso 3 — Verificar contrato congelado y preparar `TBX11K` en caché local
    ''')

    add('code', '''
    from tbxdata.colab import stage_dataset
    from tbxtrain.phase5_evaluation import verify_frozen_contract

    CONTRACT = verify_frozen_contract(PROJECT_ROOT)
    print('Contrato verificado:', CONTRACT['classifier']['architecture'],
          '| p_TB >=', CONTRACT['classifier']['operational_tb_threshold'],
          '| detector conf >=', CONTRACT['detector']['confidence_threshold'])

    source = DRIVE_BASE / 'TBX11K.zip'
    if not source.exists():
        source = DRIVE_BASE / 'TBX11K'
    DATASET_ROOT = stage_dataset(source, Path('/content/tbx11k_phase5_data_cache'), True)
    print('Dataset en caché:', DATASET_ROOT)
    ''')

    add('markdown', '''
    ## Paso 4 — Evaluación única sobre `manifests/test.csv` (`1.241` imágenes)
    ''')

    add('code', '''
    from tbxtrain.phase5_evaluation import evaluate_frozen_test

    TEST_SUMMARY = evaluate_frozen_test(
        PROJECT_ROOT,
        DATASET_ROOT,
        DRIVE_BASE,
        CONTRACT,
        OUT / 'test_evaluation',
        device='cuda',
        batch_size=8,
        lock_dir=FINAL_ROOT,
    )
    cls_m = TEST_SUMMARY['classifier_metrics']
    op_m = TEST_SUMMARY['classifier_operational_threshold']
    det_m = TEST_SUMMARY['detector_operational_point']
    conc_m = TEST_SUMMARY['concordance_summary']
    print(f"TEST argmax -> Accuracy={cls_m['accuracy']:.4f} | Macro-F1={cls_m['macro_f1']:.4f} | TB Recall={cls_m['per_class']['TB']['recall']:.4f} | TB AUC={cls_m['per_class']['TB']['roc_auc']:.4f}")
    print(f"TEST umbral p_TB>=0.30 -> Sens={op_m['sensitivity']:.4f} | Spec={op_m['specificity']:.4f} | Prec={op_m['precision']:.4f}")
    print(f"TEST detector conf>=0.25 -> LesionRecall={det_m['lesion_recall']:.4f} | AnyBox={det_m['image_sensitivity_any_box']:.4f} | Loc={det_m['image_sensitivity_localized']:.4f} | NegAlert={det_m['negative_image_alert_rate']:.4f}")
    print(f"TEST concordancia combinada -> TB Sens={conc_m['combined_tb_sensitivity']:.4f} | No-TB review={conc_m['non_tb_review_rate']:.4f} | Estados={conc_m['state_counts']}")

    for fname in ['test_confusion_matrix.png', 'test_roc_pr.png', 'test_concordance.png']:
        display(IPyImage(filename=str(OUT / 'test_evaluation' / fname), width=950))
    ''')

    add('markdown', '''
    ## Paso 5 — Auditoría y evaluación externa (`Montgomery` y `Shenzhen`)
    ''')

    add('code', '''
    from tbxdata.common import read_csv
    from tbxtrain.phase5_evaluation import evaluate_frozen_external, stage_verified_external

    external_drive = DRIVE_BASE / 'external_data'
    if external_drive.is_dir():
        EXTERNAL_ROOT = external_drive
    else:
        EXTERNAL_ROOT = stage_verified_external(PROJECT_ROOT, Path('/content/tbx11k_external_cache'))

    overlap_audit = {
        'summary': json.loads((PROJECT_ROOT / 'reports/external_sources/train_val_overlap_audit.json').read_text(encoding='utf-8')),
        'rows': read_csv(PROJECT_ROOT / 'reports/external_sources/train_val_overlap_audit.csv'),
    }
    EXTERNAL_SUMMARY = evaluate_frozen_external(
        PROJECT_ROOT,
        EXTERNAL_ROOT,
        DRIVE_BASE,
        CONTRACT,
        OUT / 'external_evaluation',
        overlap_audit=overlap_audit,
        device='cuda',
        batch_size=8,
    )
    for ds, payload in EXTERNAL_SUMMARY['datasets'].items():
        clean = payload['clean_vs_train_val_subset']
        print(f"EXTERNO {ds} (sin solapamiento train/val) ->",
              'p_TB>=0.30:', clean['p_TB_frozen_threshold'],
              '| concordancia:', clean['concordance_review_required'],
              '| ref 1-p_health:', clean['p_abnormal_1_minus_health_ref_0_50'])
    ''')

    add('markdown', '''
    ## Paso 6 — Exportar paquete de despliegue congelado (`api/` y `web/`) y cerrar Fase 5
    ''')

    add('code', '''
    from tbxtrain.phase5_evaluation import export_deployment_bundle

    DEPLOY_MANIFEST = export_deployment_bundle(PROJECT_ROOT, DRIVE_BASE, CONTRACT, OUT / 'deployment_bundle')
    atomic_json(OUT / 'final_status.json', dict(
        status='phase5_completed',
        run_id=RUN_ID,
        decisions_frozen=True,
        test_evaluated=True,
        external_evaluated=True,
        phase5_complete=True,
        deployment_manifest=DEPLOY_MANIFEST,
    ))
    print('Fase 5 completada y paquete de despliegue exportado en:', OUT / 'deployment_bundle')
    print('Guarde este notebook ejecutado (Ctrl+S).')
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

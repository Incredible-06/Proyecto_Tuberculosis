"""Genera el notebook independiente 04b para entrenamiento controlado de ViT-B/16."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = '04b_entrenamiento_vit_comparacion.ipynb'


def main():
    cells = []

    def add(kind, source):
        cell = dict(
            cell_type=kind,
            id='vit-' + str(len(cells)),
            metadata={},
            source=textwrap.dedent(source).strip().splitlines(keepends=True),
        )
        if kind == 'code':
            compile(''.join(cell['source']), NAME + ':' + cell['id'], 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    package = json.loads((ROOT / 'dist/vit_package_info.json').read_text(encoding='utf-8'))

    add('markdown', '''
    # 04b · Extensión Fase 3: entrenamiento controlado de ViT-B/16

    Este notebook agrega un cuarto clasificador (**ViT-B/16**, `torchvision.models.vit_b_16`,
    pesos oficiales `IMAGENET1K_V1` con interpolación bicúbica de embeddings posicionales
    de 224 a 512 px) bajo el **mismo protocolo controlado** que ConvNeXt-Tiny, VMamba-Tiny
    y el Híbrido de atención cruzada:

    - Mismas tres clases (`health=0`, `sick=1`, `TB=2`) y particiones congeladas (`5.792 / 1.241 / 1.241`).
    - Misma entrada RGB 512×512 con letterbox y normalización ImageNet (`configs/data.json`).
    - Misma `seed=20261019`, hasta 12 épocas, 2 de warmup (`head_lr=3e-4`), ajuste de los últimos
      bloques (`backbone_lr=5e-6`), lote efectivo 32 (`batch_size=2`, `accumulation_steps=16`),
      FP32, paciencia 4, `min_delta=0.001` y selección por `validation_macro_f1`.
    - **El conjunto de prueba (`test`) y externos permanecen reservados.**

    Al finalizar, la última celda exporta las figuras (PNG 300 dpi, PDF, SVG), tablas CSV y
    el resumen JSON en el formato exacto del informe de Fase 5 para actualizar
    `Informe_Avance_Fase_5.docx`.
    ''')

    add('markdown', '''
    ## Paso 1 — Montar Drive, verificar paquete ZIP y preparar dataset

    Coloque `tbx11k_vit_entrenamiento_colab.zip` en `MyDrive/Proyecto_Tuberculosis/`
    (junto a `TBX11K.zip` o `TBX11K/`). No sobrescribe los experimentos anteriores.
    ''')

    add('code', '''
    import hashlib, importlib.metadata as metadata, json, stat, subprocess, sys, zipfile
    from pathlib import Path, PurePosixPath
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    ARCHIVE = DRIVE_BASE / 'tbx11k_vit_entrenamiento_colab.zip'
    EXPECTED_PACKAGE_SHA256 = '__PACKAGE_SHA256__'
    def file_sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    if file_sha256(ARCHIVE) != EXPECTED_PACKAGE_SHA256:
        raise ValueError('ZIP diferente de esta versión del notebook. Use ambos archivos de la misma entrega.')
    PROJECT_ROOT = Path('/content/tbx11k_vit/project')
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('ZIP con nombres duplicados.')
        for item in archive.infolist():
            rel = PurePosixPath(item.filename)
            target = (PROJECT_ROOT / item.filename).resolve()
            if (rel.is_absolute() or '..' in rel.parts or ':' in item.filename
                    or chr(92) in item.filename or stat.S_ISLNK(item.external_attr >> 16)
                    or PROJECT_ROOT.resolve() not in target.parents):
                raise ValueError('Ruta ZIP insegura: ' + item.filename)
        checksums = json.loads(archive.read('vit_package_checksums.json'))
        if set(names) != set(checksums) | {'vit_package_checksums.json'}:
            raise ValueError('Cobertura de checksums incorrecta.')
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archivo alterado en el paquete: ' + name)
        archive.extractall(PROJECT_ROOT)
    for name in list(sys.modules):
        if name in {'tbxdata', 'tbxtrain'} or name.startswith(('tbxdata.', 'tbxtrain.')):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    print('Paquete verificado y extraído en:', PROJECT_ROOT)
    '''.replace('__PACKAGE_SHA256__', package['sha256']))

    add('code', '''
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Seleccione un runtime GPU de Colab (T4/L4/A100).')
    protected = {}
    for name in ['torch', 'torchvision', 'torchaudio']:
        try:
            protected[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    constraints = PROJECT_ROOT / 'runtime_constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\\n' for k, v in protected.items()))
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
        '-c', str(constraints), '-r', str(PROJECT_ROOT / 'requirements-training.txt'),
    ])
    if any(metadata.version(k) != v for k, v in protected.items()):
        raise RuntimeError('Se alteró torch/torchvision del runtime. Reinicie antes de continuar.')
    from tbxdata.colab import stage_dataset
    from tbxdata.validate import validate_manifests
    from tbxtrain.vit_experiment import validate_vit_config
    source = DRIVE_BASE / 'TBX11K.zip'
    if not source.exists():
        source = DRIVE_BASE / 'TBX11K'
    DATASET_ROOT = stage_dataset(source, Path('/content/tbx11k_vit/data_cache'), True)
    integrity = validate_manifests(PROJECT_ROOT, DATASET_ROOT, all_image_hashes=True)
    CONFIG_VIT = validate_vit_config(json.loads((PROJECT_ROOT / 'configs/vit_controlled.json').read_text()), PROJECT_ROOT)
    print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__, '| Integridad:', integrity['status'])
    print('Hashes verificados:', integrity['all_image_hashes_verified'])
    print(json.dumps(CONFIG_VIT, indent=2, ensure_ascii=False))
    ''')

    add('markdown', '''
    ## Paso 2 — Smoke técnico de ViT-B/16 (`RUN_SMOKES = True`)

    Una imagen por clase (`health`, `sick`, `TB`) a 512 px verifica descarga oficial de pesos,
    interpolación de embeddings posicionales, forward/backward, actualización de la cabeza y
    reproducibilidad exacta al recargar `best.pt`. No aporta métricas de comparación.
    ''')

    add('code', '''
    from tbxtrain.vit_experiment import run_vit_model
    RUN_SMOKES = True
    if RUN_SMOKES:
        smoke = run_vit_model(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, CONFIG_VIT, smoke=True)
        print('Smoke ViT-B/16:', smoke['status'], '| updates:', smoke['optimizer_updates'])
    ''')

    add('markdown', '''
    ## Paso 3 — Entrenamiento completo controlado de ViT-B/16 y recuperación automática

    Guarda `last.pt` e `history.csv` al finalizar cada época en Drive. Si la sesión se
    interrumpe, vuelva a ejecutar desde el Paso 1 (puede poner `RUN_SMOKES = False`) y esta
    celda retomará automáticamente la última época completa desde `results/vit_suite/`.
    ''')

    add('code', '''
    marker = DRIVE_BASE / 'results/vit_suite' / CONFIG_VIT['protocol_id'] / 'vit_b_16.json'
    resume = None
    VIT_RUN_ID = None
    if marker.exists():
        prior = json.loads(marker.read_text())
        prior_id = prior['run_id']
        summary_path = DRIVE_BASE / 'experiments' / prior_id / 'summary.json'
        if summary_path.exists():
            prior_summary = json.loads(summary_path.read_text())
            if prior_summary['status'] == 'completed_controlled':
                if prior_summary['protocol'] != CONFIG_VIT:
                    raise RuntimeError('El protocolo cambió. Defina otro protocol_id para una nueva tanda.')
                VIT_RUN_ID = prior_id
                print('Corrida recuperada sin reentrenar:', VIT_RUN_ID)
        if VIT_RUN_ID is None and (DRIVE_BASE / 'checkpoints' / prior_id / 'last.pt').exists():
            resume = prior_id
    if VIT_RUN_ID is None:
        vit_summary = run_vit_model(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, CONFIG_VIT, resume_run=resume, smoke=False)
        VIT_RUN_ID = vit_summary['run_id']
    print('Corrida ViT-B/16 completa:', VIT_RUN_ID)
    ''')

    add('markdown', '''
    ## Paso 4 — Exportar figuras, tablas y resumen para el informe de Fase 5

    Genera las figuras de entrenamiento, matriz de confusión, ROC OVR, umbrales TB de ViT-B/16
    y las figuras comparativas de los **4 clasificadores** (`ConvNeXt-Tiny`, `VMamba-Tiny`,
    `Híbrido`, `ViT-B/16`) con el mismo estilo de `Informe_Avance_Fase_5.docx`.

    Todas las métricas, tablas y gráficos quedan también impresos e incrustados en las salidas
    de esta celda. **Guarde el notebook con Ctrl+S y descárguelo (`.ipynb`) para actualizar el informe.**
    ''')

    add('code', '''
    import pandas as pd
    from datetime import datetime, timezone
    from IPython.display import Image, display
    from tbxtrain.protocol import atomic_json
    from tbxtrain.vit_experiment import export_vit_report_outputs

    PRIOR_RUNS = {
        'convnext_tiny': 'phase3_convnext_tiny_20261103T023002Z_4a7e2a',
        'vmamba_tiny': 'phase3_vmamba_tiny_20261103T025536Z_4784e4',
        'hybrid_cross_attention': 'phase3_hybrid_cross_attention_20261103T032205Z_71d127',
    }
    OUT_EXPORT = DRIVE_BASE / 'results/vit_phase3_export' / (
        VIT_RUN_ID + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    )
    payload = export_vit_report_outputs(PROJECT_ROOT, DRIVE_BASE, VIT_RUN_ID, OUT_EXPORT, prior_runs=PRIOR_RUNS)
    atomic_json(OUT_EXPORT / 'transfer_integrity.json', integrity)

    print('=== COMPARACIÓN DE LOS 4 CLASIFICADORES (VALIDATION) ===')
    display(pd.DataFrame(payload['comparison_4models'])[
        ['display_name', 'run_id', 'completed_epochs', 'best_epoch', 'stop_reason',
         'accuracy', 'macro_f1', 'TB_recognized', 'TB_recall', 'TB_precision', 'TB_specificity']
    ])

    print('=== HISTORIAL POR ÉPOCA · ViT-B/16 ===')
    display(pd.DataFrame(payload['vit_details']['history']))

    print('=== TABLA DE UMBRALES TB · ViT-B/16 (VALIDATION) ===')
    display(pd.DataFrame(payload['vit_details']['thresholds_grid']))

    print('=== TABLA DE UMBRALES TB · CANDIDATO PRINCIPAL POR MACRO-F1 ===', payload['candidate_by_macro_f1'])
    display(pd.DataFrame(payload['candidate_thresholds_grid']))

    for fig_name in [
        'classifier_comparison.png',
        'vit_b_16_training.png',
        'vit_b_16_confusion.png',
        'vit_b_16_roc.png',
        'TB_roc_precision_recall.png',
        'vit_b_16_thresholds.png',
    ]:
        print(fig_name)
        display(Image(filename=str(OUT_EXPORT / fig_name), width=1000))

    print('=== PAYLOAD JSON PARA ACTUALIZAR INFORME FASE 5 ===')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print('Carpeta exportada en Drive:', OUT_EXPORT)
    print('Guarde este notebook ejecutado con Ctrl+S y envíe el .ipynb (y/o esta carpeta de Drive).')
    ''')

    notebook = dict(
        nbformat=4,
        nbformat_minor=5,
        cells=cells,
        metadata=dict(
            kernelspec=dict(name='python3', language='python', display_name='Python 3'),
            language_info=dict(name='python'),
            accelerator='GPU',
            colab=dict(name=NAME, provenance=[]),
        ),
    )
    target = ROOT / 'notebooks' / NAME
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if any(c.get('outputs') or c.get('execution_count') is not None for c in previous['cells'] if c['cell_type'] == 'code'):
            raise RuntimeError('Notebook 04b ya ejecutado: conservar outputs; generar otra versión con otro nombre.')
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(NAME, len(cells), 'celdas compiladas, sin outputs simulados.')


if __name__ == '__main__':
    main()

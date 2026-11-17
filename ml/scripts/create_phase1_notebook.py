"""Construye un notebook v4, con celdas de Python válidas también para revisión AST."""
import json
import textwrap
from pathlib import Path


def main():
    cells=[]
    def md(text):
        cells.append({'cell_type':'markdown','id':f'cell-{len(cells):03d}','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code','id':f'cell-{len(cells):03d}','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('''
    # 01 · Exploración y preprocesamiento de TBX11K

    **Fase 2 del plan académico — preparación de datos, sin entrenamiento.**
    La Fase 1 corresponde al notebook 00 de entorno. Clases `health=0`, `sick=1`, `TB=2`.
    Se reutiliza la auditoría local y los manifiestos congelados. Este notebook no
    descarga TBX11K, pesos o modelos. Para esta fase basta un entorno CPU de Colab.

    Ejecute las celdas en orden. Suba a Drive `tbx11k_fases_1_y_2_colab.zip` (código y
    manifiestos) y, por separado, su carpeta `TBX11K/` o un ZIP de ella. Los archivos
    originales del dataset permanecen intactos. El notebook usa preferentemente
    `/content` y guarda las evidencias de esta ejecución en Drive.

    Los resultados de entrenamiento y la compatibilidad de modelos/GPU pertenecen
    a una fase posterior. Este notebook no los valida.
    ''')
    md('''
    ## 1. Rutas y controles de ejecución

    Estructura esperada en Drive:

    ```text
    MyDrive/Proyecto_Tuberculosis/
      tbx11k_fases_1_y_2_colab.zip
      TBX11K.zip              # alternativa preferida si existe
      TBX11K/                 # o la carpeta con imgs/, lists/, annotations/
      results/                # se crea al exportar resultados
    ```

    Ajuste `DRIVE_BASE` si usa otra ubicación. Un ZIP puede contener `TBX11K/` o
    directamente `imgs/`, `lists/` y `annotations/`. No suba el ZIP del proyecto
    dentro del dataset. `DATA_SOURCE_KIND='auto'` prefiere el ZIP existente.
    ''')
    code('''
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path
    from datetime import datetime, timezone

    IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    PROJECT_ARCHIVE = DRIVE_BASE / 'tbx11k_fases_1_y_2_colab.zip'
    DATA_SOURCE_KIND = 'auto'  # 'auto', 'zip' o 'folder'
    COPY_DATASET_TO_LOCAL = True
    VERIFY_ALL_IMAGE_HASHES = IN_COLAB  # comprueba íntegramente cada transferencia en Colab
    REBUILD_MANIFESTS_FOR_COMPARISON = False  # solo compara, no sobreescribe los congelados
    RUN_COLAB_DATALOADER_CHECK = True  # un lote CPU; sin modelo ni entrenamiento
    LOCAL_WORK = Path('/content/tbx11k_phase1')
    # Si se revisa localmente, se usan exclusivamente datos y rutas del proyecto:
    LOCAL_PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'src/tbxdata').exists() else Path.cwd().parent
    RUN_NAME = 'phase1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    print('Google Colab:', IN_COLAB)
    ''')
    md('''
    ## 2. Montar Drive y cargar el proyecto pequeño

    El montaje solicita autorización de Google en Colab. La extracción valida las
    rutas del ZIP y rechaza enlaces simbólicos. Puede volver a ejecutar tras una
    desconexión: el código se extrae nuevamente; no se borra el dataset de Drive.
    ''')
    code('''
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        import shutil
        import stat
        import zipfile
        from pathlib import PurePosixPath
        if not PROJECT_ARCHIVE.is_file():
            raise FileNotFoundError(f'Suba primero el ZIP del proyecto: {PROJECT_ARCHIVE}')
        PROJECT_ROOT = LOCAL_WORK / 'project'
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(PROJECT_ARCHIVE) as archive:
            for item in archive.infolist():
                rel = PurePosixPath(item.filename)
                target = (PROJECT_ROOT / item.filename).resolve()
                if rel.is_absolute() or '..' in rel.parts or '\\\\' in item.filename or ':' in item.filename or PROJECT_ROOT.resolve() not in target.parents:
                    raise ValueError('Ruta no segura en el ZIP del proyecto: ' + item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('Enlace no permitido en el ZIP del proyecto.')
            archive.extractall(PROJECT_ROOT)
    else:
        PROJECT_ROOT = LOCAL_PROJECT_ROOT.resolve()
        if not (PROJECT_ROOT / 'configs/data.json').is_file():
            raise FileNotFoundError('Ejecute desde la raíz del proyecto o su carpeta notebooks/.')
    print('Proyecto:', PROJECT_ROOT)
    ''')
    md('''
    ## 3. Dependencias ligeras y entorno

    Solo NumPy y Pillow (`requirements-data.txt`). Se conservan versiones instaladas
    que cumplan los rangos y se registra el entorno exacto al finalizar. Las pruebas
    locales se hicieron con Python 3.14.7, NumPy 2.5.4 y Pillow 12.3.0. Este notebook
    evita reinstalar PyTorch o CUDA en Colab. Una GPU es opcional para esta fase.
    ''')
    code('''
    if IN_COLAB:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                               '--disable-pip-version-check', '-r', str(PROJECT_ROOT / 'requirements-data.txt')])
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    import platform
    import numpy as np
    import PIL
    from PIL import Image
    from tbxdata.dataset import TBXDataset, load_config, class_weights
    from tbxdata.common import read_csv, sha256
    from tbxdata.validate import validate_manifests
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
                   'platform': platform.platform(), 'in_colab': IN_COLAB}
    if IN_COLAB:
        try:
            import torch
        except ImportError:
            print('PyTorch no disponible: puede ejecutar toda la fase de datos y omitir el lote opcional.')
        else:
            environment['torch'] = torch.__version__
            environment['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                environment['gpu'] = {'name': props.name, 'total_memory_gib': round(props.total_memory / 2**30, 2)}
    print(json.dumps(environment, indent=2))
    ''')
    md('''
    ## 4. Usar el dataset local de Colab

    Para reducir lecturas pequeñas en Drive, prefiera el ZIP y extracción temporal.
    Para una carpeta se usa una copia reanudable por tamaño. Un marcador reutiliza
    la copia terminada; la validación posterior siempre vuelve a comprobarla.
    Si la fuente cambió, use otra carpeta `LOCAL_WORK` para conservar el cache previo.
    El almacenamiento `/content` es temporal y se pierde al reiniciar el entorno.
    ''')
    code('''
    from tbxdata.colab import stage_dataset
    if IN_COLAB:
        zip_source = DRIVE_BASE / 'TBX11K.zip'
        folder_source = DRIVE_BASE / 'TBX11K'
        if DATA_SOURCE_KIND not in {'auto', 'zip', 'folder'}:
            raise ValueError('DATA_SOURCE_KIND debe ser auto, zip o folder.')
        source = zip_source if DATA_SOURCE_KIND == 'zip' or (DATA_SOURCE_KIND == 'auto' and zip_source.exists()) else folder_source
        DATASET_ROOT = stage_dataset(source, LOCAL_WORK / 'data_cache', COPY_DATASET_TO_LOCAL)
        RUN_DIR = LOCAL_WORK / 'runs' / RUN_NAME
    else:
        if VERIFY_ALL_IMAGE_HASHES:
            raise RuntimeError('La revisión local reutiliza la auditoría; deje VERIFY_ALL_IMAGE_HASHES=False.')
        DATASET_ROOT = PROJECT_ROOT / 'TBX11K'
        RUN_DIR = PROJECT_ROOT / 'reports' / 'notebook_runs' / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print('Dataset:', DATASET_ROOT)
    print('Evidencias de esta ejecución:', RUN_DIR)
    ''')
    md('''
    ## 5. Auditoría reutilizada y revisión de duplicados

    Se auditaron inicialmente 12.278 imágenes RGB de 512². Hay 8.400 archivos base
    etiquetados y 8.274 contenidos RGB únicos, después de retirar copias exactas
    solo de los manifiestos. El contenido original no se elimina.

    El filtro dHash produjo 13.783 pares. Su refinamiento con pHash DCT de 256 bits,
    SSIM local, correlación y error de píxeles dejó 44 pares, revisados visualmente
    por Codex. Los 44 se agruparon; 23 pares están dentro del conjunto etiquetado.
    Se revisaron además ocho controles rechazados, visualmente distintos. Los
    grupos son por contenido de imagen, **no por paciente**. El detector no es
    exhaustivo: solo refina candidatos de dHash igual de la auditoría inicial.

    Las evidencias y decisiones se conservan en `reports/near_duplicates/`.
    ''')
    code('''
    audit = json.loads((PROJECT_ROOT / 'reports/initial_audit/audit_summary.json').read_text())
    review = json.loads((PROJECT_ROOT / 'reports/near_duplicates/method_summary.json').read_text())
    manifest_meta = json.loads((PROJECT_ROOT / 'manifests/manifest_metadata.json').read_text())
    print('Dataset auditado:', audit['images'], audit['dimensions'], audit['modes'])
    print('Refinamiento de candidatos:', json.dumps(review, indent=2))
    print('Clases:', manifest_meta['classes'])
    print('Conteos congelados:', json.dumps(manifest_meta['counts'], indent=2))
    ''')
    md('''
    ## 6. Reproducibilidad de las particiones

    Seed 20261019. Cuotas 70/15/15 por clase con redondeo por restos mayores y grupos
    de contenido indivisibles. Se conserva un representante determinista por hash
    RGB exacto; variantes cercanas se mantienen como observaciones del mismo grupo.
    Los 8.274 contenidos están distribuidos en 8.251 grupos.

    `train.csv`: 5.792; `validation.csv`: 1.241; `test.csv`: 1.241.
    El test interno se reserva para evaluación final, sin imágenes de prueba en
    las visualizaciones ni elección de transformaciones. Las listas oficiales se
    conservan como metadatos; este protocolo propio no reproduce sus resultados.

    Rehacer es opcional: se compara contra la versión congelada, sin reemplazarla.
    ''')
    code('''
    if REBUILD_MANIFESTS_FOR_COMPARISON:
        from tbxdata.prepare import build_manifests
        rebuilt = build_manifests(PROJECT_ROOT, DATASET_ROOT, RUN_DIR / 'rebuilt_manifests')
        if rebuilt != manifest_meta:
            raise ValueError('La reproducción no coincide. No use los manifiestos hasta revisar entradas y código.')
        print('Reproducción idéntica de metadatos y hashes CSV.')
    else:
        print('Se usan los manifiestos existentes; no se regeneran particiones por defecto.')
    ''')
    md('''
    ## 7. Verificaciones de integridad

    Comprueba cobertura de las 8.400 rutas originales, representantes, etiquetas,
    listas, cajas COCO originales y ausencia de cruces internos por ruta, hash y
    grupo. Compara existencia/tamaño de todos los archivos con el inventario y
    SHA-256 de una muestra reproducible de cada clase/partición.

    Para verificar íntegramente una transferencia nueva, cambie
    `VERIFY_ALL_IMAGE_HASHES=True` **en Colab**. Esa opción lee los archivos para
    comparar los hashes ya calculados; no recalcula la exploración o duplicados.
    No se infieren etiquetas del test oficial. Las comprobaciones fallan con una
    excepción descriptiva ante una copia incompleta, metadatos o manifiestos cambiados.
    ''')
    code('''
    integrity = validate_manifests(PROJECT_ROOT, DATASET_ROOT, hash_samples_per_class=3,
                                   all_image_hashes=VERIFY_ALL_IMAGE_HASHES)
    (RUN_DIR / 'integrity.json').write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(integrity, indent=2, ensure_ascii=False))
    ''')
    md('''
    ## 8. Lector y configuración centralizada

    `configs/data.json` fija clases, seed, tamaño, normalización y aumentos. La
    normalización ImageNet usa media [0.485, 0.456, 0.406] y desviación
    [0.229, 0.224, 0.225] sobre RGB en [0,1]. Coincide con la normalización de
    ConvNeXt-Tiny de torchvision; el resize/crop oficial de 224 no se adopta aquí:
    este protocolo mantiene el tórax completo a 512². Antes del entrenamiento se
    verificarán los requisitos de los pesos concretos de ambos modelos.

    Train: brillo/contraste ±8 % y rotación ±5°, con expansión y padding para
    conservar el campo. Sin espejado, recorte agresivo, deformaciones ni borrado.
    CLAHE desactivado: queda como experimento posterior con validación.
    Validation/test: lectura RGB y normalización determinista, sin aumentos.

    No cambie la configuración congelada para explorar hiperparámetros: guarde una
    nueva versión del protocolo y nunca seleccione aumentos usando test.
    ''')
    code('''
    config = load_config(PROJECT_ROOT / 'configs/data.json')
    train_ds = TBXDataset(DATASET_ROOT, PROJECT_ROOT / 'manifests/train.csv', config)
    validation_ds = TBXDataset(DATASET_ROOT, PROJECT_ROOT / 'manifests/validation.csv', config)
    weights = class_weights(PROJECT_ROOT / 'manifests/train.csv', config)
    print('Train / validation:', len(train_ds), len(validation_ds))
    print('Pesos de clase calculados solo en train:', dict(zip(config['classes'], weights.tolist())))
    print('No se carga el test interno para visualizar imágenes o ajustar transformaciones.')
    ''')
    md('''
    ## 9. Distribución, muestras y cajas originales

    Las etiquetas y el tamaño de cada partición se muestran para verificar la
    estratificación. Las radiografías de ejemplo proceden exclusivamente de train.
    Las cajas son las COCO originales verificadas, reescaladas solo para dibujar.
    Rojo: ActiveTuberculosis; cian: ObsoletePulmonaryTuberculosis.
    ''')
    code('''
    from tbxdata.visualize import distribution, samples, transformations
    def show_figure(path):
        if IN_COLAB:
            from IPython.display import display
            with Image.open(path) as image:
                display(image.copy())
        else:
            print('Figura guardada:', Path(path).name)
    FIGURES = RUN_DIR / 'figures'
    FIGURES.mkdir(parents=True, exist_ok=True)
    distribution(PROJECT_ROOT, FIGURES / 'class_distribution.png')
    samples(PROJECT_ROOT, DATASET_ROOT, FIGURES / 'training_samples.png')
    show_figure(FIGURES / 'class_distribution.png')
    show_figure(FIGURES / 'training_samples.png')
    ''')
    md('''
    ## 10. Comprobar visualmente aumentos moderados

    Cada fila contiene una imagen de train y tres variantes ilustrativas con seeds
    fijadas. Se muestran antes de normalizar. Las cajas no se dibujan sobre imágenes
    aumentadas: el lector de clasificación no transforma cajas para detección.
    ''')
    code('''
    transformations(PROJECT_ROOT, DATASET_ROOT, FIGURES / 'training_transformations.png')
    show_figure(FIGURES / 'training_transformations.png')
    ''')
    md('''
    ## 11. Prueba ligera del pipeline de datos

    Tres ejemplos de train y tres de validation. Comprueba canales, tamaño, dtype,
    valores finitos, repetibilidad y cambios entre épocas solo en train. No hay
    modelo, gradientes, optimizador o entrenamiento.
    ''')
    code('''
    checked_samples = []
    for name, dataset in [('train', train_ds), ('validation', validation_ds)]:
        for cls in config['classes']:
            idx = next(i for i, row in enumerate(dataset.rows) if row['class_name'] == cls)
            a, label = dataset[idx]
            assert a.shape == (3, 512, 512) and a.dtype == np.float32
            assert np.isfinite(a).all() and label == config['classes'][cls]
            np.testing.assert_array_equal(a, dataset[idx][0])
            dataset.set_epoch(1)
            b, _ = dataset[idx]
            assert np.array_equal(a, b) if name == 'validation' else not np.array_equal(a, b)
            dataset.set_epoch(0)
            checked_samples.append({'split': name, 'class': cls, 'relative_path': dataset.rows[idx]['relative_path']})
    print('Seis muestras verificadas; no se ejecutó entrenamiento.')
    ''')
    md('''
    ## 12. Un lote PyTorch CPU, únicamente en Colab (opcional)

    La clase implementa el protocolo de dataset de PyTorch; `DataLoader` convierte
    automáticamente arrays NumPy a tensores. Solo se lee un lote CPU de cuatro
    imágenes. No es un smoke test de entrenamiento ni una validación CUDA.
    En una fase posterior, `set_epoch(epoch)` debe preceder al iterador, usando
    `persistent_workers=False` para propagar el estado de época a los workers.
    ''')
    code('''
    dataloader_check = {'executed': False, 'reason': 'Solo se ejecuta en Google Colab.'}
    if IN_COLAB and RUN_COLAB_DATALOADER_CHECK:
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            dataloader_check['reason'] = 'PyTorch no está instalado; no se instala automáticamente.'
        else:
            import torch
            from torch.utils.data import DataLoader
            generator = torch.Generator().manual_seed(config['seed'])
            loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0,
                                generator=generator, pin_memory=False, persistent_workers=False)
            images, labels = next(iter(loader))
            assert tuple(images.shape) == (4, 3, 512, 512)
            assert images.dtype == torch.float32 and labels.dtype == torch.int64
            assert images.device.type == 'cpu' and torch.isfinite(images).all()
            dataloader_check = {'executed': True, 'shape': list(images.shape), 'device': 'cpu'}
            del images, labels, loader
    print(dataloader_check)
    ''')
    md('''
    ## 13. Anotaciones y reservas para fases posteriores

    Los subtipos de TB y las cajas quedan en columnas JSON separadas. No se impone
    una clasificación exclusiva por subtipo: 30 imágenes tienen ambos tipos. La
    imagen `tb1199` conserva etiqueta TB y bandera `missing_tb_boxes`; no debe
    tratarse como negativo de un detector. Los XML/COCO de origen no se reescriben.

    `official_test_unlabeled.csv` conserva 3.302 imágenes con etiquetas vacías.
    Sus 93 solapamientos de contenido detectados con el conjunto interno se
    registran para trazabilidad, sin propagar etiquetas. Este test no es un test
    supervisado independiente con la información disponible.

    `reserved_external.csv` contiene 576 imágenes excluidas del clasificador:
    400 en la carpeta combinada Montgomery/Shenzhen y 176 DA/DB. No se observó
    solapamiento exacto o cercano confirmado entre esos extras y el conjunto
    interno en el alcance de esta búsqueda. Esto no garantiza independencia.
    Hay duplicados dentro de DA/DB y frente al test oficial, registrados en
    `external_overlap_report.csv`. Una validación externa futura exige etiquetas
    verificadas, procedencia, control de duplicados ampliado y mantener esas
    imágenes fuera de selección de modelos y de aumentos. Los datos mezclados y
    anonimizados del test oficial no permiten reconstruir aquí fuentes completas.
    ''')
    code('''
    reserves = {name: read_csv(PROJECT_ROOT / 'manifests' / f'{name}.csv')
                for name in ['official_test_unlabeled', 'reserved_external']}
    for name, rows in reserves.items():
        assert all(row['label'] == '' and row['class_name'] == '' for row in rows)
        print(name, len(rows), 'filas, sin etiquetas inferidas')
    print('Ejemplo de metadatos separados:', next(train_ds.metadata(i) for i, row in enumerate(train_ds.rows) if row['image_id'] == 'tb1199'))
    ''')
    md('''
    ## 14. Guardar evidencias y recuperar tras desconexión

    Se guarda configuración, manifiestos, decisiones de revisión, versiones y
    gráficas. No hay checkpoints de modelos en esta fase. Tras una desconexión,
    repita desde el montaje: los manifiestos se recuperan del ZIP y el dataset se
    vuelve a copiar al almacenamiento temporal si se perdió. Nunca se regenera
    automáticamente una partición diferente para continuar.
    ''')
    code('''
    import shutil
    environment['manifest_metadata_sha256'] = sha256(PROJECT_ROOT / 'manifests/manifest_metadata.json')
    environment['manifest_csv_sha256'] = manifest_meta['output_sha256']
    (RUN_DIR / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    (RUN_DIR / 'data_checks.json').write_text(json.dumps({'checked_samples': checked_samples,
                                                       'dataloader_check': dataloader_check}, indent=2), encoding='utf-8')
    shutil.copy2(PROJECT_ROOT / 'configs/data.json', RUN_DIR / 'config.json')
    shutil.copy2(PROJECT_ROOT / 'reports/external_sources/preparation_summary.json',
                 RUN_DIR / 'external_preparation_summary.json')
    shutil.copytree(PROJECT_ROOT / 'manifests', RUN_DIR / 'manifests', dirs_exist_ok=True)
    shutil.copy2(PROJECT_ROOT / 'reports/near_duplicates/review_decisions.csv', RUN_DIR / 'review_decisions.csv')
    if IN_COLAB:
        drive_result = DRIVE_BASE / 'results' / RUN_NAME
        drive_result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(RUN_DIR, drive_result)
        print('Evidencias guardadas en Drive:', drive_result)
    else:
        print('Comprobación local guardada en:', RUN_DIR)
    ''')
    md('''
    ## 15. Alcance de la validación y próximos pasos

    Localmente: manifiestos y reproducción idéntica, ausencia de cruces conocidos,
    cajas COCO preservadas, rutas, aumentos sobre pocas imágenes, lectura RGB y
    estructura/ejecución de las celdas de datos. No se repitió la auditoría completa.

    Evidencia histórica revisada: el notebook de datos se ejecutó en Colab CPU
    el 19 de octubre de 2026 a las 23:25 (Lima), con 14 celdas sin errores y un
    lote de cuatro imágenes. El notebook 00 registró A100, CUDA aprobada y
    12.278 hashes completos el 20 de octubre a las 00:11. Los originales se
    conservan separados. Esta revisión añade resúmenes para capturas; requiere
    su propia ejecución en Colab para obtener nuevas capturas de esa interfaz.

    Limitaciones: no hay IDs fiables de paciente, la búsqueda cercana no fue
    exhaustiva y el test interno contiene solo 120 TB. Variantes del mismo grupo
    pueden contar como varias observaciones; en evaluación futura se recomienda
    informar resultados también por grupo y usar remuestreo por grupo para
    incertidumbre. Las decisiones visuales de Codex no constituyen revisión clínica.

    La etapa de modelos comenzará después de revisar este pipeline. Entrenamiento, carga de
    modelos completos, pesos y pruebas CUDA se ejecutarán exclusivamente en Colab.

    Referencias de configuración:
    [normalización de ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
    [lectura y almacenamiento temporal en Colab](https://research.google.com/colaboratory/faq.html).
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.10+'},'colab':{'name':'01_exploracion_preprocesamiento.ipynb','provenance':[]}},
        'nbformat':4,'nbformat_minor':5}
    from refine_notebooks import refine_data
    refine_data(notebook)
    output=Path('notebooks/01_exploracion_preprocesamiento.ipynb')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(output,len(cells),'celdas')


if __name__=='__main__':main()

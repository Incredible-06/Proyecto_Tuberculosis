"""Genera el notebook 02 sin salidas ni métricas inventadas."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'id': f'train-{len(cells):03d}', 'metadata': {},
                'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''
    # 02 · ConvNeXt-Tiny: primer entrenamiento preliminar

    **Avance parcial de la Fase 3.** Se reutilizan las Fases 1 (entorno) y 2 (datos),
    sin reconstruir sus particiones. Este notebook entrena un primer clasificador,
    conserva checkpoints en Drive y permite revisar y ajustar el experimento.
    No completa la comparación ConvNeXt / Vision Mamba / híbrido ni la evaluación final.

    **Estado de esta copia:** código preparado; solo habrá evidencia de entrenamiento
    después de ejecutar la celda F3-E3 y comprobar `training_executed=True`.
    Las métricas se calculan, no se rellenan previamente.

    Ruta del ciclo: **hipótesis → entrenamiento → validación → revisión de errores →
    ajuste documentado → nueva corrida**. Las épocas pertenecen a una misma corrida;
    cada nuevo ajuste genera otra carpeta con configuración, procedencia y resultados.

    ## Antes de empezar

    1. El paquete `tbx11k_fase3_baseline_colab.zip` y `TBX11K.zip` deben estar en
       `MyDrive/Proyecto_Tuberculosis/`. No necesita repetir los notebooks 00/01.
    2. Seleccione **Entorno de ejecución → Cambiar tipo de entorno → GPU**.
       La L4 de su captura sirve para esta corrida; también puede usar T4 o A100
       disponible. Este notebook no solicita compras ni cambia su plan.
    3. Ejecute las celdas en orden y autorice el montaje de Drive cuando Colab lo pida.
    4. Conserve el notebook en Drive y pulse **Ctrl+S** tras la ejecución para guardar
       también sus salidas. Los checkpoints y métricas se escriben en Drive por época.

    **Tres clases verificadas:** `health=0`, `sick=1`, `TB=2`. El plan menciona cuatro,
    pero el protocolo ya preparado agrupa TB en una sola etiqueta; los subtipos
    activos/latentes/mixtos no son cuatro clases mutuamente excluyentes en estos manifiestos.
    No se inventa una cuarta clase. Esa diferencia debe aclararse en el plan.
    ''')
    add('code', '''
    import importlib.util
    import json
    import sys
    from pathlib import Path

    IN_COLAB = bool(importlib.util.find_spec('google') and importlib.util.find_spec('google.colab'))
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    PROJECT_ARCHIVE = DRIVE_BASE / 'tbx11k_fase3_baseline_colab.zip'
    LOCAL_WORK = Path('/content/tbx11k_phase3')
    VERIFY_ALL_IMAGE_HASHES = True
    # Para recuperar una corrida interrumpida: copie solo el nombre de su carpeta.
    # Mantenga intacta su configuración: no es una nueva prueba de hiperparámetros.
    RESUME_RUN = None
    if not IN_COLAB:
        raise RuntimeError('Abra este notebook en Google Colab con GPU. La revisión local usa el verificador separado.')
    ''')
    add('markdown', '''
    ## 1 · Montaje y carga del código

    Se usa un paquete nuevo para conservar intactos los entregables de las Fases 1 y 2.
    El ZIP se extrae a almacenamiento temporal; el dataset de Drive no se modifica.
    ''')
    add('code', '''
    from google.colab import drive
    drive.mount('/content/drive')
    import stat
    import zipfile
    from pathlib import PurePosixPath

    if not PROJECT_ARCHIVE.is_file():
        raise FileNotFoundError(f'Falta el paquete nuevo: {PROJECT_ARCHIVE}')
    PROJECT_ROOT = LOCAL_WORK / 'project'
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PROJECT_ARCHIVE) as archive:
        for item in archive.infolist():
            rel = PurePosixPath(item.filename)
            target = (PROJECT_ROOT / item.filename).resolve()
            if (rel.is_absolute() or '..' in rel.parts or '\\\\' in item.filename or ':' in item.filename
                    or PROJECT_ROOT.resolve() not in target.parents or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Miembro no seguro en el ZIP: ' + item.filename)
        import hashlib
        checksums = json.loads(archive.read('phase3_package_checksums.json'))
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Contenido alterado en el paquete: ' + name)
        archive.extractall(PROJECT_ROOT)
    print('Código:', PROJECT_ROOT)
    ''')
    add('markdown', '''
    ## 2 · Dependencias y GPU · F3-E1

    No se reinstalan PyTorch, TorchVision ni CUDA. Se comprueban los paquetes del
    runtime y se registran sus versiones exactas en cada experimento.
    Si Colab no ofrece GPU, deténgase aquí y vuelva a conectar con GPU.
    ''')
    add('code', '''
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                           '--disable-pip-version-check', '-r', str(PROJECT_ROOT / 'requirements-training.txt')])
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    try:
        import torch
        import torchvision
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    except Exception as exc:
        raise RuntimeError('Runtime torch/torchvision incompatible. Use un runtime GPU nuevo de Colab; conserve sus resultados en Drive.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible. Seleccione GPU en Colab y vuelva a conectar.')
    print('PyTorch:', torch.__version__, '| TorchVision:', torchvision.__version__)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memoria libre GiB:', round(torch.cuda.mem_get_info()[0] / 2**30, 2))
    ''')
    add('markdown', '''
    ## 3 · Reutilizar y verificar datos · F3-E2

    Se conservan 5.792 imágenes de train y 1.241 de validation. El test interno
    (1.241 imágenes), el test oficial sin etiquetas y Montgomery/Shenzhen quedan
    reservados. La integridad puede leer metadatos/hashes de esos archivos;
    **no se crean tensores ni se calculan métricas de test/externos**.

    Se mantiene RGB, 512×512, normalización ImageNet y aumentos suaves solo en train.
    El preset original de los pesos usa resize/crop a 224; aquí se documenta la
    adaptación de resolución para conservar el campo de la radiografía.
    ''')
    add('code', '''
    from tbxdata.colab import stage_dataset
    from tbxdata.dataset import TBXDataset, load_config
    from tbxdata.validate import validate_manifests
    from tbxtrain.protocol import atomic_json

    source = DRIVE_BASE / 'TBX11K.zip'
    if not source.is_file():
        source = DRIVE_BASE / 'TBX11K'
    DATASET_ROOT = stage_dataset(source, LOCAL_WORK / 'data_cache', copy_to_local=True)
    integrity = validate_manifests(PROJECT_ROOT, DATASET_ROOT, hash_samples_per_class=2,
                                   all_image_hashes=VERIFY_ALL_IMAGE_HASHES)
    print('Integridad:', integrity['status'], '| SHA256 completos:', integrity['all_image_hashes_verified'])
    print('Distribución:', json.dumps(integrity['counts'], indent=2))
    DATA_CONFIG = load_config(PROJECT_ROOT / 'configs/data.json')
    validation_sample = TBXDataset(DATASET_ROOT, PROJECT_ROOT / 'manifests/validation.csv', DATA_CONFIG)
    image, label = validation_sample[0]
    print('Tensor de entrada:', image.shape, image.dtype, '| etiqueta:', label)
    ''')
    add('markdown', '''
    ## 4 · Registrar hipótesis y configuración

    **Primera corrida:** tres épocas completas, una con cabeza entrenable y dos con
    cabeza + última etapa. Pesos `IMAGENET1K_V1`; salida de tres logits y softmax
    para probabilidades. AdamW, pérdida ponderada con pesos calculados solo en train,
    AMP y acumulación de gradientes (8×4 ≈ 32 imágenes por actualización; el último
    grupo puede ser menor). No se usa muestreo balanceado adicional ni scheduler
    en esta corrida corta. Early stopping por macro-F1 de validation, paciencia 3.

    El objetivo inicial es obtener una referencia revisable, no forzar convergencia.
    Tres épocas no demuestran estabilidad. El primer modelo tampoco es el modelo final.
    Si ocurre OOM: reduzca batch a 4 y acumule 8, luego cree una nueva corrida con
    ese cambio registrado. No cambie resolución ni particiones para resolver memoria.
    ''')
    add('code', '''
    from tbxtrain.protocol import validate_config
    CONFIG = json.loads((PROJECT_ROOT / 'configs/convnext_tiny_baseline.json').read_text())
    # Modifique antes de comenzar si necesita otro presupuesto. Tras empezar,
    # RESUME_RUN requiere exactamente la misma configuración y código.
    CONFIG = validate_config(CONFIG)
    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    ''')
    add('markdown', '''
    ## 5 · Prueba smoke técnica · F3-SMOKE

    Ejecute esta celda antes del entrenamiento completo. Usa **dos imágenes por
    clase de train y dos de validation** (6 + 6), a la misma resolución de 512 px.
    Comprueba el modelo real con pesos ImageNet, forward/backward, actualización
    efectiva, AMP, escritura del checkpoint en Drive y recarga de predicciones.
    Sus archivos llevan prefijo `smoke_` y no aparecen en la tabla de modelos base.

    **No mide calidad:** no use esas seis imágenes para reportar accuracy o F1.
    Los pesos smoke no inicializan el baseline. La siguiente celda de entrenamiento
    vuelve a los pesos ImageNet y utiliza todo train/validation.
    Si el smoke falla, resuelva el error antes de continuar. Si solo está reanudando
    una corrida que ya verificó, puede omitir el smoke con `RUN_SMOKE=False`.
    ''')
    add('code', '''
    from tbxtrain.train import run_smoke
    RUN_SMOKE = True
    if RUN_SMOKE:
        SMOKE = run_smoke(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, CONFIG)
        print(json.dumps(SMOKE, indent=2, ensure_ascii=False))
    else:
        print('Smoke omitido expresamente; no se genera una evidencia de prueba nueva.')
    ''')
    add('markdown', '''
    ## 6 · Entrenamiento completo del primer modelo base · F3-E3

    Esta celda **sí descarga pesos oficiales y entrena**. La primera vez se calcula
    una validación inicial antes de actualizar pesos; no cuenta como modelo entrenado.
    Después se muestran lotes, pérdida, macro-F1 y sensibilidad TB por época.
    La duración depende del runtime y no se presupone un tiempo fijo.

    En Drive se conservan `experiments/<run_id>/` y `checkpoints/<run_id>/`.
    `last.pt` permite continuar desde la última época completa (incluye optimizador,
    AMP, RNG e historial); `best.pt` conserva el modelo seleccionado por validación.
    Una época interrumpida se repite. No cargue checkpoints `.pt` de terceros.

    Para un ajuste nuevo use la sección 9; **reanudar** significa terminar la
    configuración existente, sin ampliar épocas ni cambiar hiperparámetros.
    ''')
    add('code', '''
    from tbxtrain.train import run_experiment
    BASELINE = run_experiment(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, CONFIG, resume_run=RESUME_RUN)
    BASELINE_DIR = DRIVE_BASE / 'experiments' / BASELINE['run_id']
    atomic_json(BASELINE_DIR / 'transfer_integrity.json', integrity)
    print('training_executed:', BASELINE['training_executed'])
    print('Mejor época:', BASELINE['best_epoch'], '| Fase 3 completa:', BASELINE['phase3_complete'])
    ''')
    add('markdown', '''
    ## 7 · Revisar evidencia · F3-E4

    Selección: mayor macro-F1 que supere al anterior en `min_delta=0.001`.
    Sensibilidad/especificidad TB usan el argmax multiclase, sin ajustar umbrales.
    Se informa AUC OVR y AP de TB como apoyo. Son métricas de validación por imagen,
    sin intervalos de confianza; no son estimaciones clínicas ni evaluación final.

    Revise si mejora la pérdida y si la clase TB se confunde con `sick` o `health`.
    Un macro-F1 mayor no garantiza mejor sensibilidad TB. El contraste contra la
    validación inicial ayuda a ver si hubo aprendizaje, pero no reemplaza nuevas
    semillas, más épocas y una comparación controlada en etapas posteriores.
    ''')
    add('code', '''
    from IPython.display import display, Image
    metrics = BASELINE['validation']
    display({k: metrics[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy', 'macro_roc_auc_ovr']})
    display(metrics['per_class'])
    for name in ['learning_curves.png', 'confusion_matrix.png', 'validation_roc.png']:
        display(Image(filename=str(BASELINE_DIR / 'figures' / name)))
    ''')
    add('markdown', '''
    ## 8 · Revisar errores y escribir la decisión · F3-E5

    Los ejemplos pertenecen únicamente a validation. Se muestran errores con mayor
    confianza para inspección técnica; no se cambian etiquetas por inspección visual.
    Complete la observación basándose en sus resultados reales antes de una nueva corrida.
    ''')
    add('code', '''
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    predictions = pd.read_csv(BASELINE_DIR / 'validation_predictions.csv')
    errors = predictions[predictions.label != predictions.prediction].copy()
    errors['confidence'] = errors[['p_health', 'p_sick', 'p_TB']].max(axis=1)
    errors = errors.sort_values('confidence', ascending=False).head(6)
    display(errors)
    if len(errors):
        names = ['health', 'sick', 'TB']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax in axes.flat:
            ax.axis('off')
        for ax, (_, row) in zip(axes.flat, errors.iterrows()):
            with PILImage.open(DATASET_ROOT / row.relative_path) as im:
                ax.imshow(im.convert('RGB'))
            ax.set_title(f"Real: {names[int(row.label)]} | pred: {names[int(row.prediction)]}")
        fig.tight_layout()
        fig.savefig(BASELINE_DIR / 'figures' / 'validation_errors.png', dpi=150)
        plt.show()
    ''')
    add('code', '''
    from tbxtrain.review import record_review
    OBSERVACION = ''  # Ejemplo de formato: indique qué muestran sus curvas y confusiones reales.
    HIPOTESIS_SIGUIENTE = ''  # Un cambio justificable por validation, no por test.
    DECISION = ''  # Continuar, hacer un ajuste concreto o detener para revisar.
    if all(v.strip() for v in [OBSERVACION, HIPOTESIS_SIGUIENTE, DECISION]):
        record_review(BASELINE_DIR, OBSERVACION, HIPOTESIS_SIGUIENTE, DECISION)
        print('Revisión guardada en review.json.')
    else:
        print('Bitácora pendiente de completar; no se inventa una conclusión.')
    ''')
    add('markdown', '''
    ## 9 · Segundo experimento opcional, después de revisar

    Se deja desactivado para poder presentar el primer avance sin completar la fase.
    Active solo cuando tenga una hipótesis registrada. Se inicia desde `best.pt` de
    la primera corrida y con optimizador nuevo; los datos no cambian.
    El ejemplo reduce las tasas de aprendizaje y permite hasta cinco épocas adicionales.
    Son varias diferencias: esta corrida es un ajuste exploratorio, **no una ablación
    aislada ni una comparación independiente desde la misma inicialización**.
    Para una ablación futura cambie una sola variable y controle pesos, seed y presupuesto.
    ''')
    add('code', '''
    RUN_REFINEMENT = False
    if RUN_REFINEMENT:
        if not (BASELINE_DIR / 'review.json').is_file():
            raise RuntimeError('Registre primero su revisión e hipótesis en la sección 8.')
        import copy
        NEXT_CONFIG = copy.deepcopy(CONFIG)
        NEXT_CONFIG.update(experiment_label='convnext_tiny_v02', epochs=5, warmup_epochs=0,
                           head_lr=3e-4, backbone_lr=5e-6,
                           parent_run=BASELINE['run_id'], hypothesis=HIPOTESIS_SIGUIENTE,
                           change_reason=DECISION)
        REFINEMENT = run_experiment(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, NEXT_CONFIG,
            init_checkpoint=DRIVE_BASE / 'checkpoints' / BASELINE['run_id'] / 'best.pt')
    else:
        print('Segundo experimento pendiente; el primer avance es suficiente para esta entrega parcial.')
    ''')
    add('markdown', '''
    ## 10 · Registro de corridas y evidencia para presentar · F3-E6

    La tabla solo incluye corridas realmente completadas. Una fila representa un
    experimento, no cada época. Puede presentar una fila y dejar ajustes futuros
    pendientes. Revise `status.json` si una corrida se interrumpe; no se incorpora
    como completada hasta regenerar su resumen. Los resultados parciales quedan
    disponibles en `history.csv`, `best_validation.json` y checkpoints.
    ''')
    add('code', '''
    from tbxtrain.review import experiment_table
    table = pd.DataFrame(experiment_table(DRIVE_BASE))
    display(table)
    print(f"Avance parcial de Fase 3: se ejecutó {BASELINE['run_id']} con ConvNeXt-Tiny "
          f"durante {BASELINE['completed_epochs']} épocas completas y se guardaron checkpoints en Drive. "
          f"Mejor macro-F1 de validación: {BASELINE['validation']['macro_f1']:.4f}. "
          "Continúan pendientes la comparación con Vision Mamba y el híbrido, "
          "las repeticiones y la evaluación final reservada.")
    print('Guarde este notebook en Colab con Ctrl+S para conservar las salidas visibles.')
    ''')
    add('markdown', '''
    ## Qué conservar para la entrega

    - Notebook ejecutado en Drive, con las salidas F3-E1 a F3-E6 que ya haya obtenido.
    - `experiments/<run_id>/`: configuración, versiones, hashes de datos/código,
      historial, validación inicial/final, predicciones, curvas, revisión y resumen.
    - `checkpoints/<run_id>/best.pt` y `last.pt`, fuera de Git.
    - Describa la Fase 3 como **iniciada, en entrenamiento y revisión preliminar**
      solo cuando haya actualizaciones guardadas. Todavía no hay tres arquitecturas comparadas.

    No use test interno, test oficial ni externos para elegir ajustes. Los grupos
    evitan cruces de duplicados confirmados, pero no prueban independencia por paciente.
    La evaluación final requerirá incertidumbre y atención a esa dependencia.

    Fuentes oficiales:
    [ConvNeXt-Tiny y pesos](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
    [AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
    [almacenamiento y runtimes Colab](https://research.google.com/colaboratory/faq.html).
    ''')
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': '02_convnext_tiny_baseline.ipynb', 'provenance': []}}}
    output = ROOT/'notebooks/02_convnext_tiny_baseline.ipynb'
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print(output, len(cells), 'celdas; sin ejecutar')


if __name__ == '__main__':
    main()

"""Notebooks independientes: cierre entrenable Fase 3 e inicio del detector Fase 4."""
import json
import textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


class Notebook:
    def __init__(self, name): self.name, self.cells = name, []
    def add(self, kind, text):
        c = {'cell_type': kind, 'id': 'complete-'+str(len(self.cells)), 'metadata': {},
             'source': textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind == 'code': c.update(execution_count=None, outputs=[])
        self.cells.append(c)
    def save(self):
        for i,c in enumerate(self.cells):
            if c['cell_type']=='code': compile(''.join(c['source']),self.name+':'+str(i),'exec')
        nb = {'nbformat':4,'nbformat_minor':5,'cells':self.cells,'metadata':{
            'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
            'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':self.name,'provenance':[]}}}
        (ROOT/'notebooks'/self.name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(self.name, len(self.cells), 'celdas compiladas, sin salidas simuladas')


def bootstrap(nb, requirements):
    nb.add('markdown','''
    ## Montar Drive y recuperar el paquete nuevo

    Abra este notebook en Colab con GPU L4/A100/T4 disponible. El código se extrae
    en otra carpeta temporal; los notebooks ejecutados 02/03 y sus resultados se
    conservan. El nuevo ZIP debe estar en MyDrive/Proyecto_Tuberculosis/.
    Guarde las salidas de cada notebook con Ctrl+S.
    ''')
    nb.add('code','''
    import json, sys, hashlib, stat, zipfile, subprocess
    from pathlib import Path, PurePosixPath
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_BASE = Path('/content/drive/MyDrive/Proyecto_Tuberculosis')
    ARCHIVE = DRIVE_BASE / 'tbx11k_fases_3_y_4_colab.zip'
    PROJECT_ROOT = Path('/content/tbx11k_complete/project')
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for item in z.infolist():
            rel = PurePosixPath(item.filename)
            target = (PROJECT_ROOT / item.filename).resolve()
            if rel.is_absolute() or '..' in rel.parts or ':' in item.filename or '\\\\' in item.filename or stat.S_ISLNK(item.external_attr >> 16) or PROJECT_ROOT.resolve() not in target.parents:
                raise ValueError('Ruta no segura: '+item.filename)
        checksums = json.loads(z.read('phase3_package_checksums.json'))
        for name, expected in checksums.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError('Archivo alterado en el paquete: '+name)
        z.extractall(PROJECT_ROOT)
    # Importa el código nuevo incluso si el runtime había cargado el paquete anterior.
    for name in list(sys.modules):
        if name in {'tbxtrain','tbxdata'} or name.startswith(('tbxtrain.','tbxdata.')):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    ''')
    nb.add('code',f'''
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                           '--disable-pip-version-check', '-r', str(PROJECT_ROOT / '{requirements}')])
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Seleccione un runtime GPU de Colab.')
    from tbxdata.colab import stage_dataset
    from tbxdata.validate import validate_manifests
    source = DRIVE_BASE / 'TBX11K.zip'
    if not source.exists(): source = DRIVE_BASE / 'TBX11K'
    DATASET_ROOT = stage_dataset(source, Path('/content/tbx11k_complete/data_cache'), True)
    integrity = validate_manifests(PROJECT_ROOT, DATASET_ROOT, all_image_hashes=True)
    print('GPU:',torch.cuda.get_device_name(0),'| Integridad:',integrity['status'])
    print('Hashes de imágenes verificados:',integrity['all_image_hashes_verified'])
    ''')


def phase3():
    nb = Notebook('04_cierre_fase3_clasificadores.ipynb')
    nb.add('markdown','''
    # 04 · Culminar Fase 3: tres clasificadores comparables

    ConvNeXt-Tiny, VMamba-T s1l8 oficial y un híbrido con atención cruzada
    bidireccional entre mapas espaciales. No es un ensemble de probabilidades.

    **Este notebook entrena los modelos que faltan y genera un cierre verificable.**
    La fuente no contiene entrenamientos ejecutados. El cierre solo se produce si
    existen tres corridas completas y sus checkpoints coinciden con sus hashes.

    v01/v02 quedan como antecedentes exploratorios. Para comparar arquitecturas,
    esta tanda parte de pesos ImageNet por rama, con iguales datos, seed, presupuesto,
    optimizador y criterio de selección. El híbrido no hereda ConvNeXt ya ajustado
    a TBX11K. Así se distingue el efecto de la arquitectura del entrenamiento previo.

    Se conservan tres clases (health/sick/TB), 512px y aumentos congelados. Cada
    modelo dispone de hasta 12 épocas, warmup 2, paciencia 4 y lote efectivo 32.
    El número realmente ejecutado puede diferir por early stopping. La comparación
    usa una seed: no prueba superioridad estadística ni constituye evaluación final.
    **Test, umbrales, Grad-CAM y selección final corresponden a Fase 5.**
    ''')
    bootstrap(nb,'requirements-phase3-suite.txt')
    nb.add('markdown','''
    ## F3-C1 · Dependencia CUDA y pesos oficiales VMamba

    Compila selective scan desde la revisión fijada del repositorio oficial;
    conserva el torch/CUDA del runtime y registra el log en /content/tbx_vmamba_build/.
    Puede tardar varios minutos. Si falla, revise el log antes de entrenar.
    No instale wheels antiguas de Python/torch que no correspondan a su runtime.
    Los pesos ImageNet se conservan en Drive/pretrained/. Su carga es estricta.
    ''')
    nb.add('code',(ROOT/'scripts/colab_vmamba_diagnostic_cell.py').read_text(encoding='utf-8'))
    nb.add('markdown','''
    ## F3-C2 · Smoke de las tres arquitecturas

    Un ejemplo por clase/split a 512px verifica forward/backward, actualización y
    guardado/recarga. Son corridas técnicas; no forman parte de la tabla de comparación.
    FP32 evita depender del escalado AMP que dificultó el smoke anterior.
    ''')
    nb.add('code','''
    from tbxtrain.suite import run_suite_model
    RUN_SMOKES = True
    if RUN_SMOKES:
        for architecture in CONFIG_SUITE['architectures']:
            smoke = run_suite_model(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, architecture,
                                    CONFIG_SUITE, weights_path=WEIGHTS, smoke=True)
            print(architecture, smoke['status'])
    ''')
    nb.add('markdown','''
    ## F3-C3 · Entrenamientos completos y recuperación automática

    Se entrena una arquitectura después de otra. Se publican last.pt e historial
    al completar cada época, con RNG y optimizador. Una época parcial se repite.
    El registro en results/phase3_suite/ permite reanudar al volver a ejecutar esta
    celda con el mismo código/configuración. Las corridas completas se recuperan
    sin entrenarlas otra vez. No cambie parámetros a mitad de una tanda controlada.
    ''')
    nb.add('code','''
    RUN_IDS = []
    for architecture in CONFIG_SUITE['architectures']:
        marker = DRIVE_BASE / 'results/phase3_suite' / CONFIG_SUITE['protocol_id'] / (architecture+'.json')
        resume = None
        if marker.exists():
            prior = json.loads(marker.read_text())
            run_id = prior['run_id']
            summary_path = DRIVE_BASE / 'experiments' / run_id / 'summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary['status'] == 'completed_controlled':
                    if summary['protocol'] != CONFIG_SUITE: raise RuntimeError('El protocolo cambió. Defina otro protocol_id para una nueva tanda.')
                    RUN_IDS.append(run_id)
                    print('Recuperada sin reentrenar:',run_id)
                    continue
            if (DRIVE_BASE / 'checkpoints' / run_id / 'last.pt').exists(): resume = run_id
        summary = run_suite_model(PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, architecture,
                                   CONFIG_SUITE, weights_path=WEIGHTS, resume_run=resume)
        RUN_IDS.append(summary['run_id'])
    print('Corridas disponibles:', RUN_IDS)
    ''')
    nb.add('markdown','''
    ## F3-C4 · Comparar y cerrar con evidencia

    Requiere las tres arquitecturas, mismo protocolo/datos/código, validación
    completa de 1.241 imágenes y checkpoints íntegros. Si algo falta, falla y no
    declara la fase terminada. El ranking es preliminar: la decisión final de
    Fase 5 incluirá detector, explicabilidad, concordancia y test reservado.
    ''')
    nb.add('code','''
    import pandas as pd
    from IPython.display import display, Image
    from tbxtrain.suite import close_phase3
    from tbxtrain.protocol import atomic_json
    CLOSURE, CLOSURE_DIR = close_phase3(DRIVE_BASE, RUN_IDS)
    atomic_json(CLOSURE_DIR / 'transfer_integrity.json', integrity)
    display(pd.DataFrame(CLOSURE['comparison']).sort_values('macro_f1',ascending=False))
    for row in CLOSURE['comparison']:
        print(row['architecture'], row['run_id'])
        folder = DRIVE_BASE / 'experiments' / row['run_id'] / 'figures'
        for name in ['learning_curves.png','confusion_matrix.png']:
            display(Image(filename=str(folder / name)))
    print('Fase 3 completa:',CLOSURE['phase3_complete'])
    print('Entregar esta carpeta:', CLOSURE_DIR)
    print('Guardar este notebook ejecutado con Ctrl+S. Fases 4 y 5 permanecen pendientes.')
    ''')
    nb.save()


def phase4():
    nb = Notebook('05_fase4_detector_TB.ipynb')
    nb.add('markdown','''
    # 05 · Fase 4: iniciar detector de lesiones TB

    YOLO11n pretrained COCO, con una clase tb_lesion. Convierte las cajas originales
    de train/validation a YOLO manteniendo sus particiones. Health/sick son negativos
    sin cajas. TB sin cajas se excluye: no es un negativo. Los subtipos se conservan
    en los metadatos originales, pero no se exigen como clases de detección.

    **Fase 4 preparada para iniciar, no ejecutada en esta fuente.** Este notebook
    no contiene la evaluación final de Fase 5. El dataset.yaml no define test.
    Ultralytics 8.3.228 se fija para reproducibilidad; su código se distribuye bajo
    AGPL-3.0. Revise esa licencia al distribuir el sistema posterior.
    ''')
    bootstrap(nb,'requirements-detection.txt')
    nb.add('code','''
    from datetime import datetime, timezone
    from tbxtrain.detection import prepare_detection
    from tbxtrain.protocol import atomic_json
    # Para reanudar, escriba el nombre de la carpeta del detector ya creada.
    DETECTOR_RESUME_RUN = None
    DET_RUN = DETECTOR_RESUME_RUN or ('detector_yolo11n_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if Path(DET_RUN).name != DET_RUN or any(c in DET_RUN for c in '/\\\\:') or DET_RUN in {'.','..'}:
        raise ValueError('Nombre de corrida detector inválido.')
    DET_DATA = Path('/content/tbx11k_complete') / DET_RUN
    if not DET_DATA.exists():
        conversion = prepare_detection(PROJECT_ROOT, DATASET_ROOT, DET_DATA)
    else:
        conversion = json.loads((DET_DATA/'conversion.json').read_text())
    print(json.dumps(conversion,indent=2,ensure_ascii=False))
    DET_OUTPUT = DRIVE_BASE / 'experiments' / DET_RUN
    DET_OUTPUT.mkdir(parents=True,exist_ok=bool(DETECTOR_RESUME_RUN))
    atomic_json(DET_OUTPUT / 'conversion.json',conversion)
    ''')
    nb.add('markdown','''
    ## F4-E1 · Revisar cajas convertidas antes de entrenar

    Los rectángulos se dibujan desde las etiquetas YOLO exportadas sobre la imagen
    copiada, para comprobar la conversión y las coordenadas. Una revisión técnica
    no cambia las anotaciones ni sustituye su revisión por el equipo.
    ''')
    nb.add('code','''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    positive = [p for p in (DET_DATA / 'labels/train').glob('*.txt') if p.stat().st_size][:6]
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for ax in axes.flat: ax.axis('off')
    for ax,label_path in zip(axes.flat,positive):
        image_path = DET_DATA / 'images/train' / (label_path.stem+'.png')
        with Image.open(image_path) as im:
            w,h=im.size; ax.imshow(im.convert('RGB'))
        for line in label_path.read_text().splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            ax.add_patch(patches.Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,
                                           fill=False,edgecolor='red',linewidth=1.5))
    fig.tight_layout(); fig.savefig(DET_OUTPUT/'converted_boxes.png',dpi=150); plt.show()
    ''')
    nb.add('markdown','''
    ## F4-E2 · Entrenamiento opcional del detector

    Se deja desactivado para que pueda revisar primero la conversión. Al activarlo
    se entrena con todo train y se valida con validation. Se desactivan espejado,
    mosaico y geometrías agresivas. El detector maneja sus transformaciones de
    imagen/cajas conjuntamente. No ajuste estos parámetros con test.

    El smoke opcional usa hasta tres positivos y tres negativos por split,
    solo como verificación técnica; sus métricas
    no sirven como resultado del detector. El entrenamiento real está separado.
    ''')
    nb.add('code','''
    from ultralytics import YOLO
    RUN_DETECTOR_SMOKE = False
    RUN_DETECTOR_TRAINING = False
    common = dict(data=str(DET_DATA/'dataset.yaml'),imgsz=512,batch=8,device=0,workers=2,
                  seed=20261019,amp=False,fliplr=0.,flipud=0.,mosaic=0.,mixup=0.,
                  degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,
                  hsv_h=0.,hsv_s=0.,hsv_v=.08,save=True,save_period=1)
    if RUN_DETECTOR_SMOKE:
        smoke_common = dict(common,data=str(DET_DATA/'smoke.yaml'),batch=2)
        YOLO('yolo11n.pt').train(**smoke_common,epochs=1,
            project=str(DET_OUTPUT),name='smoke',exist_ok=False)
    if RUN_DETECTOR_TRAINING:
        if DETECTOR_RESUME_RUN:
            last = DET_OUTPUT/'training/weights/last.pt'
            if not last.is_file(): raise FileNotFoundError(last)
            YOLO(str(last)).train(resume=True,device=0)
        else:
            if (DET_OUTPUT/'training').exists(): raise RuntimeError('Ya existe training/. Reanude esa corrida o prepare un DET_RUN nuevo.')
            YOLO('yolo11n.pt').train(**common,epochs=35,patience=7,
                project=str(DET_OUTPUT),name='training',exist_ok=False)
    else:
        print('Entrenamiento detector pendiente. Conversión y revisión ya pueden documentarse.')
    ''')
    nb.add('markdown','''
    ## F4-E3 · Validación y salida del detector

    Solo habilitada si existe un entrenamiento real. Conserva best/last.pt,
    results.csv, curvas, configuración y mAP de validation. La evaluación final,
    Grad-CAM y regla clasificación–detección se preparan después en Fase 5.
    Para reanudar una desconexión del detector, escriba su nombre de carpeta en
    DETECTOR_RESUME_RUN, repita preparación y active RUN_DETECTOR_TRAINING. Se
    reconstruye la misma ruta local y YOLO recupera training/weights/last.pt.
    ''')
    nb.add('code','''
    if RUN_DETECTOR_TRAINING:
        best = DET_OUTPUT/'training/weights/best.pt'
        if not best.is_file(): raise FileNotFoundError(best)
        trained = YOLO(str(best))
        result = trained.val(data=str(DET_DATA/'dataset.yaml'),split='val',imgsz=512,
                              device=0,project=str(DET_OUTPUT),name='validation',exist_ok=False)
        summary = dict(status='trained_validation_available',
                       map50_95=float(result.box.map),map50=float(result.box.map50),
                       validation_only=True,test_evaluated=False,phase5_complete=False)
        atomic_json(DET_OUTPUT/'detector_summary.json',summary)
        print(summary)
    ''')
    nb.save()


if __name__ == '__main__':
    phase3(); phase4()

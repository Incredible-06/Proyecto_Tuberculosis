"""Notebook independiente para la Fase 1 del plan académico."""
import json
from pathlib import Path
import textwrap


def main():
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'id':f'env-{len(cells):03d}','metadata':{},
           'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','''
    # 00 Preparación del entorno de trabajo

    Fase 1 del plan de Ingeniería de Software II. La Fase 2 del plan corresponde
    al notebook 01 de datos. Aquí se verifican entorno, Drive, rutas, seeds y una
    operación CUDA diminuta si existe GPU. No se carga ningún modelo o peso ni se
    entrena. Esta ejecución no valida la compatibilidad de futuras extensiones de
    Vision Mamba. Puede usarse cualquier GPU compatible asignada, sin asumir A100.
    ''')
    # Reutiliza el bootstrap probado de 01; solo cambia el nombre del paquete.
    original=json.loads(Path('notebooks/01_exploracion_preprocesamiento.ipynb').read_text(encoding='utf-8'))
    for index in [1,2,3,4,5,6,7,8]:
        c=original['cells'][index]
        text=''.join(c['source'])
        text=text.replace('tbx11k_phase1_colab.zip','tbx11k_fases_1_y_2_colab.zip')
        text=text.replace('VERIFY_ALL_IMAGE_HASHES = False','VERIFY_ALL_IMAGE_HASHES = IN_COLAB')
        text=text.replace('RUN_COLAB_DATALOADER_CHECK = True','RUN_COLAB_DATALOADER_CHECK = False')
        add(c['cell_type'],text)
    add('markdown','''
    ## Seeds y comprobación del dispositivo

    Se fijan seeds para random, NumPy y PyTorch, si se ejecuta en Colab. El uso de
    algoritmos deterministas se registra, sin prometer equivalencia absoluta entre
    hardware o versiones. Para la GPU: Entorno de ejecución, Cambiar tipo de entorno
    de ejecución, seleccionar GPU. Si no hay GPU, se conserva la evidencia CPU y
    el resultado GPU pendiente. No reinstale CUDA/PyTorch a ciegas.
    ''')
    add('code','''
    from tbxdata.environment import inspect_environment
    from tbxdata.dataset import load_config
    config = load_config(PROJECT_ROOT / 'configs/data.json')
    setup_report = inspect_environment(config['seed'], verify_cuda=IN_COLAB)
    print(json.dumps(setup_report, indent=2))
    ''')
    add('markdown','''
    ## Comprobar la transferencia completa

    Se reutilizan los hashes existentes. En Colab se comparan los 12.278 archivos
    para cerrar la comprobación pendiente del primer envío. No se explora de nuevo
    el dataset ni se modifican particiones. Localmente solo se usan muestras.
    ''')
    add('code','''
    transfer_report = validate_manifests(PROJECT_ROOT, DATASET_ROOT,
                                        hash_samples_per_class=2,
                                        all_image_hashes=IN_COLAB)
    print('Integridad:', transfer_report['status'])
    print('Archivos con SHA256 verificado completo:', transfer_report['all_image_hashes_verified'])
    ''')
    add('markdown','''
    ## Experimentos y checkpoints futuros

    Las carpetas de salida se preparan en Drive y los resultados se guardan al
    finalizar. No se generan checkpoints de modelos en estas fases. Los pesos
    futuros deben permanecer fuera del repositorio Git.
    ''')
    add('code','''
    import shutil
    from datetime import datetime, timezone
    setup_name = 'entorno_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    setup_output = (DRIVE_BASE / 'results' / setup_name) if IN_COLAB else (PROJECT_ROOT / 'reports' / 'environment' / setup_name)
    setup_output.mkdir(parents=True, exist_ok=True)
    if IN_COLAB:
        EXPERIMENTS_ROOT = DRIVE_BASE / 'experiments'
        CHECKPOINTS_ROOT = DRIVE_BASE / 'checkpoints'
        EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
        setup_report['output_paths'] = {'experiments': str(EXPERIMENTS_ROOT), 'checkpoints': str(CHECKPOINTS_ROOT)}
    (setup_output / 'environment_setup.json').write_text(json.dumps(setup_report, indent=2), encoding='utf-8')
    (setup_output / 'transfer_integrity.json').write_text(json.dumps(transfer_report, indent=2), encoding='utf-8')
    shutil.copy2(PROJECT_ROOT / 'configs/data.json', setup_output / 'config.json')
    shutil.copy2(PROJECT_ROOT / 'manifests/manifest_metadata.json', setup_output / 'manifest_metadata.json')
    print('Entregar esta carpeta para actualizar el informe:', setup_output)
    if setup_report['cuda_probe']['status'] != 'passed':
        print('GPU/CUDA no verificados en esta ejecución; no presentar como completados.')
    ''')
    add('markdown','''
    ## Evidencias para cerrar las dos fases del curso

    Entregar este notebook ejecutado y la carpeta entorno creada en Drive. Para
    declarar GPU verificada, environment_setup.json debe mostrar cuda_probe passed.
    transfer_integrity.json debe registrar 12.278 hashes completos. El notebook 01
    ya tiene evidencia de datos en CPU; no necesita repetir el entrenamiento porque
    no existe entrenamiento en estas dos fases. La URL y commit de GitHub se
    documentarán al publicar el repositorio del equipo. No se promete una A100.
    ''')
    nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},'colab':{'name':'00_configuracion_entorno.ipynb','provenance':[]}}}
    from refine_notebooks import refine_environment
    refine_environment(nb)
    target=Path('notebooks/00_configuracion_entorno.ipynb')
    target.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(target,len(cells),'celdas')


if __name__=='__main__':main()

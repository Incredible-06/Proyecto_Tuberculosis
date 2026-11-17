"""Controles y trazabilidad de experimentos; importable sin PyTorch."""
import json
import re
from pathlib import Path
from tbxdata.common import sha256


def validate_config(config):
    expected = {
        'experiment_label', 'model', 'weights', 'seed', 'epochs', 'warmup_epochs',
        'unfreeze_last_stage', 'batch_size', 'accumulation_steps', 'num_workers',
        'head_lr', 'backbone_lr', 'weight_decay', 'gradient_clip', 'amp', 'patience',
        'min_delta', 'selection_metric', 'hypothesis', 'parent_run', 'change_reason'}
    if set(config) != expected:
        raise ValueError(f'Campos de configuración incorrectos: {set(config)^expected}')
    if config['model'] != 'convnext_tiny' or config['weights'] != 'IMAGENET1K_V1':
        raise ValueError('Este pipeline implementa únicamente ConvNeXt-Tiny IMAGENET1K_V1.')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', config['experiment_label']):
        raise ValueError('Etiqueta de experimento inválida.')
    for key in ['seed', 'epochs', 'warmup_epochs', 'batch_size', 'accumulation_steps', 'num_workers', 'patience']:
        if type(config[key]) is not int or config[key] < (1 if key in {'epochs', 'batch_size', 'accumulation_steps', 'patience'} else 0):
            raise ValueError(f'Entero inválido: {key}')
    if config['warmup_epochs'] >= config['epochs'] and config['unfreeze_last_stage']:
        raise ValueError('Debe quedar al menos una época para ajustar la última etapa.')
    for key in ['head_lr', 'backbone_lr', 'gradient_clip']:
        if not 0 < config[key] < float('inf'):
            raise ValueError(f'Valor positivo requerido: {key}')
    for key in ['weight_decay', 'min_delta']:
        if not 0 <= config[key] < float('inf'):
            raise ValueError(f'Valor no negativo requerido: {key}')
    for key in ['amp', 'unfreeze_last_stage']:
        if type(config[key]) is not bool:
            raise ValueError(f'Booleano requerido: {key}')
    if config['selection_metric'] != 'macro_f1':
        raise ValueError('El criterio predefinido es macro-F1 en validation.')
    if not config['hypothesis'].strip() or not config['change_reason'].strip():
        raise ValueError('Registre hipótesis y motivo antes de entrenar.')
    return dict(config)


def provenance(project):
    project = Path(project)
    files = [project/'configs/data.json', project/'manifests/manifest_metadata.json']
    files += [project/f'manifests/{s}.csv' for s in ['train', 'validation', 'test']]
    files += sorted((project/'src/tbxtrain').glob('*.py'))
    files += sorted((project/'src/tbxdata').glob('*.py'))
    return {p.relative_to(project).as_posix(): sha256(p) for p in files}


def check_warm_start(parent, current):
    # Un ajuste nuevo puede cambiar hiperparámetros; nunca mapa de clases/datos.
    keys = ['configs/data.json', 'manifests/train.csv', 'manifests/validation.csv', 'manifests/test.csv']
    if any(parent.get(key) != current.get(key) for key in keys):
        raise ValueError('Checkpoint de otro protocolo de datos; no se puede usar como inicio.')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

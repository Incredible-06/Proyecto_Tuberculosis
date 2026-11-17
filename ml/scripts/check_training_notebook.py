"""Validación estática del notebook 02 y del protocolo congelado; sin entrenamiento."""
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxdata.validate import validate_manifests
from tbxtrain.protocol import atomic_json, validate_config


def main():
    target = ROOT/'notebooks/02_convnext_tiny_baseline.ipynb'
    nb = json.loads(target.read_text(encoding='utf-8'))
    ids, code_cells = [], 0
    for index, cell in enumerate(nb['cells']):
        ids.append(cell['id'])
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'{target.name}:cell{index}', 'exec')
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('La fuente debe conservarse sin salidas; no simular ejecución.')
            code_cells += 1
    if len(ids) != len(set(ids)):
        raise ValueError('IDs de celda duplicados.')
    for source in (ROOT/'src/tbxtrain').glob('*.py'):
        ast.parse(source.read_text(encoding='utf-8'))
    integrity = validate_manifests(ROOT)
    config = validate_config(json.loads((ROOT/'configs/convnext_tiny_baseline.json').read_text()))
    report = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'code_cells_compiled': code_cells, 'all_source_cells_unexecuted': True,
        'frozen_manifests': integrity['status'], 'class_map': {'health': 0, 'sick': 1, 'TB': 2},
        'epochs_configured': config['epochs'], 'colab_training_executed': False,
        'scope': 'Compilación estática, configuración y metadatos. No prueba CUDA, descarga de pesos ni entrenamiento Colab.'}
    atomic_json(ROOT/'reports/phase3/notebook_validation.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

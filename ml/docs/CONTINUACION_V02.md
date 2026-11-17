# Celdas para añadir después de v01


# Continuación v02 · conservar resultados de v01

**Estas son celdas adicionales para copiar al final de su notebook 02 ejecutado.**
Primero guarde sus salidas con Ctrl+S; puede guardar además una copia de respaldo
mediante Archivo → Guardar una copia en Drive. Después ejecute solo las celdas
nuevas. La opción «Ejecutar todas» volvería a lanzar el entrenamiento v01.

Se conserva la corrida `convnext_tiny_v01_20261027T034012Z_860908e3`, cuya mejor
época fue la 2. La continuación usa su `best.pt` y guarda resultados en otro
ID `convnext_tiny_v02_...`. Los outputs previos permanecen en sus celdas.

**No cambie la celda original para pasar de 3 a 8 épocas:** una nueva corrida
desde el mejor checkpoint permite conservar v01 como referencia.

Este bloque reutiliza el runtime actual. Si se desconectó, en el notebook 02
ejecute solo las celdas de rutas, montaje, dependencias y staging/integridad
(secciones 1–3). Después vuelva directamente aquí. No hace falta ejecutar el
smoke ni el entrenamiento v01 para recuperar un resultado ya guardado.

## Celda adicional 1 · recuperar la referencia sin entrenarla

Se lee el resumen ya guardado, para que las variables no dependan de volver a
ejecutar v01. Sus archivos se usan como origen y no se sobrescriben.

```python
import json
import sys
from pathlib import Path

required = ['PROJECT_ROOT', 'DATASET_ROOT', 'DRIVE_BASE']
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError('Recupere primero montaje, código y dataset en las secciones 1–3 del notebook 02. Faltan: ' + ', '.join(missing))
PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE = map(Path, (PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE))
if not (PROJECT_ROOT / 'src/tbxtrain/train.py').is_file() or not (DATASET_ROOT / 'imgs').is_dir():
    raise RuntimeError('El runtime perdió el código o dataset temporal. Ejecute solo la preparación de las secciones 1–3 del notebook 02.')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from tbxtrain.train import run_experiment
from tbxtrain.protocol import validate_config

V01_RUN = 'convnext_tiny_v01_20261027T034012Z_860908e3'
V01_DIR = DRIVE_BASE / 'experiments' / V01_RUN
V01 = json.loads((V01_DIR / 'summary.json').read_text())
if V01['status'] != 'completed_preliminary' or not V01['training_executed'] or V01['run_id'] != V01_RUN:
    raise RuntimeError('La referencia v01 no es una corrida completada válida.')
V01_BEST = DRIVE_BASE / 'checkpoints' / V01_RUN / 'best.pt'
if not V01_BEST.is_file():
    raise FileNotFoundError(V01_BEST)
print('Referencia recuperada, sin entrenarla:', V01_RUN)
print('Época seleccionada:', V01['best_epoch'])
print('Macro-F1:', round(V01['validation']['macro_f1'], 4))
print('Sensibilidad TB:', round(V01['validation']['per_class']['TB']['recall'], 4))
```

## Celda adicional 2 · definir el siguiente ajuste

Hasta **cinco épocas adicionales desde el mejor estado de v01**, con tasas de
aprendizaje menores y FP32. No se repite warmup: la cabeza ya fue adaptada.
Early stopping mantiene paciencia 3 y selección por macro-F1 de validation.

Es un ajuste exploratorio que combina continuación, tasas menores y otro
presupuesto; no demuestra el efecto aislado de una variable. No garantiza que
mejoren los resultados. La comparación debe incluir sensibilidad TB y falsos
positivos, además de macro-F1. El test continúa reservado.

Las épocas de v02 se numeran 1–5 dentro de su corrida y parten de la **época 2**
de v01; no equivalen a continuar desde la época 3 descartada.

```python
V02_CONFIG = dict(V01['config'])
V02_CONFIG.update(
    experiment_label='convnext_tiny_v02',
    parent_run=V01_RUN,
    epochs=5,
    warmup_epochs=0,
    head_lr=3e-4,
    backbone_lr=5e-6,
    amp=False,
    hypothesis='Un ajuste más suave desde la mejor época de v01 podría mejorar la validación y reducir TB confundida con sick.',
    change_reason='En v01 la época 3 empeoró validación; en la mejor época, 24 TB se predijeron sick y 8 health. Se explora continuación desde best.pt con tasas menores.'
)
V02_CONFIG = validate_config(V02_CONFIG)
# Solo si se interrumpe v02: copie aquí el nombre exacto de SU carpeta v02.
# Mantenga esta configuración igual; la época parcial se repetirá.
V02_RESUME_RUN = None
print(json.dumps(V02_CONFIG, indent=2, ensure_ascii=False))
```

## Celda adicional 3 · entrenar v02 y guardar sus propias salidas

Al ejecutar se crea una carpeta nueva en experiments/ y checkpoints/. La
inicialización lee `v01/best.pt`; el optimizador empieza de nuevo. Si se usa
V02_RESUME_RUN, se recupera v02/last.pt con su optimizador y estado guardado.

Ejecute esta celda una vez para crear v02. Repetirla con V02_RESUME_RUN=None
crearía otra corrida independiente con un ID nuevo.

```python
V02 = run_experiment(
    PROJECT_ROOT, DATASET_ROOT, DRIVE_BASE, V02_CONFIG,
    resume_run=V02_RESUME_RUN,
    init_checkpoint=None if V02_RESUME_RUN else V01_BEST
)
V02_DIR = DRIVE_BASE / 'experiments' / V02['run_id']
print('Nueva corrida:', V02['run_id'])
print('Resultados:', V02_DIR)
print('Mejor época de v02:', V02['best_epoch'])
print('Épocas completas de v02:', V02['completed_epochs'])
```

## Celda adicional 4 · comparar ambas versiones sin borrar resultados

Los gráficos nuevos se muestran en esta celda; las salidas antiguas permanecen
arriba. El checkpoint elegido dentro de v02 usa macro-F1, igual que en v01.
Revise también la sensibilidad TB antes de decidir si v02 merece conservarse
como nueva referencia. Una corrida nueva puede resultar peor que v01.

```python
import pandas as pd
from IPython.display import display, Image

comparison = []
for version, summary in [('v01', V01), ('v02', V02)]:
    m = summary['validation']
    tb = m['per_class']['TB']
    cm = m['confusion_matrix']
    comparison.append({
        'version': version, 'run_id': summary['run_id'],
        'epochs_in_run': summary['completed_epochs'], 'best_epoch': summary['best_epoch'],
        'macro_f1': m['macro_f1'], 'accuracy': m['accuracy'],
        'TB_recall': tb['recall'], 'TB_precision': tb['precision'],
        'TB_specificity': tb['specificity'],
        'TB_to_sick': cm[2][1], 'TB_to_health': cm[2][0]
    })
comparison = pd.DataFrame(comparison)
display(comparison)
# La comparación se guarda solo en la carpeta nueva.
comparison.to_csv(V02_DIR / 'comparison_v01_v02.csv', index=False)
for name in ['learning_curves.png', 'confusion_matrix.png', 'validation_roc.png']:
    display(Image(filename=str(V02_DIR / 'figures' / name)))
print('Guarde el notebook con Ctrl+S para conservar las salidas de ambas versiones.')
```

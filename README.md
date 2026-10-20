# Proyecto Tuberculosis — Preparación del entorno y de los datos

## Correspondencia con el plan de Ingeniería de Software II

En el plan del curso, la **Fase 1** es preparar el entorno y la **Fase 2** es
preparar el dataset. El término histórico «phase1» de algunas carpetas y del primer
ZIP se refería al pipeline de datos; no representa la numeración del plan del curso.
El entrenamiento de clasificadores es una etapa posterior y no se ha iniciado.

- `notebooks/00_configuracion_entorno.ipynb`: entorno, seeds, GPU/CUDA en Colab y verificación completa de transferencia.
- `notebooks/01_exploracion_preprocesamiento.ipynb`: datos, particiones y transformaciones; ya cuenta con ejecución CPU entregada.
- Repositorio del equipo: https://github.com/Incredible-06/Proyecto_Tuberculosis

Para cerrar la evidencia pendiente, suba `dist/tbx11k_fases_1_y_2_colab.zip` a la
misma carpeta de Drive y ejecute el notebook 00 en Colab con GPU, cuando esté
disponible. Guarde `environment_setup.json` y `transfer_integrity.json` de la
carpeta `results/entorno_<fechaUTC>/`. El notebook no carga modelos ni entrena.
El notebook 01 conserva su paquete original para reproducir la ejecución previa.
No es necesario rehacer sus particiones ni volver a explorar el dataset.

Los datos originales, archivos ZIP, resultados ejecutados, pesos y figuras
generadas están excluidos de Git. Las figuras se regeneran con los scripts.
`api/`, `web/` y `models/` documentan componentes reservados; no implementan todavía
una API, una web o modelos entrenados.

La Fase 1 está implementada: revisión de duplicados cercanos, particiones propias,
lector reutilizable y notebook de Colab. **No se ha iniciado entrenamiento de
ConvNeXt ni Vision Mamba.** El dataset original no se modifica.

Entregables principales:

- [01_exploracion_preprocesamiento.ipynb](notebooks/01_exploracion_preprocesamiento.ipynb)
- [Paquete listo para subir a Drive](dist/tbx11k_phase1_colab.zip), sin dataset ni pesos.
- [Decisiones y validación](reports/phase1/decisiones_y_validacion.md)
- [Configuración centralizada](configs/data.json)
- Manifiestos [train](manifests/train.csv), [validation](manifests/validation.csv) y [test interno](manifests/test.csv).

## Particiones congeladas

| Partición | health (0) | sick (1) | TB (2) | Total |
|---|---:|---:|---:|---:|
| train | 2.660 | 2.572 | 560 | 5.792 |
| validation | 570 | 551 | 120 | 1.241 |
| test interno | 570 | 551 | 120 | 1.241 |

Los 8.400 archivos base etiquetados contienen 8.274 contenidos RGB exactos únicos.
Se usa un representante determinista por hash; las variantes cercanas confirmadas
se conservan agrupadas en el mismo split. Hay 8.251 grupos de contenido. La seed
es 20261019 y las cuotas por clase son aproximadamente 70/15/15. No hay cruces
internos por ruta, hash o grupo confirmado. No hay IDs fiables de paciente.

El test interno se reserva para evaluación final: no seleccionar modelos,
umbrales, CLAHE o aumentos con él. La búsqueda cercana no fue exhaustiva.

## Ejecutar en Google Colab

1. Cree `MyDrive/Proyecto_Tuberculosis/` en Google Drive.
2. Suba `dist/tbx11k_phase1_colab.zip` a esa carpeta con ese nombre.
3. Suba por separado el dataset local existente: la carpeta `TBX11K/` completa o
   un `TBX11K.zip` que contenga esa carpeta, o directamente sus `imgs/`, `lists/`
   y `annotations/`. Mantenga nombres y subcarpetas. No vuelva a descargar TBX11K.
   Para reducir lecturas pequeñas en Drive, se prefiere el ZIP si existe.
4. Abra el notebook en Colab mediante **Archivo → Subir notebook**. Para esta
   fase basta CPU; una GPU es opcional.
5. Ajuste `DRIVE_BASE` si cambió la ubicación. Deje `COPY_DATASET_TO_LOCAL=True`
   para usar `/content/tbx11k_phase1/data_cache/`. `DATA_SOURCE_KIND` acepta
   `auto`, `zip` o `folder`.
6. Ejecute en orden y autorice el montaje de Drive cuando Colab lo solicite.
   Se instalan solo NumPy/Pillow en los rangos de `requirements-data.txt`; se
   registran las versiones exactas. No se instala ni cambia PyTorch/CUDA.
7. Para la primera transferencia, se recomienda activar
   `VERIFY_ALL_IMAGE_HASHES=True` en Colab: compara todos los archivos contra los
   hashes existentes. Por defecto se comprueban tamaños de todos y hashes de
   tres muestras por clase/partición.
8. Revise distribución, muestras y aumentos. El lote opcional PyTorch usa cuatro
   imágenes CPU, sin modelo ni entrenamiento.
9. La última celda guarda evidencias en
   `MyDrive/Proyecto_Tuberculosis/results/phase1_<fechaUTC>/`.

El paquete pequeño contiene código, configuración, notebooks, manifiestos y
reportes. Los 4,24 GB de imágenes permanecen fuera de ese ZIP. Una raíz incorrecta,
archivos ausentes, configuración o manifiestos alterados producen errores antes
de continuar.

Tras una desconexión, ejecute de nuevo desde el montaje. Si se conserva `/content`,
se reutiliza una copia completa cuyo marcador coincide; una copia interrumpida de
carpeta continúa por existencia/tamaño. Después siempre se valida la copia.
Si la fuente cambió, use otra carpeta cache. Al perder el runtime, vuelva a copiar
el dataset desde Drive. Los manifiestos se recuperan sin crear otras particiones.
No existen checkpoints de modelos en esta fase.

`REBUILD_MANIFESTS_FOR_COMPARISON=True` reproduce en una carpeta de la ejecución y
exige coincidencia con la versión congelada. No edite `configs/data.json` para una
ejecución ordinaria: forma parte de la huella del protocolo. Para cambiar aumentos
posteriormente, cree una versión documentada y mantenga cerrado el test.

## Código reutilizable y preprocesamiento

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('src').resolve()))
from tbxdata.dataset import TBXDataset, load_config

config = load_config('configs/data.json')
train = TBXDataset('TBX11K', 'manifests/train.csv', config)
image, label = train[0]  # float32 CHW, normalizado: (3, 512, 512)
metadata = train.metadata(0)  # cajas ORIGINALES, no ajustadas al tensor aumentado
```

El lector acepta solo manifiestos supervisados. Rechaza test oficial, externos
sin etiqueta y aumentos en validation/test. Usa RGB y 512²; para otras proporciones
aplica padding sin recorte. Train emplea brillo/contraste ±8 % y rotación ±5° con
expansión para conservar el campo. Sin espejado, recorte agresivo, deformación o
borrado. CLAHE está desactivado.

Se normaliza RGB en [0,1] con media `[0.485, 0.456, 0.406]` y desviación
`[0.229, 0.224, 0.225]`, compatibles con la normalización documentada de
[ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html).
La resolución de 512 y conservación del campo difieren de su preset de inferencia
resize/crop a 224. La Fase 2 deberá confirmar los requisitos de los pesos concretos
de ambos modelos.

Los pesos para una futura pérdida se calculan solo desde train. La augmentación
es reproducible por seed, época y ruta. En Fase 2 se llamará `set_epoch(epoch)`
antes del iterador, con `persistent_workers=False` para propagar la época a los
workers nuevos. No se ha implementado entrenamiento.

## Manifiestos y anotaciones

Cada `relative_path` parte de la raíz TBX11K: `imgs/tb/tb0004.png`, por ejemplo.

| Campos | Significado |
|---|---|
| `class_name`, `label`, `split` | Tres clases y partición interna |
| `sha256`, `rgb_sha256`, `bytes`, `width`, `height`, `mode` | Huellas y metadatos reutilizados |
| `group_id` | Componente de duplicados confirmados, no paciente |
| `official_memberships`, `alias_official_memberships` | Listas oficiales del representante y sus copias |
| `exact_aliases_json` | Rutas originales representadas por ese contenido |
| `coco_file`, `coco_image_id`, `boxes_json`, `bbox_count` | Enlace y copia fiel de metadatos COCO |
| `tb_subtypes_json`, `annotation_status` | Subtipos separados e incidencias |
| `overlaps_official_test`, `overlaps_reserved_external` | Solapamientos de contenido conocidos |

Lea columnas `_json` con `json.loads`, nunca `eval`. Las cajas usan `xywh` en la
imagen original de 512², con todos sus campos originales. No se convirtieron desde
XML ni se modificaron COCO/XML. El futuro detector deberá transformar cajas junto
con la geometría; el lector actual es exclusivamente de clasificación.

`all_labeled_members.csv` mantiene las 8.400 rutas originales, representante,
grupo y split. `manifest_metadata.json` registra cuotas, conteos, algoritmo y hashes
de entradas/salidas. Los 30 casos mixtos conservan ambos subtipos. `tb1199` está en
train como TB, sin cajas y con `missing_tb_boxes`; no es un negativo de detección.

## Test oficial y validación externa futura

`official_test_unlabeled.csv` conserva 3.302 imágenes con etiquetas vacías.
No se usan para entrenar o calcular métricas supervisadas. Hay 93 solapamientos
exactos/cercanos detectados con datos internos, registrados sin propagar etiquetas.

`reserved_external.csv` excluye las 576 imágenes `extra/`: 400 de la carpeta
Montgomery/Shenzhen y 176 DA/DB. Conserva su pertenencia a listas originales.
No se detectaron solapamientos exactos/cercanos confirmados entre extras y datos
internos dentro del alcance revisado; esto no prueba independencia por paciente.
`external_overlap_report.csv` documenta duplicados DA/DB entre sí y frente al test
oficial.

Antes de evaluar externamente: verificar etiquetas y procedencia, ampliar la
búsqueda de duplicados contra todo el entrenamiento, excluir casos solapados del
análisis externo y definir la unidad de evaluación. No usar externos para elegir
modelos, aumentos o umbrales. El test anonimizado no permite reconstruir aquí los
datasets externos completos. No se infieren etiquetas desde nombres de archivo.

## Verificaciones locales reproducibles

Entorno probado: Python 3.14.7, NumPy 2.5.4 y Pillow 12.3.0. No necesita PyTorch.
Si faltan dependencias ligeras:

```bash
python -m pip install -r requirements-data.txt
```

Desde la raíz del proyecto:

```bash
python scripts/validate_data.py --dataset TBX11K --hash-samples 2 --output reports/phase1/integrity.json
python -m unittest discover -s tests -v
python scripts/check_phase1_notebook.py --execute-local
python scripts/make_data_figures.py
```

Para reproducir en otra carpeta usando auditoría y decisiones existentes:

```bash
python scripts/build_manifests.py --dataset TBX11K --output manifests_reproduced
```

Las pruebas comprueban reproducción idéntica, integridad, exclusión de fuentes,
transformaciones, pesos y copia/extracción sobre un fixture diminuto.
El verificador del notebook compila todas sus celdas y ejecuta las ramas locales:
no importa torch, monta Drive ni entrena. Guarda evidencias en
`reports/notebook_runs/` y `reports/phase1/notebook_validation.json`.

No vuelva a ejecutar la auditoría completa o el refinamiento para una ejecución
ordinaria. El refinamiento leyó únicamente las 4.020 imágenes candidatas de dHash
igual. Las decisiones visuales quedan en `reports/near_duplicates/review_decisions.csv`.
Rehacerlas exige revisar sus pares/hashes; no se confirman automáticamente.

Para regenerar el paquete pequeño tras cambios documentados:

```bash
python scripts/package_colab.py
```

## Organización y límites

```text
TBX11K/                         # originales, solo lectura
notebooks/                      # Fase 1
src/tbxdata/                     # preparación, lectura, validación, figuras y staging
configs/data.json               # protocolo centralizado
manifests/                      # particiones, aliases y reservas
reports/initial_audit/           # auditoría previa reutilizada
reports/near_duplicates/         # refinamiento y revisión visual
reports/phase1/                  # decisiones, integridad y figuras
scripts/                        # comandos y generadores
tests/                          # pruebas ligeras
dist/                           # paquete pequeño para Drive
```

La búsqueda cercana no es exhaustiva y no garantiza independencia por paciente.
La revisión visual de Codex no sustituye revisión clínica. El test interno tiene
120 imágenes TB. Las métricas futuras necesitarán incertidumbre y atención a la
dependencia dentro de grupos; conviene reportar también métricas por grupo y usar
remuestreo por grupo.

La [ejecución entregada de Colab](reports/phase1/colab_review/revision_resultados.md)
ya permitió comprobar montaje/copia temporal, versiones del runtime, lote CPU
PyTorch y exportación a Drive según sus salidas. Los siete manifiestos coinciden
byte a byte con los congelados. La ejecución comprobó hashes de 27 muestras;
queda pendiente comparar todos los archivos transferidos antes de entrenar.
Entrenamiento, pesos y compatibilidad CUDA/A100 no se han probado; la Fase 2 no se
ha comenzado. Los apartados históricos de validación local conservan su alcance.
[Documentación de almacenamiento de Colab](https://research.google.com/colaboratory/faq.html).

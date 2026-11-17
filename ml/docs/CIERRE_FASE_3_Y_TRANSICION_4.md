# Culminar Fase 3 y comenzar Fase 4

Revisión del 2 de noviembre de 2026. Esta entrega prepara el trabajo restante;
no afirma que Vision Mamba o el híbrido ya se hayan entrenado en Colab.

## Situación comprobada

Las fases 1 y 2 ya cuentan con ejecución y hashes de datos. ConvNeXt-Tiny dispone
de dos corridas exploratorias guardadas en Drive. La v02 continúa desde el mejor
estado de v01: sus épocas no forman un experimento independiente desde ImageNet.

| Referencia | Mejor época de la corrida | Macro-F1 validation | Sensibilidad TB | TB→sick | TB→health |
|---|---:|---:|---:|---:|---:|
| ConvNeXt v01, 3 épocas | 2 | 0,9211 | 0,7333 | 24 | 8 |
| ConvNeXt v02, 5 épocas adicionales | 4 | 0,9488 | 0,8250 | 19 | 2 |

La mejora v02 es evidencia real de iteración. El estado seleccionado es el de la
época 4 de v02, no la última época 5. No se concluye convergencia ni superioridad
clínica con estas métricas de validación. No se ha abierto el test.

La fase 3 del plan exige **tres clasificadores entrenados y comparados**. La fase
4 corresponde al detector. La fase 5 integra selección final, umbral,
concordancia, explicabilidad y evaluación reservada. Completar solo ConvNeXt no
cierra la fase 3 tal como está escrita en el plan.

## Entregables preparados

- `notebooks/04_cierre_fase3_clasificadores.ipynb`: entorno independiente, kernel
  CUDA, pesos, smoke por arquitectura, tres entrenamientos, recuperación y cierre.
- `notebooks/05_fase4_detector_TB.ipynb`: conversión de cajas, revisión visual,
  smoke opcional y entrenamiento/validación del detector, inicialmente desactivados.
- `configs/phase3_suite.json`: protocolo común fijado antes de esta tanda.
- `src/tbxtrain/architectures.py`: ConvNeXt, VMamba y fusión por atención cruzada.
- `src/tbxtrain/suite.py`: entrenamiento, versionado y cierre con comprobaciones.
- `src/tbxtrain/detection.py`: conversión de detección con trazabilidad de exclusiones.
- `vendor/vmamba/`: código oficial fijado, hashes, procedencia y licencia MIT.
- `dist/tbx11k_fases_3_y_4_colab.zip`: paquete nuevo; no reemplaza los ZIP anteriores.

Los notebooks 02/03 ya ejecutados y sus outputs se conservan. Abra los nuevos
notebooks por separado y guarde cada ejecución con Ctrl+S. Los modelos y sus
carpetas de Drive tienen IDs nuevos; no se sobrescribe v01/v02.

## Diseño de la comparación de Fase 3

Se ejecuta una tanda nueva para distinguir arquitectura de historia de ajustes:

1. **ConvNeXt-Tiny** desde IMAGENET1K_V1 oficial, con cabeza de tres clases.
2. **VMamba-T s1l8** oficial desde ImageNet-1K, con cabeza de tres clases.
3. **Híbrido** con ambas ramas inicializadas en ImageNet, mapas espaciales
   proyectados a dimensión 256 y atención cruzada bidireccional de cuatro cabezas.
   Los mapas se agrupan a una rejilla de 8×8 para limitar memoria. Se concatenan
   los tokens medios actualizados y se clasifican. No es un promedio de scores.

El híbrido no hereda ramas ya entrenadas en TBX11K: eso daría una historia de
optimización diferente. La tanda nueva no elimina la utilidad del ConvNeXt v02,
que sigue disponible como referencia exploratoria para la futura selección.

| Elemento común | Protocolo |
|---|---|
| Clases | health=0, sick=1, TB=2 |
| Datos | Train 5.792, validation 1.241, mismos manifiestos y grupos |
| Entrada | RGB 512×512, letterbox y normalización ImageNet |
| Aumentos | Los congelados de Fase 2, solo train; sin CLAHE/flips |
| Presupuesto máximo | 12 épocas por arquitectura |
| Warmup | 2 épocas de cabeza/fusión, con extractores congelados |
| Fine tuning | Cabeza/fusión y últimas etapas de cada extractor |
| Optimizador | AdamW; LR cabeza 3e-4, extractor 5e-6; weight decay 0,01 |
| Precisión | FP32 en todos, conservando la corrección del smoke anterior |
| Lote | 2 imágenes, acumulación 16; último grupo puede ser menor |
| Desbalance | Cross entropy ponderada con pesos calculados solo en train |
| Selección | Macro-F1 en validation; mejora mínima 0,001 |
| Parada | Paciencia 4; épocas efectivas registradas |
| Seed | 20261019, una seed para esta tanda |

La comparación controla datos y reglas de entrenamiento; no iguala número de
parámetros, coste computacional o cantidad de preentrenamiento original de los
pesos oficiales. Las arquitecturas pueden requerir distintos ajustes. Una seed
y una búsqueda limitada no demuestran superioridad estadística. En Fase 5 se
deberá considerar sensibilidad TB, falsos positivos e incertidumbre.

El protocolo de datos tiene tres clases; la mención a cuatro salidas en versiones
del plan debe corregirse/documentarse. No se añade una cuarta clase sin resolver
sus etiquetas ni se crean nuevas particiones para esta tanda.

## Ejecutar y recuperar en Colab

1. El ZIP nuevo estará en `MyDrive/Proyecto_Tuberculosis/`, junto al TBX11K.zip
   existente. Abra el notebook 04 en Colab con GPU disponible.
2. Ejecute montaje, dependencias, transferencia y verificación de hashes.
3. Ejecute **F3-C1**: compila selective scan oficial para torch/CUDA y GPU actuales.
   Se usa el código fijado, no una wheel antigua de Python/torch. La compilación
   deja log en `/content/tbx_vmamba_build/<revision>/build.log`.
4. Los pesos oficiales se guardan en `Drive/pretrained/`. El SHA-256 debe coincidir
   con el checkpoint comprobado localmente y la arquitectura carga estrictamente.
5. Ejecute **F3-C2**, los smokes. Deben completar actualización y recarga de pesos.
   El backend de referencia PyTorch se usa para fixtures CPU, no como sustituto
   silencioso del kernel GPU para entrenar todo el dataset a 512px.
6. Ejecute **F3-C3**. Cada arquitectura guarda por época. Al terminar, el siguiente
   modelo comienza sin depender de variables de las corridas anteriores.
7. Tras desconexión, repita preparación del runtime y kernel. Puede omitir los
   smokes ya comprobados mediante `RUN_SMOKES=False`. F3-C3 consulta el registro
   persistente, recupera corridas terminadas y reanuda las interrumpidas desde su
   última época completa. Una época parcial se repite.
8. Mantenga iguales configuración y código al reanudar. Una tanda diferente
   necesita otro `protocol_id`. No sustituya el paquete durante una tanda activa.

El runtime y su kernel deben verificarse en Colab: las pruebas locales en CPU
no prueban disponibilidad de GPU, nvcc ni compatibilidad de compilación CUDA.
La fuente oficial declara kernels Triton de forma incondicional; el adaptador
protege esas declaraciones en CPU y delega selective scan en el csms6s oficial.
En GPU con Triton se usan los kernels originales. Estos cambios no sustituyen
VMamba por otra arquitectura ni cambian la ecuación de su selective scan.

## Cuándo se puede declarar terminada la fase 3

La celda **F3-C4** requiere tres corridas con:

- Arquitecturas diferentes y requeridas, entrenamiento real y validación completa.
- Mismo protocolo, fuentes y hashes de datos/código.
- `best.pt` presente y con el SHA-256 declarado por cada corrida.
- Test y externos sin evaluar.

Genera `results/phase3_closure_<UTC>/comparison.csv` y `closure.json` con
`phase3_complete=true`. También conserva la evidencia de integridad. Si falta
un modelo, hay una corrida fallida/smoke o el protocolo difiere, el cierre falla
y no produce una afirmación de fase completa.

Cada corrida conserva entorno, configuración, historial, matriz, ROC y
predicciones de validation, más `last.pt` (modelo/optimizador/RNG/mejor estado) y
`best.pt`. El registro persistente está en `results/phase3_suite/<protocol_id>/`.
El cierre documenta comparación preliminar; deja `winner_finalized=false` para
la fase 5 y no abre el test.

## Inicio de Fase 4

El notebook 05 prepara **YOLO11n** pretrained COCO con una clase `tb_lesion`.
Su paquete Ultralytics queda fijado a 8.3.228; la licencia de Ultralytics es
AGPL-3.0 y debe tenerse en cuenta al distribuir el sistema posterior.

La conversión usa train/validation originales y sus cajas COCO `xywh`.
Las convierte a centro/anchura/altura normalizados y mantiene imágenes originales:

- TB con cajas: positivo con todas las cajas TB, unificando subtipos en la clase
  de lesión. La procedencia y subtipos originales no se reescriben.
- health/sick: negativos sin cajas TB; no se inventan rectángulos para esas clases.
- TB sin cajas (`tb1199`, train): excluida de detección; no es un negativo.

Se guardan `conversion_ledger.csv`, conteos, YAML y revisión visual de cajas.
El smoke usa muestras positivas y negativas explícitas; no se usa un prefijo del
dataset que pudiera contener solo negativos. Las métricas smoke no son resultados.

`RUN_DETECTOR_SMOKE=False` y `RUN_DETECTOR_TRAINING=False` permiten detenerse tras
la preparación/revisión. Al activarlos, el entrenamiento completo tiene hasta
35 épocas y paciencia 7, imagen 512 y FP32. Geometrías agresivas, flips, mosaico y
mixup están desactivados. El detector transforma imagen/cajas conjuntamente.

Los resultados viven en una carpeta nueva `experiments/detector_yolo11n_<UTC>/`.
Se guardan pesos por época, best/last y mAP de validation. El YAML no incluye test.
La fase 4 no se declara completada por convertir datos ni por un smoke.

## Qué queda para Fase 5

Seleccionar la solución con evidencia de validation y del detector; documentar
el caso de conservar ConvNeXt si el híbrido no mejora. Definir umbral de alerta
y concordancia clasificación–detección sin alterar probabilidades arbitrariamente.
Incorporar Grad-CAM donde corresponda, revisar discrepancias y explicar sus límites.
Después congelar la decisión y usar una sola vez el test reservado para reportar
métricas finales e incertidumbre, con atención a grupos/dependencia por paciente.
Exportar pesos y configuración final. Los externos requieren la revisión de
procedencia/solapamientos pendiente de las fases previas.

## Evidencias para el informe

Conserve el notebook 04 ejecutado, carpeta de cierre y tres carpetas de corridas.
Para inicio de Fase 4: conversión y captura de cajas, y si ya se entrenó, curvas,
mAP validation y pesos. Un notebook fuente sin ejecutar demuestra implementación,
pero no permite declarar entrenamientos ni métricas obtenidas.

Fuentes oficiales: [VMamba y modelos](https://github.com/MzeroMiko/VMamba),
[revisión de código fijada](https://github.com/MzeroMiko/VMamba/tree/2ed52ead062a51a64521ed3871d52914bf532876),
[YOLO11](https://docs.ultralytics.com/models/yolo11/),
[formato de detección](https://docs.ultralytics.com/datasets/detect/).

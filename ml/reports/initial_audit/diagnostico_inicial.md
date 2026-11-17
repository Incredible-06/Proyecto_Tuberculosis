# TBX11K: inspección local y diagnóstico inicial

Fecha: 19 de octubre de 2026 (America/Lima).

**Hallazgo principal:** las listas oficiales no comparten rutas, pero sí imágenes
idénticas con nombres diferentes. Hay 38 pares duplicados entre entrenamiento y
validación oficiales. Antes de entrenar conviene definir particiones sin esta fuga.

Este entregable se limita a la inspección solicitada. Las propuestas siguientes
todavía no se han convertido en particiones, transformaciones o notebooks.

## 1. Alcance y evidencias

Dataset encontrado en `TBX11K/` dentro del workspace. Se inspeccionaron su README,
siete listas, 800 XML, diez JSON COCO y las 12.278 imágenes bajo `imgs/`.
Solo hay utilidades originales de conversión de anotaciones en `TBX11K/code/`;
no se encontró una implementación de ConvNeXt, VMamba o MedMamba en el proyecto.

La auditoría empleó Python 3.14.7 y Pillow 12.3.0, un único proceso y una imagen a
la vez, con una pausa de 0,01 s por archivo. La exploración de imágenes duró
aproximadamente cuatro minutos. No se importó PyTorch ni se cargaron modelos,
usó CUDA, descargaron pesos o ejecutó entrenamiento.

Evidencias reproducibles:

- `audit_summary.json`: estadísticas, inconsistencias y todas las agrupaciones de duplicados.
- `image_inventory.csv`: ruta relativa a `TBX11K/imgs`, tamaño, dimensiones, modo, formato, listas oficiales y hashes.
- `metadata_sha256.json`: trazabilidad del README, listas, anotaciones y código original.
- `representative_samples.jpg` y `sample_records.json`: muestras y sus cajas COCO.
- `scripts/audit_dataset.py` y `scripts/summarize_audit.py`: código reproducible, ejecutado desde la raíz del proyecto.

## 2. Estructura y distribución observada

```text
TBX11K/
  README.md, TBX11K_CVPR2020.pdf, teaser.jpg
  imgs/
    health/                 3800
    sick/                   3800
    tb/                      800
    test/                   3302
    extra/da+db/train/         88
    extra/da+db/val/           88
    extra/mc+shenzhen/train/  200
    extra/mc+shenzhen/val/    200
  lists/                       7 TXT
  annotations/xml/           800 XML
  annotations/json/           10 JSON
  code/                        3 utilidades
```

Se contaron únicamente imágenes bajo `imgs/`; `teaser.jpg` es una ilustración
del paquete y no se incluyó. Tamaño de las imágenes: 4.241.167.568 bytes, unos
4,24 GB decimales. Todas son **512 × 512 y modo RGB**; 12.174 PNG y 104 JPEG.
Los JPEG aparecen en `extra/da+db`. En las diez muestras del mosaico, los tres
canales son iguales píxel a píxel: contenido monocromático almacenado como RGB.
Esa comprobación de igualdad de canales fue muestral, no sobre todo el dataset.

El README describe 11.200 imágenes TBX11K base y datos externos. La estructura
local mezcla el test base con datos externos bajo nombres `unknown_*`, por lo que
no corresponde interpretar los 12.278 archivos como 12.278 casos TBX11K base.
Tampoco hay identificadores verificables que permitan separar el origen de cada
archivo de test. La suma nominal de los cuatro datasets externos citados en el
README es 1.106; la diferencia total respecto a 11.200 es 1.078. Hay una diferencia
documental de 28 imágenes que no puede atribuirse a archivos concretos con los
metadatos disponibles; no demuestra por sí sola una descarga incompleta.

Distribución de las listas base, antes de controlar duplicados:

| Lista | Saludable (`health`) | Enfermo sin TB (`sick`) | TB (`tb`) | Total |
|---|---:|---:|---:|---:|
| TBX11K_train | 3.000 | 3.000 | 600 | 6.600 |
| TBX11K_val | 800 | 800 | 200 | 1.800 |
| TBX11K_trainval | 3.800 | 3.800 | 800 | 8.400 |

TB supone el 9,52 % del conjunto etiquetado base: un desbalance aproximado de
4,75:4,75:1. Las listas `all_train` y `all_val` añaden 288 imágenes externas cada
una y suman 6.888 y 2.088. `all_trainval` contiene 8.976 imágenes; `all_test`, 3.302.
No se han inferido etiquetas de imágenes externas a partir de prefijos `n/p` o
sufijos de sus nombres.

Las siete listas no tienen rutas repetidas ni archivos ausentes. Sus uniones
trainval son consistentes. Todas las imágenes pertenecen a alguna lista.

## 3. Etiquetas y anotaciones

Las tres carpetas etiquetadas permiten plantear una clasificación de imagen
saludable / enfermo sin TB / TB, conservando la semántica declarada por el paquete.
Los subtipos de TB de los XML son **etiquetas de lesiones**:

| Etiqueta original | Objetos | Imágenes con solo esa etiqueta |
|---|---:|---:|
| ActiveTuberculosis | 972 | 630 |
| ObsoletePulmonaryTuberculosis | 239 | 139 |
| PulmonaryTuberculosis | 0 | 0 |

Otras 30 imágenes contienen simultáneamente objetos activos y antiguos: 23 en
train y siete en val. Una imagen tiene XML vacío: **`tb/tb1199.png`**, en train.
Las cifras por imagen suman 630 + 139 + 30 + 1 = 800; por objeto suman 1.211.
El código original relaciona `ObsoletePulmonaryTuberculosis` con «Latent TB» y
`PulmonaryTuberculosis` con «Unknown TB». Se recomienda conservar esos nombres
como etiquetas del dataset, sin derivar un diagnóstico clínico adicional.

Consecuencias:

- No construir automáticamente cinco clases exclusivas a partir de las tres categorías COCO y las carpetas health/sick: existen casos mixtos y una categoría sin ejemplos.
- Mantener `tb1199` como TB para clasificación según la carpeta, con bandera de anotación ausente y revisión pendiente; no convertirlo en negativo para detección ni inventar una caja.
- No usar los JSON `*_only_tb` para construir el clasificador: excluyen negativos y omiten `tb1199`. Train incluye 599 imágenes con cajas y val, 200.
- Los 576 archivos externos no tienen XML ni cajas en los JSON examinados. Ausencia de caja no equivale a ausencia de TB.

Los XML contienen dimensiones de origen variables, con 291 combinaciones
distintas; por ejemplo `tb0003.xml` declara 2.840 × 2.827 aunque el PNG es 512².
Sus cajas no pueden dibujarse directamente sobre los PNG. Los COCO ya contienen
coordenadas para 512². Se verificaron las 1.211 cajas contra la conversión del
código original, con tolerancia de 0,001 píxeles: ninguna discrepancia.

La conversión original aplica `x = xmin * 512 / ancho_XML - 1`,
`y = ymin * 512 / alto_XML - 1`, `w = (xmax-xmin) * 512 / ancho_XML` y
`h = (ymax-ymin) * 512 / alto_XML`. Conviene reutilizar las cajas COCO verificadas,
sin aplicar de nuevo ese escalado ni otra corrección de índices.

No se detectaron cajas inválidas, objetos `difficult`, IDs COCO duplicados,
referencias inválidas, diferencias entre listas y sus JSON equivalentes, ni
desajustes entre las dimensiones COCO y los archivos. Los 800 XML corresponden
exactamente a las 800 imágenes TB etiquetadas.

## 4. Integridad, duplicados y fuga de datos

Todas las imágenes superaron `Pillow.verify()` y una decodificación completa
mediante `load()`: no se detectaron archivos corruptos o ilegibles.

Se calcularon SHA-256 del archivo y de los píxeles RGB, incluyendo dimensiones.
Ambos métodos encontraron los mismos **216 grupos de dos archivos idénticos**:
432 archivos implicados, 216 copias redundantes y 12.062 contenidos únicos en
todo `imgs/`. No se observaron conflictos entre etiquetas de carpetas base dentro
de los grupos exactos. Esto no valida las etiquetas desconocidas del test.

| Cruce de listas base y test disponible | Pares exactos |
|---|---:|
| TBX11K_train ↔ TBX11K_val | 38 |
| TBX11K_train ↔ all_test | 64 |
| TBX11K_val ↔ all_test | 9 |

Ejemplos verificables mediante los hashes del inventario:

- `sick/s0055.png` y `sick/s0480.png`: train ↔ val.
- `sick/s0189.png` y `test/unknown_3221.png`: train ↔ test.
- `sick/s0143.png` y `test/unknown_1705.png`: val ↔ test.
- `extra/da+db/train/n28.png` y `extra/da+db/val/n1.png`: fuga también entre extras.

En las 8.400 imágenes etiquetadas base hay **8.274 contenidos únicos**:
3.800 health, 3.674 sick y 800 tb. Las 126 repeticiones internas detectadas en este
conjunto pertenecen a sick. No se eliminó ningún archivo.

Como filtro exploratorio adicional, dHash de 64 bits produjo 1.213 grupos con
hash perceptual igual. Es una representación muy gruesa y hay colisiones entre
radiografías diferentes; **no son 1.213 duplicados confirmados**. No se hizo una
búsqueda exhaustiva de vecinos con distancias Hamming distintas de cero ni una
revisión visual completa de esos candidatos. Deben priorizarse comparaciones más
discriminantes y revisión de pares sospechosos antes de fijar las particiones.

No hay un campo de paciente verificable en XML, JSON o listas. Los nombres y las
fechas de generación COCO no prueban identidad del paciente. Por tanto, eliminar
duplicados exactos no permite garantizar independencia por paciente o estudio.

## 5. Test disponible y limitación de evaluación

`all_test.json` enumera 3.302 imágenes y **cero anotaciones**; los nombres están
anonimizados y no existe una tabla local de etiquetas. El README indica que el
ground truth del test no se publica. En consecuencia, no se pueden calcular
accuracy, sensibilidad, especificidad, F1 o AUC supervisadas sobre ese test con
esta descarga. Tampoco deben propagarse etiquetas desde sus duplicados conocidos.

Además de esta ausencia de etiquetas, los duplicados encontrados impiden asumir
que una evaluación oficial sea independiente del entrenamiento local sin medidas
adicionales. El test oficial podrá conservarse para inferencia o un protocolo
externo documentado, separado del test interno propuesto a continuación.

## 6. Estrategia recomendada antes de implementar

### Clasificación y particiones

1. Baseline principal de **tres clases** con las imágenes base etiquetadas. Mantener los subtipos, las cajas y las banderas de incidencias como metadatos separados. Dejar `extra/` para una comparación posterior entre fuentes.
2. Conservar las listas originales y su pertenencia en columnas de trazabilidad. Debido a la fuga comprobada, recomendar particiones propias estratificadas aproximadamente **70/15/15**, sobre los 8.274 contenidos únicos etiquetados, agrupando por hash y revisando candidatos cercanos. No se especifican todavía conteos finales ni se genera el split.
3. En el futuro manifiesto, conservar un representante determinista por contenido para evitar sobreponderar copias, registrando todas las rutas equivalentes; nunca borrar originales. Los grupos de duplicados cercanos confirmados deberán permanecer juntos. Esta agrupación será por imagen, no por paciente.
4. Fijar seed, algoritmo, política de duplicados y manifiestos antes de entrenar ambos modelos. Usar validación para decisiones e hiperparámetros; mantener cerrado el test interno hasta la comparación final. Informar que este protocolo no reproduce los splits oficiales.
5. Calcular pesos de pérdida únicamente a partir del train definitivo. Comenzar con entropía cruzada ponderada; evitar combinar de entrada sobremuestreo y pesos fuertes. Reportar métricas macro y por clase, incluyendo sensibilidad TB y falsos negativos. Cualquier umbral se decide solo en validación.

La alternativa de conservar las listas oficiales exigiría al menos excluir del
entrenamiento las copias de validación y gestionar las copias del test. Seguiría
faltando un test etiquetado independiente. Por eso se propone el protocolo propio
para el objetivo académico de esta fase, conservando toda la trazabilidad oficial.

### Preprocesamiento

- Mantener 512 × 512 como resolución inicial: ya es la resolución distribuida; un resize no puede recuperar las proporciones originales perdidas. Si una GPU requiere menos resolución, registrar ese cambio y aplicarlo a ambos modelos.
- Convertir a RGB de forma defensiva en el lector futuro; aquí los archivos ya son de tres canales. No aplicar conversión obligatoria a escala de grises ni recortes del tórax.
- Usar la normalización documentada para los pesos preentrenados que finalmente se verifiquen. La resolución, población, aumentos y política de selección serán comparables; cualquier requisito distinto entre modelos se registrará.
- Probar aumentos moderados solo en train, por ejemplo pequeñas rotaciones y variaciones acotadas de brillo/contraste, comprobándolos visualmente sobre pocas muestras. Evitar inicialmente espejado, recortes aleatorios agresivos, deformaciones y borrado de regiones.
- CLAHE queda como ablación opcional y se elige solo con validación. Val y test tendrán transformaciones deterministas, sin aumentos.
- Mantener intactos XML y COCO. Para un detector futuro, toda transformación geométrica deberá actualizar también las cajas; no dibujar las cajas dentro de las imágenes de entrada del clasificador.

### Modelos y entorno, pendientes de la siguiente etapa

Preparar pipelines independientes de ConvNeXt-Tiny y Vision Mamba con los mismos
manifiestos y protocolo. Antes de elegir VMamba-T o MedMamba, verificar repositorio,
licencia, pesos, versión/commit y requisitos de sus extensiones CUDA. Esa
investigación y la selección de dependencias aún no se han realizado. No se
sustituirá Vision Mamba silenciosamente si hay incompatibilidades.

Solo en Colab: carga de modelos completos, descarga de pesos, compilación de
extensiones, detección y compatibilidad CUDA, smoke test de entrenamiento, AMP,
memoria disponible, entrenamiento y evaluación final. La GPU se detectará sin
suponer A100. Guardado en Drive y reanudación deberán incluir modelo, optimizador,
scheduler, scaler AMP, época, configuración y estados aleatorios.

## 7. Muestras revisadas y siguiente paso

El mosaico contiene diez imágenes del **train oficial**: dos health, dos sick,
dos con lesiones activas, dos con lesiones antiguas y dos mixtas. Rojo indica
ActiveTuberculosis y cian ObsoletePulmonaryTuberculosis. Las cajas se tomaron del
COCO verificado y se escalaron únicamente para la visualización. La selección es
ilustrativa y determinista, no una muestra estadística ni una revisión clínica.
No se usaron muestras del test para decidir transformaciones.

El siguiente entregable será `01_exploracion_preprocesamiento.ipynb`, una vez
revisadas estas decisiones sobre clases, duplicados y partición propia. Después
se preparará `02_entrenamiento_ablacion.ipynb`. Ningún resultado de entrenamiento,
rendimiento en GPU o compatibilidad A100 está validado en esta etapa.

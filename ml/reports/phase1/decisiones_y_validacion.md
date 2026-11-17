# Documento histórico del pipeline de datos del 19 de octubre

Este registro conserva la numeración técnica antigua y los pendientes de esa fecha.
En el plan académico estos datos corresponden a la Fase 2. Las comprobaciones de
Colab, GPU y hashes completos se acreditaron después; el estado vigente está en
`docs/CIERRE_FASES_1_Y_2.md` y `reports/revision_20261021/verification.json`.
Montgomery y Shenzhen completos se incorporaron separados en la revisión del 21.

Fecha de referencia: 19 de octubre de 2026, America/Lima. Se implementó únicamente
el pipeline de datos. No se entrenaron modelos ni se descargaron pesos.

## Reutilización de la auditoría y duplicados cercanos

Se reutilizaron los hashes, inventario, listas y estadísticas existentes. No se
repitió la auditoría de las 12.278 imágenes. Solo se leyeron las **4.020 imágenes
incluidas en candidatos de dHash igual**, de una en una, con un hilo de cálculo
NumPy y pausas breves. No se usaron modelos o embeddings pesados.

Se refinaron 13.783 pares de 1.213 grupos mediante pHash DCT de 256 bits. Los 216
pares exactos ya conocidos se identificaron desde el inventario. Otros 13.523
pares superaron la distancia máxima de 32 bits y se descartaron como candidatos
de esta búsqueda, sin afirmar que ese descarte pruebe independencia de paciente.
Los **44 pares restantes** tuvieron distancia pHash de 0–18, SSIM local a 128²
de 0,988–0,9999 y NCC mayor de 0,98. Se calcularon también error absoluto y RMSE.

Se revisaron visualmente los 44 pares en seis láminas; los tres de menor SSIM se
inspeccionaron además a 512² junto con mapas de diferencia. Se observó el mismo
contenido radiográfico con cambios localizados de enmascarado/intensidad. Los 44
se confirmaron como duplicados cercanos para prevenir fuga, **sin inferir identidad
de paciente ni un diagnóstico clínico**. SSIM a 512²: 0,98565–0,99939; como máximo
el 1,093 % de píxeles de un par difirió más de 5/255. Se revisaron también ocho
controles de dHash coincidente rechazados por pHash: mostraron imágenes distintas.
Estos controles no se agruparon.

Las decisiones las realizó Codex mediante inspección visual, no un especialista
clínico ni una aprobación humana individual de cada par. Se preservan:

- `reports/near_duplicates/refined_pairs.csv`: métricas y umbrales de los pares refinados.
- `review_decisions.csv`: decisión visual, rutas, hashes y fundamento por par.
- `full_resolution_metrics.csv`: comprobación original de los 44 pares.
- `suspicious_01.jpg` a `suspicious_06.jpg`: láminas revisadas.
- `detail_lowest_similarity.png`: comparación ampliada de los tres pares más diferentes.
- `rejected_controls.csv` y `rejected_controls_01.jpg`: controles del descarte.

Los umbrales son reglas de auditoría configuradas para este conjunto, no un
criterio clínico universal. Esta búsqueda **solo refina candidatos de dHash igual**:
puede omitir duplicados con otro dHash, recortes, distinta proyección o cambios
mayores. La agrupación conocida no garantiza independencia por paciente.

## Particiones congeladas y trazabilidad

Se utilizaron solo las tres carpetas base etiquetadas: health=0, sick=1 y TB=2.
Se conservó un representante lexicográfico por hash RGB exacto, manteniendo las
8.400 rutas originales como aliases en el índice de miembros. No se borró nada.
Las 23 parejas cercanas entre contenidos etiquetados se mantienen como imágenes
distintas dentro de un grupo indivisible: **8.274 contenidos y 8.251 grupos**.

| Partición | health | sick | TB | Total |
|---|---:|---:|---:|---:|
| train | 2.660 | 2.572 | 560 | 5.792 |
| validation | 570 | 551 | 120 | 1.241 |
| test interno | 570 | 551 | 120 | 1.241 |

Cuotas por clase 70/15/15, redondeo por restos mayores y asignación de grupos
indivisibles. Los grupos se ordenan por tamaño y una clave SHA-256 dependiente de
seed/clase/grupo; se asignan según cuota proporcional pendiente. Seed 20261019.
No hay identificadores aleatorios dependientes del hash de Python. En este caso
se alcanzaron exactamente las cuotas redondeadas de las tres clases.

Las listas oficiales se preservan como procedencia, no como partición actual. Se
eligió un protocolo propio por la fuga de duplicados comprobada. El test interno
queda cerrado para selección de modelos, hiperparámetros o transformaciones.
Solo se visualizan radiografías de train; las verificaciones de test son de
integridad y distribución, no evaluación de modelos.

Cada manifiesto contiene rutas relativas a la raíz de TBX11K, clase/etiqueta,
split, hashes, tamaño, grupo, aliases exactos, listas originales y metadatos COCO.
`manifest_metadata.json` conserva fingerprints de configuración, decisiones,
inventario, código de preparación y salidas. La reproducción local en otra
carpeta produjo metadatos y CSV **idénticos byte a byte**.

## Anotaciones preservadas

Los subtipos y las cajas se copiaron como metadatos del COCO original y se
comprobaron contra él. No hubo nueva conversión XML ni modificación de fuentes.
Las cajas conservan `xywh`, IDs, categoría y el resto de campos originales sobre
512². Los subtipos no se fuerzan a ser mutuamente excluyentes: se conservan los
30 casos mixtos. `tb1199` permanece en train con clase TB y bandera de cajas
ausentes; no puede usarse automáticamente como negativo de detección.

El lector entrega únicamente matriz de clasificación y etiqueta. Sus metadatos
se consultan por separado y declaran que las cajas corresponden a la imagen
original, no a una versión aumentada. La adaptación geométrica de cajas para un
futuro detector sigue fuera del alcance actual.

## Preprocesamiento y desbalance

RGB, resolución 512², arrays float32 CHW y normalización ImageNet. El tamaño ya
distribuido no recupera proporciones originales perdidas. Para otra proporción se
conserva el campo con padding. Train usa brillo/contraste ±8 % y rotación ±5° con
expansión antes de ajustar al lienzo; no hay recorte agresivo, espejado, deformación
o borrado. Validation/test son deterministas. CLAHE desactivado, como posible
ablación futura elegida únicamente en validación.

La normalización coincide con los parámetros documentados para ConvNeXt-Tiny;
el protocolo de 512² difiere deliberadamente de su preset de resize/crop a 224.
La Fase 2 deberá validar requisitos de los pesos concretos de cada modelo. Las
figuras muestran las transformaciones antes de normalización. Se comprobó el
retorno a RGB desde la normalización determinista sin diferencias de píxeles.

Los pesos de clase para una futura pérdida se calculan exclusivamente sobre los
5.792 ejemplos de train; no se aplica sobremuestreo ni aumento antes de dividir.
La augmentación depende de seed, época y ruta. `set_epoch` permite cambiarla de
forma reproducible en una futura época de entrenamiento.

## Reservas oficial y externa

El manifiesto oficial contiene las 3.302 imágenes con etiquetas vacías. **No se
les asignaron etiquetas a partir de nombres, pares cercanos o duplicados.** Se
registraron 93 imágenes con coincidencia de contenido conocida frente a datos
internos. Este test no admite métricas supervisadas con los datos disponibles y
no debe asumirse independiente de los clasificadores.

Quedan fuera de todas las particiones supervisadas las 576 imágenes de extra:
400 de la carpeta combinada Montgomery/Shenzhen y 176 DA/DB. No se observó
solapamiento exacto o cercano confirmado entre esas reservas y datos internos
dentro del alcance revisado. Hay 11 filas externas involucradas en grupos
duplicados: cuatro grupos internos DA/DB y tres pares DA/DB–test oficial.
El informe externo registra cada ruta afectada, no once pares independientes.

Una futura evaluación externa requiere etiquetas y procedencia verificadas,
revisión ampliada de duplicados y definición de unidad de evaluación. Ninguna
imagen externa se usará para ajustar modelos o preprocesamiento antes de esa
evaluación. El test anonimizado mezcla fuentes que no se pueden reconstruir de
forma fiable; esta reserva no equivale a los datasets externos completos.

## Qué se validó localmente

- Integridad de manifiestos, conteos, clases y cobertura de todas las rutas auditadas.
- Ausencia de cruces internos por ruta, SHA-256 de archivo, SHA-256 RGB y grupo de duplicados confirmado.
- Correspondencia de aliases, decisiones cercanas y listas oficiales.
- Preservación de todas las cajas/metadatos COCO usados por el clasificador.
- Existencia/tamaño de todos los archivos; hashes de muestras reproducibles, reutilizando la auditoría previa.
- Reproducción byte a byte de todos los manifiestos y su metadata en otra carpeta.
- Lectura, forma `(3,512,512)`, float32, valores finitos, determinismo y variación por época sobre pocas imágenes.
- Exclusión del test oficial y externos en el lector supervisado; rechazo de aumentos en validation/test.
- Copia, reuso de cache, detección de cambio de fuente y extracción ZIP sobre un fixture diminuto.
- Revisión visual de las figuras de distribución, cajas y aumentos moderados.
- Estructura del notebook v4.5, compilación y ejecución de sus 14 celdas de código en ramas locales, sin importar torch.

Las nueve pruebas automáticas ligeras están en `tests/test_data_pipeline.py` y sus
resultados en `reports/phase1/tests.json` y `tests.log`. La ejecución local del
notebook está registrada en `notebook_validation.json`. No se ejecutó ninguna
prueba de entrenamiento, compatibilidad CUDA o carga de modelos.

## Pendiente de ejecutar en Google Colab

Montaje y autorización reales de Drive, extracción/copia del dataset real en
`/content`, versiones efectivas del runtime, verificación íntegra de transferencia
si se activa, lote PyTorch CPU y exportación real de resultados a Drive. Las
funciones se comprobaron con datos pequeños localmente; esto no prueba el servicio
Drive ni la ejecución remota. El notebook instala solo dependencias ligeras y no
descarga modelos. Esta fase puede ejecutarse en CPU.

No se ha creado el notebook de entrenamiento ni investigado/validado VMamba o
MedMamba en esta continuación. ConvNeXt, Vision Mamba, pesos y pruebas GPU/CUDA
pertenecen a una Fase 2 posterior y se ejecutarán exclusivamente en Colab.

## Limitaciones metodológicas restantes

No hay IDs fiables de paciente/estudio. Una búsqueda no exhaustiva de duplicados
no demuestra independencia. Variantes cercanas del mismo grupo permanecen como
observaciones distintas dentro del split; los resultados futuros deberían
informarse también por grupo, con intervalos/remuestreo por grupo. El test tiene
120 contenidos TB y debe interpretarse con su incertidumbre. Los labels del
dataset no constituyen una adjudicación clínica adicional.

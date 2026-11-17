# Informe de las fases 1 y 2

Facultad de Ingeniería, Universidad Nacional del Santa

Escuela Profesional de Ingeniería de Sistemas e Informática

Desarrollo de un Sistema Web Basado en Deep Learning para el Apoyo a la Detección de Tuberculosis Pulmonar en Radiografías de Tórax

INFORME DE AVANCE DE LAS FASES 1 Y 2

Alumnos:

Villar Bazan, Ronney Junior

Calvo Andrade, Sebastián Jefferson

Rodríguez Polo, Andy Javier

Rodas Aquino, Cesar Eduardo

Grupo:

A

Docente:

Dr. Ing. Alfredo Daza Vergaray

Curso:

Ingeniería de Software II

Ciclo:

VIII

 Nuevo Chimbote

1 Fase 1 Preparación del entorno de trabajo

El objetivo de estas dos fases fue dejar listo el entorno de trabajo y preparar los datos para los próximos experimentos. El entorno funciona en Colab, las imágenes están organizadas y las comprobaciones se ejecutaron sin errores. Este informe reúne las salidas más importantes y explica qué aporta cada una.

El repositorio separa el código, los notebooks, los datos, los modelos, la API y la web. Colab se usa para ejecutar el trabajo y Drive para guardar los archivos entre sesiones. El notebook de entorno comprobó que la GPU estuviera disponible y que pudiera realizar un cálculo con CUDA.

Figura 1. Prueba del entorno en Colab. La salida muestra una GPU NVIDIA A100 y el resultado passed, que significa que la prueba CUDA se completó correctamente. La GPU pudo realizar un cálculo pequeño. El campo Entrenamiento: False aclara que esta prueba no entrenó ningún modelo.

Figura 2. Revisión de la copia del dataset. Las huellas digitales SHA-256 de las 12.278 imágenes coincidieron con el inventario original. El resultado passed confirma que los archivos comparados conservaron su contenido al copiarse a Colab.

Los resultados quedaron guardados en Drive. Las carpetas experiments y checkpoints están listas para los experimentos y los pesos futuros, por eso todavía están vacías. El siguiente paso será comprobar la compatibilidad de los modelos que se van a entrenar.

2 Fase 2 Preparación del dataset y sus categorías

El paquete reúne 12.278 imágenes de 512 × 512 píxeles. La base etiquetada incluye 3.800 imágenes saludables, 3.800 de enfermedad sin TB y 800 de TB. Se mantienen separadas 3.302 imágenes del test oficial sin etiquetas públicas y 576 adicionales. La revisión inicial no encontró errores de lectura.

Se revisaron las cuatro categorías previstas: saludable, enfermedad sin TB y los dos subtipos de TB. El clasificador inicial usará tres clases: health, sick y TB. Se agrupan los subtipos porque algunas imágenes tienen ambos; así se evita asignarles una categoría arbitraria y se conservan sus anotaciones.

| Anotación de las imágenes TB | Imágenes |
| --- | --- |
| Solo ActiveTuberculosis | 630 |
| Solo ObsoletePulmonaryTuberculosis | 139 |
| Ambos subtipos | 30 |
| TB sin cajas anotadas | 1 |
| Total TB | 800 |

Los 30 casos mixtos conservan sus dos subtipos dentro de la clase TB.

Montgomery y Shenzhen se descargaron desde NLM, con sus lecturas y documentos de uso, y quedaron separados de TBX11K.

| Validación externa | Normal | Anormal compatible con TB | Total |
| --- | --- | --- | --- |
| Montgomery | 80 | 58 | 138 |
| Shenzhen | 326 | 336 | 662 |
| Total | 406 | 394 | 800 |

Las 800 imágenes externas y sus lecturas se comprobaron localmente y quedaron reservadas. Sus etiquetas binarias no equivalen a las tres clases internas, por lo que no se añadieron a las particiones de TBX11K.

3 Distribución de las imágenes

Después de controlar las copias exactas, quedaron 8.274 imágenes únicas para el trabajo supervisado: 3.800 health, 3.674 sick y 800 TB. Se repartieron en entrenamiento, validación y prueba, con una proporción aproximada de 70/15/15. El gráfico siguiente es la salida real que generó el notebook con esas particiones.

Figura 3. Distribución real por clase y partición, exportada por el notebook. Azul corresponde a entrenamiento, naranja a validación y verde a prueba. En las tres particiones hay menos imágenes de TB que de las otras clases. La tabla de la siguiente sección muestra los mismos conteos junto con sus controles de integridad.

Entrenamiento servirá para ajustar los modelos; validación, para comparar alternativas; y prueba, para la evaluación final. La clase TB representa alrededor del 9,7 % de las imágenes utilizadas. Por eso se calcularon pesos de clase solo con entrenamiento, sin usar validación o prueba para ajustar ese cálculo.

4 Control de duplicados e integridad

Las listas originales tenían imágenes repetidas entre particiones. Para reducir esa fuga, se mantuvieron los archivos originales y se excluyeron de las listas de uso 126 copias exactas redundantes. Las imágenes parecidas ya identificadas se agruparon para que quedaran en una misma partición. Así se conservaron 8.274 imágenes únicas distribuidas en 8.251 grupos.

Figura 4. Salida real del control de particiones. La tabla confirma 5.792 imágenes para entrenamiento, 1.241 para validación y 1.241 para prueba. Los ceros muestran que las particiones no comparten rutas, archivos idénticos, contenidos RGB idénticos ni grupos de duplicados conocidos. También se completó la comparación de las 12.278 imágenes con el inventario.

Los siete archivos CSV exportados desde Colab coinciden con las listas locales que definen las particiones. Esto permite repetir el proceso con la misma distribución. El control evita que los duplicados conocidos aparezcan en conjuntos diferentes, aunque no garantiza independencia por paciente: faltan identificadores fiables y la búsqueda de imágenes parecidas no fue exhaustiva.

5 Muestras y preprocesamiento

Las imágenes TB conservan 1.211 cajas COCO que señalan las zonas anotadas. Se usaron las cajas que ya corresponden al tamaño de 512 × 512, para evitar aplicar las coordenadas de los XML sobre una resolución distinta. El caso tb1199 sigue etiquetado como TB, pero está marcado porque no tiene cajas; no debe tratarse como un ejemplo negativo del detector.

Figura 5. Muestras reales del conjunto de entrenamiento. De izquierda a derecha: una imagen saludable, una de enfermedad sin TB y una etiquetada como TB. En la tercera, las cajas rojas marcan lesiones activas y las cian marcan lesiones antiguas, según las anotaciones del dataset. Estas cajas son referencias originales para preparar la detección.

Las imágenes se preparan en RGB, con tres canales y tamaño de 512 × 512. La normalización ajusta la escala de los píxeles con media [0,485; 0,456; 0,406] y desviación [0,229; 0,224; 0,225]. Solo entrenamiento recibe cambios de brillo y contraste de ±8 % y rotaciones de ±5°, con relleno para conservar el campo. Validación y prueba no reciben aumentos. CLAHE quedó desactivado.

Figura 6. Salida real de los aumentos de entrenamiento. La primera columna muestra la imagen de referencia y las tres siguientes muestran variaciones pequeñas de brillo, contraste y rotación. La idea es dar más variedad al entrenamiento sin cambiar la clase de la imagen. Estos cambios no se aplican a validación ni a prueba.

El lector actual está preparado para clasificación. Para detección ya se conservan las cajas, sus categorías y la partición de cada imagen. Cuando se prepare el detector, habrá que adaptar sus transformaciones para mover las cajas junto con la imagen.

6 Comprobaciones y cierre de las fases

| Evidencia | Resultado |
| --- | --- |
| Pruebas locales del pipeline | 9 aprobadas; sin fallos |
| Ejecuciones en Colab | 9 celdas de entorno y 20 de datos sin errores |
| Lote de datos en Colab CPU | 4 × 3 × 512 × 512 |
| Integridad en ambas ejecuciones | 12.278 imágenes comparadas en cada una |
| Fuentes NLM completas | 800 imágenes y 800 lecturas verificadas |

La fase 1 dejó listo el entorno con Colab, Drive, repositorio y una prueba básica de GPU/CUDA. La fase 2 dejó los datos organizados, las particiones reproducibles y el preprocesamiento comprobado. Las capturas muestran los resultados de esas comprobaciones y permiten revisar el avance desde el informe.

Montgomery y Shenzhen están descargados y reservados, pero todavía no deben presentarse como una prueba externa independiente. TBX11K incluye versiones de fuentes externas y hay candidatos de solapamiento que deben revisarse antes de evaluar modelos. El test oficial, al no tener etiquetas públicas, tampoco permite calcular métricas supervisadas con la información disponible.

Colab comprobó los hashes de las listas externas y guardó el resumen de su preparación local. Las 800 imágenes externas originales se revisaron localmente, no en esa sesión de Colab. El entrenamiento de modelos y la integración con la web quedan para las siguientes etapas.

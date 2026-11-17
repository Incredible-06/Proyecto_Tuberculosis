INGENIERÍA DE SOFTWARE II


INFORME DE AVANCE DE LAS FASES 1 Y 2


Desarrollo de un Sistema Web Basado en Deep Learning para el Apoyo a la Detección de Tuberculosis Pulmonar en Radiografías de Tórax


Preparación del entorno de trabajo y preparación del dataset


Alumnos


Villar Bazan, Ronney Junior


Calvo Andrade, Sebastián Jefferson


Rodríguez Polo, Andy Javier


Rodas Aquino, Cesar Eduardo


Grupo A


Docente


Dr. Ing. Alfredo Daza Vergaray


20 de octubre de 2026



---


## Introducción y objetivos


El proyecto propone un prototipo académico de sistema web para apoyar el análisis de radiografías de tórax mediante Deep Learning. Este avance desarrolla la base técnica y los datos necesarios para los experimentos posteriores. Su alcance comprende la preparación del entorno de trabajo y la preparación de TBX11K; no incluye entrenamiento de clasificadores, fusión de arquitecturas, detección de lesiones o despliegue web.


Se organizaron el proyecto, la configuración y los manifiestos reproducibles. Las ejecuciones entregadas de Google Colab comprobaron lectura, transformaciones y un lote PyTorch en CPU, además de una operación CUDA en una GPU A100 y la coincidencia SHA-256 de las 12.278 imágenes transferidas. Con esta evidencia se cierran las verificaciones técnicas previstas para estas dos fases, dentro de las limitaciones documentadas.


### Objetivo general


Preparar y documentar un entorno reproducible y un conjunto de datos trazable que permitan desarrollar posteriormente los modelos del proyecto sin mezclar los conjuntos de entrenamiento, validación y prueba por duplicados conocidos.


### Objetivos específicos


Organizar el código, dependencias, rutas y mecanismos de versionado para ejecutar el trabajo en Colab.


Caracterizar la copia disponible de TBX11K y conservar sus anotaciones y procedencia.


Construir particiones estratificadas y un lector con transformaciones moderadas y verificaciones reproducibles.


Documentar resultados observados y separar las comprobaciones pendientes de los logros verificados.


La numeración usada en este informe corresponde al plan del curso: Fase 1, entorno; Fase 2, dataset. Los nombres históricos phase1 de algunas carpetas identifican el primer desarrollo técnico de datos, sin cambiar esa correspondencia.



---


## Fase 1 Preparación del entorno de trabajo


Se separaron notebooks, módulos reutilizables, configuración, manifiestos, resultados y documentación. Los directorios api, web y models reservan componentes futuros y explican su estado. Los originales de TBX11K se leen sin editarlos. Las reglas de Git excluyen el dataset, sus ZIP, pesos, resultados ejecutados y figuras generadas.


| Componente | Resultado comprobado |
| --- | --- |

| Python local | 3.14.7 |

| Python en Colab notebook 00 | 3.13.15 |

| NumPy y Pillow en Colab | 2.1.3 y 11.3.0 |

| PyTorch en Colab notebook 00 | 2.11.0+cu130 |

| GPU y CUDA | A100-SXM4-40GB; CUDA build 13.0; prueba aprobada |

| Persistencia | Drive montado y resultados exportados |



El notebook 01 permite montar Drive, extraer o copiar el dataset a almacenamiento temporal y registrar versiones y resultados. El almacenamiento temporal se reconstruye después de una desconexión; Drive conserva los archivos persistentes [2]. Los requisitos ligeros están separados de PyTorch/CUDA para evitar reinstalaciones innecesarias.


El notebook 00 verifica seeds, rutas de experimentos y checkpoints, dispositivo y una multiplicación CUDA de 4 × 4. Las siete celdas entregadas desde Colab se ejecutaron sin errores y conservan el código original. La GPU registrada fue NVIDIA A100-SXM4-40GB y cuda_probe.status fue passed. Esta comprobación acredita una operación básica en ese entorno; la compatibilidad de cada modelo y sus extensiones se verificará después en Colab.


El proyecto fue publicado en https://github.com/Incredible-06/Proyecto_Tuberculosis. El commit de código de referencia es 57e12472f4eb. Se comprobó que los archivos preparados para Git conservan sus bytes exactos; esta regla evita que cambios automáticos de fin de línea invaliden los hashes del protocolo.



---


## Fase 2 Caracterización y organización del dataset


TBX11K fue presentado por Liu et al. como un conjunto de 11.200 radiografías con anotaciones de regiones tuberculosas [1]. La descarga disponible incorpora también fuentes adicionales. Por ello, la cantidad local no debe interpretarse como pacientes únicos ni como imágenes exclusivamente del conjunto base.


| Contenido local | Cantidad |
| --- | --- |

| Health | 3.800 |

| Sick sin TB | 3.800 |

| TB etiquetado | 800 |

| Extra reservado | 576 |

| Test oficial sin etiquetas | 3.302 |

| Total bajo imgs | 12.278 |



Las 12.278 imágenes son de 512 × 512 píxeles y están almacenadas en RGB: 12.174 PNG y 104 JPEG. La auditoría inicial no detectó errores de lectura o decodificación. Su inventario y hashes se reutilizaron en esta continuación; no se repitió el análisis completo.


Se adoptaron tres clases de imagen: health, sick y TB. Los subtipos pertenecen a las lesiones y pueden coexistir: hay 630 imágenes con solo lesiones activas, 139 con solo lesiones antiguas, 30 mixtas y una imagen TB sin objetos anotados. El clasificador inicial agrupa los subtipos de TB; esta decisión ajusta el alcance experimental respecto de las cuatro categorías descriptivas de la planificación.


Existen 800 XML y diez JSON COCO. Las 1.211 cajas COCO se conservaron con sus campos e identificadores y se comprobaron contra su fuente. Los XML usan dimensiones originales variables; las coordenadas COCO ya corresponden a 512². No se realizó una conversión nueva. tb1199 conserva clase TB y una bandera de cajas ausentes; no constituye un negativo para un futuro detector.



---


## Control de duplicados y construcción de particiones


La auditoría encontró 216 pares exactos en todo el paquete. Las listas oficiales no comparten rutas, pero incluyen 38 pares exactos entre entrenamiento y validación, 64 entre entrenamiento y test disponible y nueve entre validación y test. Esto motivó un protocolo propio, preservando las listas originales como metadatos.


Dentro de las 8.400 imágenes base etiquetadas se identificaron 8.274 contenidos RGB exactos únicos. Se eligió un representante determinista por contenido y se conservaron los aliases en el índice de miembros. Las 126 copias redundantes se excluyeron de los manifiestos de uso, sin eliminar ningún archivo original.


Se refinaron los 13.783 pares candidatos de dHash igual mediante pHash de 256 bits, SSIM local, correlación y error de píxeles. Quedaron 44 pares, revisados visualmente por Codex junto con ocho controles rechazados. Se confirmaron como copias cercanas para el control de fuga, sin identificar pacientes. Veintitrés pares pertenecen al conjunto etiquetado. Los contenidos cercanos se mantienen juntos y producen 8.251 grupos indivisibles.


| Partición | Health | Sick | TB | Total |
| --- | --- | --- | --- | --- |

| Train | 2.660 | 2.572 | 560 | 5.792 |

| Validación | 570 | 551 | 120 | 1.241 |

| Test interno | 570 | 551 | 120 | 1.241 |

| Total | 3.800 | 3.674 | 800 | 8.274 |



Se fijó seed 20261019 y una proporción aproximada 70/15/15 por clase, con redondeo por restos mayores y asignación de grupos indivisibles. Los manifiestos guardan rutas relativas, etiquetas, hashes, grupos, listas oficiales, aliases y anotaciones. El test interno se reserva para evaluación final; no se utiliza para seleccionar transformaciones o modelos.



---


## Preprocesamiento y comprobaciones de funcionamiento


El lector entrega matrices float32 de forma 3 × 512 × 512 y etiquetas enteras. Convierte a RGB y conserva el campo mediante padding si cambia la proporción. Usa normalización configurable sobre RGB en [0,1], con media [0,485; 0,456; 0,406] y desviación [0,229; 0,224; 0,225], compatibles con la normalización documentada de ConvNeXt-Tiny [3]. Los requisitos de los pesos futuros deberán verificarse por arquitectura.


Solo train aplica variaciones de brillo y contraste de ±8 % y rotaciones de ±5°, con expansión antes de ajustar al lienzo. No se usan espejado, recortes agresivos o borrado de regiones. Validación y test son deterministas. CLAHE permanece desactivado para una posible ablación posterior. Los pesos de clase se calculan únicamente desde train.


| Comprobación | Resultado | Ámbito |
| --- | --- | --- |

| Pruebas automáticas ligeras | 9 aprobadas | Local |

| Celdas del notebook de datos | 14 sin errores | Colab CPU |

| Celdas del notebook de entorno | 7 sin errores | Colab A100 |

| Cruces internos conocidos | 0 por ruta, hash o grupo | Local y Colab |

| Lote de cuatro imágenes | 4 × 3 × 512 × 512 | Colab CPU |

| GPU y multiplicación CUDA 4 × 4 | passed | Colab A100 |

| Comparación completa SHA-256 | 12.278 coincidentes | Colab |



Los siete CSV entregados desde Colab coinciden byte a byte con los manifiestos locales congelados. Se conservaron la configuración y las decisiones de revisión. Las figuras se inspeccionaron técnicamente; esto no equivale a una validación clínica de los aumentos ni del sistema.



---


## Dificultades soluciones y limitaciones


Las clases de imagen se confundían con subtipos de lesión en la planificación. La inspección permitió separar tres clases supervisadas y conservar los subtipos como metadatos, incluyendo los casos mixtos. Las cajas XML y las imágenes distribuidas tenían dimensiones distintas; se reutilizó el COCO verificado para evitar un escalado incorrecto.


Los duplicados entre listas oficiales podían introducir fuga de datos. Se construyeron particiones propias con representantes exactos y grupos de copias cercanas confirmadas. La búsqueda cercana solo refinó candidatos de dHash igual y no fue exhaustiva. No existen identificadores fiables de paciente, por lo que no se garantiza independencia por paciente o estudio.


El test oficial disponible carece de etiquetas públicas. Sus 3.302 imágenes se conservaron separadas, sin inferir clases por nombres o duplicados. Se documentaron 93 solapamientos de contenido conocidos con datos internos. No se calcularon métricas supervisadas sobre ese test.


Se reservaron las 576 imágenes de extra, incluidas 400 de la carpeta combinada Montgomery/Shenzhen y 176 DA/DB. Una evaluación externa futura requiere etiquetas y procedencia verificadas y un control ampliado de solapamientos. Esta reserva no reconstruye las fuentes externas completas.


### Conclusiones y siguientes pasos


El avance dispone de una estructura versionada y un pipeline de datos reproducible. La lectura y los aumentos funcionaron en Colab CPU; el notebook 00 comprobó una operación CUDA en A100 y los SHA-256 de todos los archivos transferidos. Se cierran las comprobaciones técnicas del entorno y de los datos para este avance. Las precauciones de licencia y la ausencia de identificadores de paciente siguen documentadas.


La etapa posterior será verificar implementaciones y pesos de ConvNeXt-Tiny y Vision Mamba y diseñar su comparación. La prueba CUDA básica no valida todavía esos modelos. El test interno permanecerá reservado. Los entrenamientos y comprobaciones de modelos completos se ejecutarán exclusivamente en Colab. Este informe no presenta métricas de clasificación, una web funcional o un detector entrenado.



---


## Anexos de evidencia visual y archivos


Figura 1. Muestras de train de la ejecución Colab. De izquierda a derecha: health h0001, sick s0001 y TB tb0135. Las cajas del caso TB proceden del COCO original.


La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.


Figura 2. Ejemplo health h0001 de la ejecución Colab. Izquierda: transformación determinista. Derecha: una variante de train con los aumentos moderados configurados. Se muestra antes de normalizar.


La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.


Evidencias principales: results/01_exploracion_preprocesamiento.ipynb y el ZIP de resultados de Colab; reports/phase1/colab_review/verification.json; reports/phase1/tests.json; reports/environment/notebook_validation.json; manifests/manifest_metadata.json. Los nombres y hashes permiten rastrear los resultados sin publicar las imágenes originales en Git.



---


## Distribución de las clases y control visual de duplicados


Figura 3. Cantidades por clase y partición, elaboradas a partir de los manifiestos congelados. La escala de las barras es común. TB conserva una frecuencia menor, que deberá considerarse al evaluar los clasificadores.


La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.


Figura 4. Par near_00004: sick/s1174.png y sick/s1964.png, confirmado en la revisión previa. Sus contenidos RGB difieren; se conserva cada variante y ambas pertenecen al mismo grupo y partición. La semejanza no demuestra identidad de paciente.


La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.


La figura 3 es un gráfico elaborado con los datos exportados; la figura 4 presenta imágenes originales ya revisadas. Las figuras 1 y 2 proceden de las salidas del notebook 01. Estas procedencias se distinguen para no presentar gráficos nuevos como capturas de una ejecución.



---


## Evidencia del repositorio y del entorno Colab


Figura 5. Captura aportada del repositorio privado Proyecto_Tuberculosis, rama main y estructura versionada. Corresponde al commit c62c15c visible en la captura; no representa necesariamente el último estado. Se recortó el navegador y la barra lateral, conservando el contenido de la interfaz.


La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.


La ejecución del entorno quedó registrada como entorno_20261020T051106311884Z, el 20 de octubre de 2026 a las 00:11 en America/Lima. El repositorio privado requiere acceso concedido al docente si debe revisarlo directamente. La captura conserva las fuentes y colores propios de GitHub; el texto y los pies del informe usan Times New Roman 12 negro.



---


## Resultados registrados de integridad y CUDA


La siguiente tabla transcribe campos de environment_setup.json y transfer_integrity.json entregados. Es una síntesis editable de resultados reales, no una captura simulada de Colab.


| Campo registrado | Valor |
| --- | --- |

| GPU | NVIDIA A100-SXM4-40GB |

| PyTorch y CUDA build | 2.11.0+cu130 / 13.0 |

| cuda_probe.status | passed |

| Operación | Multiplicación float32 de 4 × 4, sin modelo |

| training_executed | false |

| transfer_integrity.status | passed |

| all_image_hashes_verified | 12278 |

| Imágenes supervisadas / grupos | 8.274 / 8.251 |



| Criterio entre particiones | Train / val. | Train / test | Val. / test |
| --- | --- | --- | --- |

| Ruta relativa | 0 | 0 | 0 |

| SHA-256 del archivo | 0 | 0 | 0 |

| SHA-256 del contenido RGB | 0 | 0 | 0 |

| Grupo de duplicados | 0 | 0 | 0 |



Los doce cruces internos registrados son cero. La configuración y los metadatos entregados coinciden con los locales, y los hashes del protocolo y de los siete CSV coinciden con la versión congelada. El notebook conserva sus siete celdas originales y no presenta salidas de error. La revisión local leyó estos archivos sin ejecutar CUDA ni entrenar.


Esta evidencia acredita la copia y las comprobaciones implementadas; no demuestra independencia por paciente ni compatibilidad de futuras arquitecturas. Los algoritmos deterministas se habilitaron con avisos y PYTHONHASHSEED no estaba fijado al iniciar el intérprete. Las particiones congeladas son reproducibles; la reproducibilidad de entrenamientos se revisará en la etapa de modelos.



---


## Referencias y trazabilidad del proyecto


[1] Liu, Y., Wu, Y. H., Ban, Y., Wang, H. y Cheng, M. M. (2020). Rethinking Computer-Aided Tuberculosis Diagnosis. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2646–2655. https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html


[2] Google. Google Colab Frequently Asked Questions. https://research.google.com/colaboratory/faq.html


[3] PyTorch. Torchvision convnext_tiny. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html


Las cantidades, verificaciones y estados de este informe proceden de los artefactos generados y entregados por el proyecto. Se revisó la ejecución Colab identificada como phase1_20261020T042524626304Z, iniciada el 19 de octubre de 2026 a las 23:25 en America/Lima. La evidencia es el notebook ejecutado y sus salidas; no se observó una sesión en vivo.


| Directorio | Función |
| --- | --- |

| notebooks | Configuración del entorno y preparación de datos |

| src/tbxdata | Lectura, preparación, validación y figuras |

| configs y manifests | Protocolo y particiones congeladas |

| reports y results | Evidencias y salidas de ejecución |

| scripts y tests | Reproducción y comprobaciones ligeras |

| api web models | Componentes futuros documentados |



También se revisó entorno_20261020T051106311884Z: siete celdas de código sin modificaciones, CUDA passed y comparación completa de 12.278 archivos aprobada. Los JSON y la revisión se conservan en reports/environment/colab_review. Se contrastaron archivos entregados; no se observó una sesión en vivo. No se requieren nuevos entrenamientos para sustentar estas dos fases.


La revisión de fuentes y la discrepancia entre los términos publicados y los metadatos locales de licencia se documentan en docs/FUENTES_Y_LICENCIAS.md. Los originales no se modificaron.


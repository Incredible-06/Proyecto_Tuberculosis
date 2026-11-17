# Fuentes y condiciones declaradas

## TBX11K

Fuente primaria de procedencia y términos: https://mmcheng.net/tb/

La sección Terms of Use de la página oficial consultada declara Creative Commons
Attribution 4.0. La página también explica las imágenes mixtas activas/latentes,
la publicación a 512 × 512 y la ausencia de anotaciones públicas del test. El
artículo de referencia está disponible en:
https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html

Los JSON COCO de la copia local contienen una declaración distinta:
Attribution-NonCommercial-ShareAlike, con enlace a
http://creativecommons.org/licenses/by-nc-sa/2.0/ . Se conserva sin cambiarla.
Esta revisión registra la discrepancia; no concluye que ambas declaraciones sean
intercambiables ni extiende permisos de TBX11K a las otras fuentes del paquete.
Antes de redistribuir imágenes o de un uso diferente al avance académico, debe
confirmarse con los responsables del conjunto qué términos rigen la copia usada.

El repositorio del equipo contiene código, manifiestos y metadatos autorizados
por el usuario; no contiene imágenes originales, sus archivos ZIP o pesos.
Las figuras del informe local derivan de muestras de train y conservan la
referencia al conjunto y a las salidas de Colab. Los metadatos originales de
licencia se preservan para trazabilidad.

## Fuentes adicionales

### Incorporación de las fuentes oficiales completas el 21 de octubre de 2026

Se consultaron la página oficial de recursos NLM y los documentos de origen:

- https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html
- https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/
- https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/

Los documentos `NLM-MontgomeryCXRSet-ReadMe.pdf` y `NLM-ChinaCXRSet-ReadMe.docx`
se conservan con los originales en `external_data`. Solicitan atribución y citas,
y que la distribución del dataset se limite al grupo u organización de investigación.
Se registran estas condiciones, sin asignarles una licencia Creative Commons distinta.
El código y el paquete pequeño no redistribuyen esas imágenes ni sus lecturas.

Los documentos explican la convención 0/1 de los nombres: normal y anormal con
manifestaciones compatibles con TB. Esa documentación sustenta las etiquetas binarias
de los nuevos manifiestos; no se infiere la etiqueta de imágenes anónimas del test TBX11K.
La adquisición comprende las 138 y 662 radiografías originales y sus lecturas asociadas.
No se descargan las máscaras o anotaciones auxiliares de segmentación, fuera de estas
dos fases de preparación para clasificación y detección TBX11K.

Estas fuentes quedan en reserva externa. La búsqueda de candidatos de solapamiento
se registra en `reports/external_sources/overlap_candidates.csv`; no es una revisión
exhaustiva ni demuestra independencia. La verificación de sus manifiestos no cambia
el protocolo congelado TBX11K.

### Reserva histórica incluida dentro del paquete TBX11K

La copia local combina 400 imágenes de Montgomery/Shenzhen en una misma carpeta
de extra y 176 imágenes DA/DB. Se mantienen reservadas. Las licencias individuales,
etiquetas de imagen y procedencia deben verificarse con sus fuentes originales
antes de incorporarlas a una evaluación externa o redistribución. No se han
inferido esas etiquetas desde los nombres de archivo.

## Referencias del entorno

- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Normalización de ConvNeXt-Tiny:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html

La revisión de estas fuentes no sustituye los resultados de ejecución: las
versiones y estados CPU/GPU del informe proceden de environment.json y del
notebook entregado, no de las posibilidades generales de Colab.

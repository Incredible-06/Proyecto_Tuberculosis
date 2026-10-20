# Qué falta para cerrar el informe de las dos fases

La numeración es la del plan del curso: Fase 1 entorno y Fase 2 dataset. El
entrenamiento es posterior. La ejecución de datos en CPU ya fue entregada y
verificada; no se requieren nuevos experimentos de modelos para este informe.

## Evidencia técnica pendiente

1. Subir el nuevo ZIP `tbx11k_fases_1_y_2_colab.zip` a
   `Mi unidad/Proyecto_Tuberculosis/`, junto al dataset y el ZIP anterior.
2. Abrir `00_configuracion_entorno.ipynb` en Colab. Seleccionar GPU en la
   configuración del entorno si hay una disponible; no se necesita una A100.
3. Ejecutar las celdas. Se hará una operación 4×4 en CUDA, sin modelos ni
   entrenamiento, y se compararán todos los hashes de la transferencia.
4. Conservar el notebook ejecutado y la carpeta `results/entorno_<fechaUTC>/`
   creada en Drive. Debe contener `environment_setup.json`,
   `transfer_integrity.json`, configuración y metadatos de manifiestos.
5. Para marcar GPU como comprobada, `cuda_probe.status` debe ser `passed`.
   Si no se asigna GPU, el informe debe mantener ese punto pendiente.
6. Para cerrar integridad completa, `all_image_hashes_verified` debe ser 12278
   y `status` debe ser `passed`. No es necesario rehacer duplicados o splits.

La evidencia existente del notebook 01 muestra GPU no disponible y hashes de
27 muestras, además de tamaños y existencia de todos los archivos. Esto permite
informar validación de datos en CPU, pero no afirmar CUDA o copia íntegra por hash.

## Información académica

Se dispone del título completo, cuatro integrantes, curso, docente y grupo A.
Solo faltan universidad y carrera si la portada debe incluirlas, fecha oficial de
entrega y cualquier rúbrica o plantilla institucional. Se ha adoptado Times New
Roman 12, texto negro, A4, márgenes de 2,5 cm e interlineado 1,5 para el informe.
Una norma de citas o extensión exigida por el profesor puede requerir ajustar el
formato; no se ha supuesto una norma institucional inexistente.

## Coherencia con el documento de planificación

Debe sustituirse la mención de cuatro categorías por tres clases de imagen:
health, sick y TB. Los subtipos se conservan como metadatos de lesión y hay casos
mixtos; no equivalen a clases mutuamente excluyentes. Conviene adjuntar el
documento de planificación completo para corregir también las secciones que no
aparecen en la captura. La captura no permite revisar el resto del documento.

La distribución de responsabilidades de la captura es planificada. Para atribuir
actividades realizadas a cada integrante se necesitan sus aportes reales; no se
ha supuesto que una persona ejecutó cada tarea por figurar como responsable.

No se necesitan métricas de clasificación, checkpoints, entrenamiento de Mamba,
fusión o detector para presentar estas dos fases. Tampoco se necesitan de nuevo
el dataset o el ZIP de resultados del notebook 01, ya disponibles.

## Entregables y publicación

El repositorio ya se publicó en
https://github.com/Incredible-06/Proyecto_Tuberculosis con código, notebooks y
manifiestos autorizados. El informe editable está en
`docs/Informe_Fases_1_y_2_Final.docx`; el texto versionable, en
`docs/Informe_Fases_1_y_2.md`.

Se documentó una discrepancia de licencias entre la página oficial y los JSON de
la copia local en `FUENTES_Y_LICENCIAS.md`. Antes de redistribuir imágenes o darles
un uso distinto, se requiere aclarar los términos aplicables con sus responsables.

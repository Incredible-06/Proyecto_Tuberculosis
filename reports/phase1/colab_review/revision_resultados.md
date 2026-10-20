# Revisión de la ejecución entregada de Colab

Ejecución: `phase1_20261020T042524626304Z` (inicio: 19 de octubre de 2026,
23:25, America/Lima). Se revisaron el ZIP y el notebook ejecutado de `results/`
sin ejecutar sus celdas ni volver a analizar el dataset.

## Resultado

Las comprobaciones ejecutadas de la Fase 1 pasaron en Colab. El notebook tiene
14 celdas de código ejecutadas, sin salidas de error. Su código coincide con la
versión preparada localmente. Las salidas evidencian montaje de Drive, uso del
dataset temporal y exportación de resultados a Drive. No se observó una sesión
en vivo; la evidencia procede de los archivos entregados.

Se verificó de forma independiente:

- ZIP íntegro y siete manifiestos CSV idénticos byte a byte a la versión local congelada.
- Coincidencia de hashes registrados, configuración y decisiones de duplicados.
- Conteos: 5.792 train, 1.241 validation y 1.241 test interno.
- Ausencia de cruces internos por los grupos de duplicados conocidos.
- Resultado de integridad `passed`, con cero cruces por rutas, hashes o grupos.
- Lectura y transformaciones comprobadas en seis muestras.
- Un lote PyTorch CPU de forma `(4, 3, 512, 512)` ejecutado correctamente.
- Figuras de muestras y aumentos revisadas visualmente: no se identificaron problemas de presentación o aplicación de las transformaciones previstas.

Entorno registrado: Python 3.13.16, NumPy 2.1.3, Pillow 11.3.0 y
PyTorch 2.11.0+cpu. `cuda_available=False`, apropiado para esta fase de datos.
No hay evidencia de entrenamiento ni de compatibilidad CUDA/modelos.

## Comprobación pendiente

`VERIFY_ALL_IMAGE_HASHES` permaneció desactivado. Se comprobaron SHA-256 de
**27 muestras**, además de existencia/tamaño de todos los archivos. El registro
`all_image_hashes_verified=0` significa que no se ejecutó el modo de comparación
completa; no invalida las comprobaciones muestrales realizadas.

Antes del entrenamiento conviene comparar todos los archivos transferidos contra
los hashes de la auditoría. Puede hacerse en Colab con la opción existente
`VERIFY_ALL_IMAGE_HASHES=True`, o como comprobación previa de la Fase 2. No exige
rehacer duplicados ni particiones. No se ha ejecutado esa lectura completa en la PC.

Para la Fase 2 también faltan un entorno Colab con GPU y las verificaciones de
dependencias, modelos, pesos y CUDA correspondientes. Los manifiestos y la
configuración entregados bastan como punto de partida para preparar esa fase.

Persisten las limitaciones metodológicas documentadas: búsqueda de duplicados
cercanos no exhaustiva e independencia por paciente no garantizada.

`verification.json` conserva hashes de los archivos recibidos y el detalle de
esta revisión. Las figuras de esta carpeta son copias de las salidas entregadas;
los archivos originales de `results/` no se alteraron.

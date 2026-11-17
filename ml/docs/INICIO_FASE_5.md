# Inicio de Fase 5: nuevo Colab y validation

Esta entrega prepara el notebook independiente
`notebooks/06_fase5_seleccion_explicabilidad_evaluacion.ipynb`.
No afirma una ejecución GPU ni la culminación de Fase 5.

## Ejecutar en Colab

1. Colocar `dist/tbx11k_fase5_preparacion_colab.zip` en
   `MyDrive/Proyecto_Tuberculosis/`, conservando los ZIP anteriores.
2. Abrir el notebook **06** nuevo con runtime GPU. Mantener los notebooks
   ejecutados 02/03/04/05 y sus outputs. No ejecutar sus plantillas encima.
3. Ejecutar pasos 1–7 en orden: montaje, entorno, verificación de tres corridas,
   transferencia, kernel oflex, smoke separado y curvas del clasificador.
4. Revisar las salidas. El paso 8 permite activar
   `RUN_DETECTOR_VALIDATION=True` para obtener diagnóstico nuevo del detector.
   El paso 9 compara alertas usando umbrales de ejemplo, sin congelarlos.
5. Ejecutar paso 10 y guardar el notebook con Ctrl+S. Compartir las salidas de
   `results/phase5/<run_id>/` para continuar con errores, explicabilidad y selección.

Cada ejecución del paso 1 crea rutas nuevas en `/content/tbx11k_phase5/` y
Drive `results/phase5/`. La transferencia del dataset usa un caché separado.
El notebook verifica el SHA del ZIP completo y todos sus miembros antes de
extraer; notebook y ZIP deben pertenecer a la misma entrega.

## Gráficos para el informe

El paso 3.1 exporta 12 figuras desde los artefactos existentes, sin reentrenar:

- Comparación de accuracy, Macro-F1, sensibilidad, precisión y especificidad TB.
- Tres figuras de entrenamiento, una por clasificador: pérdidas train/validation,
  accuracy train/validation y Macro-F1/sensibilidad TB por época, señalando la mejor.
- Tres matrices de confusión: cada figura incluye conteos y proporciones por clase
  real; orden health/sick/TB, filas reales y columnas predichas por argmax.
- Tres curvas ROC OVR, una por clasificador, con AUC por clase.
- Comparación ROC y precisión–recall de TB para las tres arquitecturas.
- Historial YOLO11n: pérdidas, mAP, precisión/recall por caja y mejor época,
  distinguiendo la validación final de best.pt del CSV de entrenamiento.

Cada figura tiene PNG a 300 dpi, PDF y SVG vectoriales, CSV de los datos, más un
índice README y un manifiesto de hashes/fuentes. El paso puede ejecutarse antes
de copiar el dataset o compilar oflex. No se escriben figuras en los experimentos
anteriores. La carpeta nueva queda en `results/phase5/<run_id>/figures_<id>/`.

Además, el paso 7 exporta la curva de umbrales del clasificador en los tres
formatos; el paso 8, si se activa, exporta curvas de sensibilidad y FP por imagen
del detector. Se conservan sus tablas CSV. Las figuras corresponden a validation,
sin intervalos de confianza todavía y sin evaluación de test.

## Controles y límites

- No hay inferencia de test ni mecanismo para activarla en esta entrega.
  El test interno, el oficial sin etiquetas y los externos siguen reservados.
  Leer hashes/manifiestos para integridad no constituye evaluación o selección.
- El módulo nuevo `src/tbxtrain/phase5.py` verifica todas las fuentes originales
  registradas por los checkpoints. No se modifica el código del entrenamiento.
- Best.pt de los tres clasificadores se compara con SHA histórico, resumen y
  contrato interno; carga estricta con `pretrained=False`, mejor época y eval.
- Del detector se registra el SHA observado ahora: no existe un SHA histórico
  disponible en la evidencia consultada para contrastarlo. Se conserva esa limitación.
- Pip usa constraints con versiones instaladas de torch/torchvision/torchaudio.
  El instalador existente de oflex incorpora la reparación CUDA 13 y conserva
  esos paquetes. Sus logs de esta ejecución se guardan en la carpeta nueva.
- La comparación usa argmax original. Las tablas de `p_TB >= t` describen alertas
  adicionales, no reemplazan silenciosamente la clase multiclase.
- El diagnóstico del detector diferencia sensibilidad por imagen con cualquier
  caja, sensibilidad con una localización correcta, recall por lesión y FP/caja
  por imagen. Matching IoU>=0,5; no se presenta como mAP de Ultralytics.
- Concordancia propuesta: alerta de cualquiera de los dos componentes requiere
  revisión. Clasificador TB sin cajas nunca pierde su alerta. Se conservan
  probabilidades y cajas originales, sin multiplicación de scores.
- Umbrales 0,5 y 0,25 son ejemplos exploratorios. La selección, los umbrales y
  la concordancia final permanecen sin definir; no se inventa un objetivo clínico.

## Continuidad

Tras revisar validation: seleccionar ejemplos con regla explícita; inspeccionar
cajas/errores; medir coste con protocolo comparable; remuestrear por grupos de
duplicados (sin asumir pacientes); implementar y verificar Grad-CAM donde proceda;
decidir si mejorar el detector antes de congelar. Luego crear un contrato de
inferencia inmutable con hashes de pesos, datos/código, preprocesamiento, umbrales,
NMS y concordancia, más la evidencia de validation. Solo después añadir evaluación
reservada, informe final y exportación. Si se modifica una decisión después de
abrir test, ese test ya no puede volver a presentarse como reservado.

## Validación local

Usar el entorno CPU temporal existente, sin atribuirle la ejecución GPU de Colab:

```powershell
& tmp/phase3_test_env/Scripts/python.exe scripts/package_phase5_preparation.py
& tmp/phase3_test_env/Scripts/python.exe scripts/create_phase5_notebook.py
& tmp/phase3_test_env/Scripts/python.exe -m unittest discover -s tests -p test_phase5_preparation.py -v
```

El generador solo escribe el notebook 06 y rechaza sobrescribirlo si ya contiene
outputs o contadores de ejecución. El empaquetador no incluye notebooks anteriores.

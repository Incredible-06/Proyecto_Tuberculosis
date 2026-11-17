# Revisión de fases 3 y 4 y preparación de fase 5

Revisión: 2 de noviembre de 2026, hora de Lima. Los nombres de corridas usan UTC
y por ello comienzan con `20261103`. Esta revisión consulta resultados existentes;
no ejecuta entrenamientos ni abre el test reservado.

## Conclusión

Se puede iniciar la fase 5. La fase 3 tiene su cierre y los tres clasificadores
entrenados, comparados y versionados. La fase 4 tiene un detector entrenado,
validado y versionado; su rendimiento limita el papel que debe tener en la
solución. Completar ese entregable técnico no acredita una localización fiable.

La fase 5 debe producir **un notebook nuevo ejecutado y un informe de resultados**,
además de pesos y configuración de inferencia exportados. No basta con redactar
un informe ni con volver a mostrar las curvas de entrenamiento.

## Clasificadores: resultados verificados

Se leyeron los tres historiales completos y las 1.241 predicciones de validación
por modelo. Se recalcularon las matrices de confusión, accuracy y Macro-F1:
coinciden con los resúmenes. Los tres usan las mismas imágenes, etiquetas,
protocolo y hashes registrados de procedencia. También se comprobó que existen
los archivos `best.pt` y `last.pt` en Drive. No se descargaron otra vez los pesos
para recalcular su SHA-256: ese control consta en el cierre ejecutado en Colab.

| Modelo | Épocas / mejor | Macro-F1 | Accuracy | Sensibilidad TB | TB correctos / total | TB omitidos |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 12 / 11 | 0,9427 | 0,9710 | 0,7833 | 94 / 120 | 26 |
| VMamba-Tiny | 12 / 11 | 0,9497 | 0,9686 | 0,8417 | 101 / 120 | 19 |
| Híbrido con atención cruzada | 7 / 3 | 0,9645 | 0,9782 | 0,8917 | 107 / 120 | 13 |

El híbrido es el candidato principal con el criterio fijado de Macro-F1 de
validación. Detecta como TB 13 imágenes más que ConvNeXt en total y tiene cuatro
falsos positivos TB entre las 1.121 imágenes no TB. Los 13 TB omitidos se reparten
en siete predicciones `sick` y seis `health`. Su precisión TB es 0,9640 y su
especificidad TB 0,9964. No se modificó todavía el umbral: se usa argmax softmax.

Las siete épocas del híbrido son una terminación normal por early stopping:
la mejor fue la tercera, seguida por cuatro sin la mejora requerida. Las pérdidas
de validación posteriores aumentan mientras la de entrenamiento baja; conservar
la época 3 tiene sentido y no debe sustituirse por la última época 7.

La comparación usa una sola seed y una validación reutilizada para elegir época
y arquitectura. No demuestra superioridad estadística ni rendimiento final.
Hay 1.237 grupos de duplicados en las 1.241 imágenes; no hay garantía de
independencia por paciente. El conjunto de prueba sigue sin evaluar según los
registros consultados.

Fuentes: [cierre y comparación en Drive](https://drive.google.com/drive/folders/1fncC1MxRqjKcK83W2zI0noozNtfy9DfI),
[ConvNeXt](https://drive.google.com/drive/folders/1uhSW6P1pYxNiCtdQ9TY5BXVh7OycG916),
[VMamba](https://drive.google.com/drive/folders/12S0S0JriOrFMzYePXpmve0PNe4ywHzWT),
[híbrido](https://drive.google.com/drive/folders/1ah_kyFzhnRBh0rRub-krY-hk8MWKn5ks).

## Detector: ejecución completa con rendimiento limitado

YOLO11n entrenó 33 épocas de un máximo de 35. El historial registra su mejor
mAP50–95 en la época 26 (0,21553), con siete épocas posteriores sin superarlo,
coherente con la paciencia 7 configurada. La validación final del `best.pt` dio:

| Métrica de validación | Resultado |
|---|---:|
| mAP50 | 0,484232 |
| mAP50–95 | 0,215776 |
| Precisión por caja, redondeada de la salida de Colab | 0,609 |
| Recall por caja, redondeado de la salida de Colab | 0,451 |
| Imágenes / positivas / negativas | 1.241 / 120 / 1.121 |
| Lesiones anotadas | 176 |

La pequeña diferencia entre CSV de entrenamiento y validación final se conserva
como tal: para el resultado del modelo guardado se usa `detector_summary.json`.
No se sustituye por el máximo mAP50 de otra época: seleccionar con mAP50–95 puede
elegir un checkpoint distinto del que maximiza exclusivamente mAP50.

Hay aprendizaje: mAP50–95 sube de 0,01019 en la primera época a aproximadamente
0,216. Al final la pérdida de caja de train sigue bajando y la de validación
empeora ligeramente. Esto sugiere que añadir épocas sin otros ajustes no
garantiza mejorar. El menor mAP al exigir más solapamiento también señala límites
en la localización, sin identificar por sí solo su causa.

El recall 0,451 es por lesión/caja, no la sensibilidad para clasificar una imagen
o paciente con TB. Ultralytics calcula P/R de esta tabla en el punto elegido
por máximo F1 de su curva; todavía no es un umbral operativo acordado para el
producto. Tampoco mAP50=0,484 significa 48,4 % de accuracy de pacientes.
La definición se comprobó en el [código oficial de la versión 8.3.228](https://github.com/ultralytics/ultralytics/blob/v8.3.228/ultralytics/utils/metrics.py).

La conversión conservó 559 positivos y 5.232 negativos en train, con 851 cajas;
excluyó el único TB sin anotaciones de caja. Validation conserva los 120 TB con
176 cajas. No se exportó test. No se identificó una inconsistencia en esos
conteos; esto no reemplaza una revisión visual de anotaciones y predicciones.
Los PNG de curvas y matrices se localizaron en Drive, pero no se pudieron
descargar para inspección visual desde este entorno; el análisis de curvas se
basa en sus historiales numéricos completos.

`conversion.json` conserva `prepared_not_trained` y `phase4_complete=false`
porque documenta el momento de conversión. El cierre de fase 3 también mantiene
fase 4 pendiente como instantánea anterior. Eso no invalida el entrenamiento
posterior: `detector_summary.json`, el historial y los pesos son su evidencia.

Fuente: [corrida del detector](https://drive.google.com/drive/folders/1KjksA8z7NGaIqolZxOVlR1H1983RAqAe).

## Cómo avanzar a la fase 5

Crear un notebook independiente, con nombre previsto
`06_fase5_seleccion_explicabilidad_evaluacion.ipynb`, cargando los mejores pesos
de Drive y conservando los notebooks 04/05 ejecutados. Este documento define su
contenido; no afirma que ese notebook ya esté implementado ni ejecutado.

1. **Recuperar y verificar artefactos.** Configuración, particiones, hashes,
   pesos seleccionados, clase health/sick/TB y entorno compatible con VMamba.
   Volver a cargar los mejores modelos; no continuar entrenando por defecto.
2. **Seleccionar usando validation.** Confirmar el híbrido como candidato según
   el criterio fijado; documentar coste y errores, y mantener las otras dos
   arquitecturas como comparación. Examinar incertidumbre con remuestreo por
   grupo conocido, reconociendo que esos grupos no equivalen a pacientes.
3. **Evaluar puntos de operación.** Curvas y umbrales de alerta TB del
   clasificador; umbral de confianza del detector, sensibilidad por imagen,
   localización por lesión y falsos positivos por imagen. Definir el criterio
   de elección antes de abrir test; no inventar un objetivo clínico alcanzado.
4. **Definir concordancia.** Mantener probabilidades originales y registrar
   alertas por separado. Una alerta TB del clasificador sin cajas debe generar
   revisión, nunca suprimirse automáticamente. Una caja con score bajo de TB
   también es una discrepancia. No multiplicar ambos scores arbitrariamente ni
   presentar la ausencia de cajas como descarte de TB. Evaluar estas reglas en
   validation antes de congelarlas.
5. **Explicar y revisar errores.** Grad-CAM donde el gradiente y la arquitectura
   lo permitan, incluyendo las ramas del híbrido si corresponde; comprobar que
   los mapas respondan al modelo. Mostrar aciertos, falsos positivos y falsos
   negativos seleccionados con una regla explícita. Un mapa de atención no
   constituye por sí mismo una localización validada de lesión.
6. **Congelar la solución.** Guardar modelo, preprocesamiento, umbrales,
   concordancia, limitaciones y hashes en una configuración de inferencia.
   Si se decide mejorar el detector, registrar otra corrida de fase 4 y usar
   solo validation para compararla antes de este paso. El detector actual se
   conserva como baseline.
7. **Evaluación final reservada.** Ejecutar test después de congelar esas
   decisiones. Reportar accuracy, Macro-F1, sensibilidad/especificidad TB, AUC,
   matrices y métricas de detección, con intervalos y límites de dependencia.
   No volver a ajustar con test. Evaluar externos solo tras confirmar
   procedencia, ausencia de solapamientos y compatibilidad de etiquetas.
8. **Exportar y redactar.** Pesos y contrato de inferencia reproducible, tablas,
   figuras, predicciones, registro de versiones e informe que explique la
   elección y el rendimiento real del detector. El informe final se alimenta
   de los resultados ejecutados del notebook.

Se puede comenzar ahora con selección, diagnóstico, umbrales y explicabilidad.
No es necesario reabrir los tres entrenamientos para comenzar. Si se exige una
localización más sólida como parte central del producto, conviene mejorar el
detector y compararlo antes de congelar la solución y abrir test.

## Evidencia local de esta revisión

- `reports/phase5_readiness/review.json`: comprobaciones y métricas recalculadas.
- `reports/phase5_readiness/sources.json`: enlaces e identificadores consultados.
- `reports/phase5_readiness/drive_evidence/`: copias textuales de resúmenes,
  historiales y predicciones de validation. No contiene datos nuevos de test.

Esta revisión no cambia los notebooks ejecutados, los resultados ni los pesos
de Drive. La fase 5 continúa pendiente de implementación y ejecución.

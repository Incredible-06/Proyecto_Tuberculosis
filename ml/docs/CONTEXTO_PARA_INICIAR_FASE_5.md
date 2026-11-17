# Contexto para iniciar la fase 5 en un nuevo chat

Actualizado al 2 de noviembre de 2026, hora de Lima. Los nombres de corridas
con fecha 20261103 usan UTC. Este archivo permite continuar sin el chat anterior.

## Objetivo y estado

- Proyecto local: `C:/Users/andyr/Desktop/Proyecto_Tuberculosis`.
- Clasificación de radiografías health/sick/TB con ConvNeXt, VMamba e híbrido
  de atención cruzada; localización de lesiones TB con YOLO11n.
- Entrenamientos ejecutados por el usuario en Google Colab GPU. Datos y
  artefactos persistentes en Google Drive.
- Fases 1/2 ejecutadas: entorno, auditoría, particiones y preprocesamiento.
- Fase 3 terminada: tres entrenamientos controlados y cierre comprobado.
- Fase 4: detector entrenado, validado y versionado; rendimiento limitado.
- Fase 5 pendiente: **todavía no existe su notebook**. El usuario quiere
  iniciarla en el chat nuevo; se acordó notebook nuevo + informe + exportación.
- Test y evaluación externa siguen reservados según los registros consultados.

Leer primero `docs/REVISION_FASES_3_4_Y_PLAN_FASE_5.md` y
`reports/phase5_readiness/review.json`. Son la revisión más reciente.
`reports/phase5_readiness/sources.json` guarda los identificadores de Drive.
`reports/phase5_readiness/drive_evidence/` contiene resúmenes, historiales y
las predicciones de validation consultadas: no hace falta descargarlos de nuevo.

## Resultados de clasificación verificados

Se revisaron las 1.241 predicciones por modelo y se recalcularon las matrices,
accuracy y Macro-F1: coinciden. Mismas imágenes, etiquetas, protocolo y hashes
registrados de procedencia. Best/last.pt existen en Drive. Los SHA de pesos
son los declarados por el cierre ejecutado en Colab; no se descargaron otra
vez los checkpoints para recalcularlos durante la última revisión.

| Modelo | Run ID | Épocas / mejor | Macro-F1 | Accuracy | Recall TB |
|---|---|---:|---:|---:|---:|
| convnext_tiny | phase3_convnext_tiny_20261103T023002Z_4a7e2a | 12 / 11 | 0,9427155571 | 0,9709911362 | 94/120 = 0,783333 |
| vmamba_tiny | phase3_vmamba_tiny_20261103T025536Z_4784e4 | 12 / 11 | 0,9496720393 | 0,9685737309 | 101/120 = 0,841667 |
| hybrid_cross_attention | phase3_hybrid_cross_attention_20261103T032205Z_71d127 | 7 / 3 | 0,9644699191 | 0,9782433521 | 107/120 = 0,891667 |

Filas reales / columnas predichas, orden health, sick, TB:

```text
ConvNeXt: [[568, 2, 0], [5, 543, 3], [4, 22, 94]]
VMamba:   [[569, 1, 0], [15, 532, 4], [12, 7, 101]]
Híbrido:  [[568, 2, 0], [8, 539, 4], [6, 7, 107]]
```

El híbrido es el candidato principal según Macro-F1 de validation; falta
formalizar la selección final. Precisión TB 0,963964, especificidad 0,996432.
Omite 13 TB: siete sick y seis health. Las siete épocas terminaron normalmente
por early stopping; cargar la mejor época 3. La regla actual es argmax softmax,
sin umbral operativo ajustado. Una seed y validación reutilizada no prueban
superioridad estadística. Los grupos por duplicados no son pacientes.

SHA-256 registrados de best.pt:

```text
convnext_tiny: 33869c4791569f95c262df2987c9a688649aa4f65c92e51e79f8340c1d0c3616
vmamba_tiny: 56efbe2bc8230f7fb52cc199d89489f98134bb2364cc06fa20f353d471860b53
hybrid_cross_attention: 1db1965bd191456cec05f9d91a45539e7dcc22bfbbab0adcc474f351b7473830
```

Protocolo `configs/phase3_suite.json`: phase3_controlled_v1, seed 20261019,
máximo 12 épocas, warmup 2, paciencia 4, min_delta 0,001, batch 2 con
acumulación 16, FP32, AdamW, LR cabeza 3e-4 / últimas etapas 5e-6, weight
decay 0,01 y CE ponderada calculando pesos solo en train. Inicio ImageNet por
rama; el híbrido no hereda ramas ajustadas a TBX11K. Es atención cruzada
bidireccional de mapas espaciales, no promedio de probabilidades.
ConvNeXt v01/v02 anteriores son antecedentes exploratorios; v02 continúa v01.
No mezclarlos como corridas independientes de esta comparación controlada.

## Detector y sus límites

Run ID: `detector_yolo11n_20261103T032033865005Z`.
YOLO11n desde COCO, Ultralytics 8.3.228, 512px, batch 8, FP32, máximo 35
épocas, paciencia 7. Ejecutó 33; mejor mAP50–95 del historial en época 26.
Validación final de best.pt según detector_summary.json:

- mAP50 = 0,48423225059138064; mAP50–95 = 0,21577616090176216.
- Precisión/recall por caja mostrados en Colab ≈ 0,609 / 0,451.
- Validation: 1.241 imágenes, 120 TB con 176 cajas y 1.121 negativas.
- Train: 559 TB con 851 cajas y 5.232 negativas. tb1199, sin cajas, excluida;
  nunca se trató como negativa. Una clase de detección: tb_lesion.

El recall es por lesión/caja, no sensibilidad por imagen o paciente. P/R de
Ultralytics corresponden al punto de máximo F1 de su curva; falta definir
el punto de operación del producto. No multiplicar los recalls de ambos modelos.
Hubo aprendizaje, pero añadir épocas por sí solo no garantiza mejorar.
Se puede iniciar fase 5 conservándolo como baseline; si se necesita mejor
localización, comparar otra corrida usando validation antes de congelar y abrir test.
**Una alerta TB del clasificador sin cajas debe provocar revisión y no borrarse
automáticamente.** Conservar probabilidades originales y alertas por separado;
no combinar scores arbitrariamente ni usar ausencia de cajas para descartar TB.

Flags de fase 4 pendiente en conversion.json y el cierre de fase 3 son
instantáneas anteriores. Historial, pesos y detector_summary.json evidencian
el entrenamiento posterior. Se localizaron PNG remotos pero no se pudieron
descargar para inspección visual; las curvas se analizaron mediante CSV completos.

## Datos y preprocesamiento congelados

- `configs/data.json`: health=0, sick=1, TB=2; seed 20261019.
- El plan antiguo menciona cuatro clases: documentar la diferencia, no inventar otra.
- Train 5.792 = 2.660 health + 2.572 sick + 560 TB.
- Validation 1.241 = 570 health + 551 sick + 120 TB.
- Test interno etiquetado 1.241, con las mismas cantidades que validation.
- Usar manifests/train.csv, validation.csv y test.csv existentes.
- El test oficial TBX11K sin etiquetas (3.302) es otro conjunto; no calcular
  métricas supervisadas sobre él sin etiquetas.
- RGB 512×512, letterbox y normalización ImageNet. Reutilizar
  `src/tbxdata/dataset.py` para clasificación; no cambiar el preprocesamiento.
- Aumentos solo train: rotación ±5°, brillo/contraste ±0,08; sin flips ni CLAHE.
- Auditoría: 12.278 archivos verificados; sin solapamientos registrados por
  rutas/hashes/grupos. Validation contiene 1.237 grupos en 1.241 imágenes.
  No hay garantía de independencia por paciente.
- Shenzhen/Montgomery siguen reservados: revisar procedencia, licencias,
  solapamientos y etiquetas antes de evaluar. Sus etiquetas binarias requieren
  una evaluación compatible; no inventar tres clases ni anotaciones de cajas.
- validate_manifests(..., all_image_hashes=True) verifica integridad. Comprobar
  hashes de test no equivale a usar sus imágenes para selección o inferencia.

## Drive y artefactos exactos

Base Colab: `/content/drive/MyDrive/Proyecto_Tuberculosis`.

| Recurso | URL |
|---|---|
| Raíz | https://drive.google.com/drive/folders/1eN1n7g3-f8qxCS_DjDPBFmhz2qRdI-zG |
| Experimentos | https://drive.google.com/drive/folders/1oShA2XqeladDmRowawLN0IQ0QAMsYb3k |
| Checkpoints | https://drive.google.com/drive/folders/1StwQrnh-cNYjMDSvAZa7LSidrRV1CZy7 |
| Results | https://drive.google.com/drive/folders/17_bjqSSNiebmgXsYR56ODjdcPG0Tvo1b |
| Cierre fase 3 | https://drive.google.com/drive/folders/1fncC1MxRqjKcK83W2zI0noozNtfy9DfI |
| Experimento híbrido | https://drive.google.com/drive/folders/1ah_kyFzhnRBh0rRub-krY-hk8MWKn5ks |
| Experimento detector | https://drive.google.com/drive/folders/1KjksA8z7NGaIqolZxOVlR1H1983RAqAe |
| Notebook 04 ejecutado | https://colab.research.google.com/drive/1eTn6yj2lmn3Ny2FJO8yBIl2p-ti_bytU |
| Notebook 05 ejecutado | https://colab.research.google.com/drive/1AE1-5SPb4YP3W5QMqk7w-M0yGS0_Zpem |
| ZIP fases 3/4 | https://drive.google.com/file/d/1rtGmdMXTK4O_FUyBQKnFmzf-DbfGViCQ/view |

Clasificadores: `checkpoints/<run_id>/best.pt` y last.pt; resultados en
`experiments/<run_id>/`: summary.json, history.csv, validation_predictions.csv,
best_validation.json, configuración, procedencia y figuras.
Detector: `experiments/<run_id_detector>/training/weights/best.pt` y last.pt;
training/results.csv, training/args.yaml, validation/ y detector_summary.json.

| Archivo best.pt | Drive ID |
|---|---|
| ConvNeXt | 1yvukiuaY2MrMzlNLOwlsADywxpDbDa_o |
| VMamba | 1GVlrGkZqMXnNnhnEqpXW-BlRciWG35vv |
| Híbrido | 16h4sh6D4avXr6nL_Rrwy7BNXnEcaDtRG |
| Detector | 1TmeAidR_3BVss_XO_5RijtKa_o8M0kh3 |

El ZIP tiene revisiones: verificar los bytes/checksum actuales al descargar;
no usar como actual el checksum histórico de delivery.json.
**Conservar notebooks ejecutados 02/03/04/05 y sus outputs en Drive.** Las
versiones locales son plantillas; crear otro notebook y resultados para fase 5.

## Código reutilizable y carga de pesos

- src/tbxtrain/architectures.py: build_classifier, scan_backend y arquitecturas.
- src/tbxtrain/suite.py: protocolo, checkpoints y cierre.
- src/tbxtrain/train.py: evaluate, loaders, atomic_checkpoint y seeds.
- src/tbxtrain/metrics.py y review.py: métricas, figuras y errores.
- src/tbxtrain/detection.py: conversión de detección; no exporta test todavía.
- scripts/create_phase3_completion_notebooks.py: generador de 04/05.
- scripts/package_phase3_completion.py: paquete con checksums.

Best.pt de la suite contiene architecture, run_id, epoch, model_state, config,
provenance, classes, data_config y validation. Para restaurarlo: construir con
build_classifier(nombre, pretrained=False), cargar estrictamente model_state,
pasar a GPU y eval. No hace falta descargar otra vez ImageNet para restaurar
el checkpoint entrenado completo. Conservar compatibilidad de arquitectura.
Verificar los archivos originales registrados en provenance; añadir un módulo
de evaluación no equivale a alterar el código original del entrenamiento.

## Reparación CUDA resuelta: evitar repetir el diagnóstico

Runtime de los entrenamientos: Python 3.13.16, torch 2.11.0+cu130, CUDA 13,
NVIDIA L4; selective scan **oflex**. Colab puede cambiar: verificar el runtime
actual y evitar reemplazar torch/CUDA indiscriminadamente.
VMamba vendorizado, revisión 2ed52ead062a51a64521ed3871d52914bf532876 de
MzeroMiko/VMamba; hashes en vendor/vmamba/source.json.

scripts/setup_vmamba_colab.py incorpora la reparación que ya funcionó:

- CUDA13 retiró helpers CUB: cub::LaneId() pasa a
  ::cuda::ptx::get_sreg_laneid() con <cuda/ptx>; cub::CTA_SYNC() pasa a
  __syncthreads(). Se conservan respaldos y logs.
- MAX_JOBS=1, nvcc --threads 1, nvcc compatible y C++ según headers de torch.
- Pip verbose, --no-build-isolation y --no-deps, conservando torch/CUDA.

El usuario instaló el kernel y completó smokes GPU y entrenamientos de los
tres modelos. En un runtime nuevo puede requerirse recompilar; en una sesión
con oflex activo, reutilizarlo. No sustituir silenciosamente por torch_reference.
También está el hotfix setup_vmamba_colab_reparado.py en Drive raíz, ID
1TaVB6kh1Egp3D-6oIcQ_s8g_9D_XbEv5; se copió al proyecto Colab y ejecutó
con runpy. El instalador local actual ya incluye esos cambios.
Pesos ImageNet VMamba en Drive/pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth:
SHA-256 d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a.

Warnings no deterministas de atención/adaptive pooling no interrumpieron el
smoke del híbrido. Se usa determinismo warn_only=True; misma seed no garantiza
igualdad bit a bit. No activar determinismo estricto que aborte sin comprobarlo.

## Trabajo previsto de fase 5

Nombre propuesto aún sin crear:
`notebooks/06_fase5_seleccion_explicabilidad_evaluacion.ipynb`.

1. Preparación reproducible, recuperación de best.pt y verificación de hashes,
   configuración y manifiestos. Smoke de carga/inferencia separado.
2. Selección usando validation; híbrido como candidato principal. Documentar
   coste, errores e incertidumbre; conservar las otras arquitecturas.
3. Diagnóstico del detector: cajas, errores, sensibilidad por imagen, recall
   por lesión y falsos positivos por imagen frente al umbral.
4. Elegir umbrales del clasificador y detector usando validation. Aún no hay
   objetivo de sensibilidad/especificidad acordado; mostrar los compromisos
   y no inventar un objetivo clínico alcanzado.
5. Concordancia explícita con probabilidades originales y alertas separadas;
   conservar alerta TB cuando no hay cajas y evaluar las discrepancias.
6. Grad-CAM donde corresponda y se verifiquen gradientes/comportamiento;
   considerar ramas del híbrido. Un mapa visual no valida por sí mismo lesiones.
7. Congelar pesos, preprocesamiento, umbrales, regla y hashes antes de test.
   Si se mejora el detector, comparar nuevas corridas antes de este paso.
8. Evaluar test reservado tras congelar decisiones: accuracy, Macro-F1, AUC,
   sensibilidad/especificidad TB, matrices, detección e intervalos. No ajustar
   con test. Los grupos conocidos son duplicados, no identidad del paciente.
9. Exportar pesos/contrato de inferencia, tablas, figuras, predicciones e
   informe respaldado por las salidas ejecutadas. Externos solo tras revisar
   procedencia, solapamientos y etiquetas.

La revisión previa no implementó ni ejecutó estos pasos de fase 5.

## Preferencias y práctica de trabajo

- Español; actualizaciones breves y frecuentes. Ante fallos de Colab, dar la
  celda concreta para reemplazar y su ubicación. Evitar largas esperas silenciosas.
- Conservar outputs; nuevos notebooks o celdas sin borrar salidas anteriores.
- Usar el conector y skill de Drive cuando estén disponibles. Verificar subidas
  antes de entregar enlaces. No atribuir al agente ejecuciones GPU del usuario.
- Muchos cambios locales anteriores sin commit: conservarlos; no limpiar ni
  descartar archivos. No se ha pedido worktree nuevo.
- PowerShell. Python del PATH puede fallar por pyenv; para verificaciones
  existe tmp/phase3_test_env/Scripts/python.exe, solo CPU. Su versión de torch
  no es la del Colab del usuario. Es un entorno temporal.
- Pruebas previas: 7 suite/detector + 11 entrenamiento/revisión pasaron en CPU;
  carga estricta de pesos oficiales verificada. Evidencia GPU posterior en Drive.
- delivery.json/validation.json antiguos pueden mostrar fases pendientes:
  son instantáneas anteriores, no el estado actual. No repetir trabajo por ello.

## Etiquetas de los notebooks

F4 identifica fase 4. E3 identifica el tercer bloque de ejecución,
«Validación y salida del detector»; E1 revisa cajas y E2 entrena.
Son etiquetas del proyecto, no convenciones de Colab ni números de época.
No había una leyenda explícita. En fase 5 conviene usar «Fase 5 · Paso 1»,
etc., para evitar abreviaturas ambiguas.

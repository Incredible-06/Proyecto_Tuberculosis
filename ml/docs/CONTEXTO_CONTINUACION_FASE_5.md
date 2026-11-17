# Contexto de continuación para completar la Fase 5

Actualizado al 15 de noviembre de 2026 (hora de Lima; los identificadores de corridas con `20261116` usan UTC). Este documento resume el estado completo del proyecto tras incorporar **ViT-B/16** y el **Híbrido ConvNeXt-Tiny + ViT-B/16**, y detalla todo lo que falta para cerrar al 100 % la Fase 5.

---

## 1. Estado general del proyecto

- **Workspace local:** `C:\Users\andyr\Desktop\Proyecto_Tuberculosis` (la parte ML vive en `ml/`; el sistema web/API en `web/` y `api/`).
- **Fases 1 y 2 (completadas):** entorno validado, auditoría de duplicados cercanos, particiones congeladas (`seed=20261019`) y preprocesamiento RGB `512×512` con letterbox y normalización ImageNet (`ml/configs/data.json`).
  - `train`: `5.792` imágenes (`2.660 health`, `2.572 sick`, `560 TB`)
  - `validation`: `1.241` imágenes (`570 health`, `551 sick`, `120 TB`, `1.237` grupos de contenido)
  - `test` interno: `1.241` imágenes (`570 health`, `551 sick`, `120 TB`) — **RESERVADO, SIN EVALUAR**
  - `reserved_external`: `576` imágenes (`Montgomery`/`Shenzhen` y `DA`/`DB`) — **RESERVADO, SIN EVALUAR**
- **Fase 3 (completada y extendida a 5 clasificadores controlados):** todos entrenados bajo el mismo protocolo (`epochs=12`, `warmup_epochs=2`, `batch_size=2`, `accumulation_steps=16`, `head_lr=3e-4`, `backbone_lr=5e-6`, `weight_decay=0.01`, `gradient_clip=1.0`, `patience=4`, `min_delta=0.001`, `precision=fp32`, `seed=20261019`, selección por `validation_macro_f1`).
- **Fase 4 (entrenado baseline, diagnóstico pendiente en Fase 5):** `YOLO11n` (`detector_yolo11n_20261103T032033865005Z`), 33/35 épocas, mejor en época 26: `mAP50=0,484232`, `mAP50–95=0,215776`, precisión/recall por caja en punto Ultralytics `≈ 0,609 / 0,451`.
- **Fase 5 (en progreso — NO completada al 100 %):** solo se ejecutó la preparación inicial de validación (`06_fase5_seleccion_explicabilidad_evaluacion.ipynb`, con `RUN_DETECTOR_VALIDATION=False`) y se actualizó [Informe_Avance_Fase_5.docx](file:///c:/Users/andyr/Desktop/Proyecto_Tuberculosis/ml/docs/Informe_Avance_Fase_5.docx) con los cinco clasificadores. Faltan diagnóstico del detector, revisión visual, Grad-CAM/explicabilidad, coste comparable, incertidumbre por grupos, selección final, congelamiento, evaluación única de `test` y exportación final.

---

## 2. Resultados verificados de los 5 clasificadores en `validation` (`1.241` imágenes)

| Modelo | Arquitectura / código | Run ID | Épocas / mejor | Exactitud | Macro F1 | TB (`argmax`) | TB Prec. | TB Espec. | TB ROC AUC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | `convnext_tiny` (`architectures.py`) | `phase3_convnext_tiny_20261103T023002Z_4a7e2a` | 12 / 11 | 97,10 % | 0,9427 | 94 / 120 (78,33 %) | 96,91 % | 99,73 % | 0,9958 |
| VMamba-Tiny | `vmamba_tiny` (`architectures.py`) | `phase3_vmamba_tiny_20261103T025536Z_4784e4` | 12 / 11 | 96,86 % | 0,9497 | 101 / 120 (84,17 %) | 96,19 % | 99,64 % | 0,9934 |
| Híbrido (ConvNeXt+VMamba) | `hybrid_cross_attention` (`architectures.py`) | `phase3_hybrid_cross_attention_20261103T032205Z_71d127` | 7 / 3 | **97,82 %** | **0,9645** | 107 / 120 (89,17 %) | 96,40 % | 99,64 % | 0,9958 |
| ViT-B/16 | `vit_b_16` (`vit_experiment.py`) | `phase3_vit_b_16_20261116T014808Z_2e8a27` | 12 / 11 | 97,74 % | 0,9644 | **110 / 120 (91,67 %)** | 94,02 % | 99,38 % | 0,9969 |
| Híbrido (ConvNeXt+ViT-B/16) | `hybrid_convnext_vit` (`hybrid_vit_experiment.py`) | `phase3_hybrid_convnext_vit_20261116T031647Z_376827` | 6 / 2 | **97,82 %** | 0,9630 | 105 / 120 (87,50 %) | **97,22 %** | **99,73 %** | **0,9975** |

Matrices de confusión en validación (filas reales, columnas predichas, orden `health`, `sick`, `TB`):

```text
ConvNeXt-Tiny:               [[568, 2, 0], [5, 543, 3], [4, 22, 94]]
VMamba-Tiny:                 [[569, 1, 0], [15, 532, 4], [12, 7, 101]]
Híbrido (ConvNeXt+VMamba):   [[568, 2, 0], [8, 539, 4], [6, 7, 107]]
ViT-B/16:                    [[566, 4, 0], [7, 537, 7], [3, 7, 110]]
Híbrido (ConvNeXt+ViT-B/16): [[566, 3, 1], [6, 543, 2], [4, 11, 105]]
```

SHA-256 registrados de `best.pt`:

```text
convnext_tiny:               33869c4791569f95c262df2987c9a688649aa4f65c92e51e79f8340c1d0c3616
vmamba_tiny:                 56efbe2bc8230f7fb52cc199d89489f98134bb2364cc06fa20f353d471860b53
hybrid_cross_attention:      1db1965bd191456cec05f9d91a45539e7dcc22bfbbab0adcc474f351b7473830
vit_b_16:                    2873fbcff636ad151b41451022b0fc34eeb197866cf5acd200e3f764f4f3d50d
hybrid_convnext_vit:         bd88a5ee557b6ad1b6b1da59cbada3a47946d69defbe897b25721bbd357ecf1d
```

Resumen de umbrales exploratorios `p_TB >= t` en validación (los 3 punteros):

- **Híbrido (ConvNeXt+VMamba):** `0,10 -> 111 TB, 9 FN, 5 FP` | `0,50 -> 107 TB, 13 FN, 4 FP` | `0,90 -> 100 TB, 20 FN, 1 FP`
- **ViT-B/16:** `0,10 -> 115 TB, 5 FN, 22 FP` | `0,50 -> 110 TB, 10 FN, 6 FP` | `0,90 -> 100 TB, 20 FN, 2 FP`
- **Híbrido (ConvNeXt+ViT-B/16):** `0,10 -> 111 TB, 9 FN, 15 FP` | `0,30 -> 108 TB, 12 FN, 5 FP` | `0,50 -> 105 TB, 15 FN, 2 FP` | `0,90 -> 88 TB, 32 FN, 0 FP` (AP = `0,9806`)

---

## 3. Regla crítica de procedencia y archivos protegidos

`src/tbxtrain/phase5.py` (`verify_provenance`) comprueba el SHA-256 exacto de cada archivo original registrado en `summary.json['provenance']`.

**NO modificar** los siguientes archivos para no invalidar la verificación de los 5 checkpoints ya entrenados:
- `ml/configs/data.json`, `ml/configs/phase3_suite.json`, `ml/configs/vit_controlled.json`, `ml/configs/hybrid_convnext_vit_controlled.json`
- `ml/manifests/*`
- `ml/vendor/vmamba/*`
- `ml/src/tbxdata/*.py`
- `ml/src/tbxtrain/`: `__init__.py`, `architectures.py`, `detection.py`, `metrics.py`, `protocol.py`, `review.py`, `suite.py`, `train.py`, `vit_experiment.py`, `hybrid_vit_experiment.py`

Cualquier nueva funcionalidad de Fase 5 (Grad-CAM / rollout, diagnóstico del detector, bootstrap por grupos, coste computacional, congelamiento y evaluación reservada de `test`) debe implementarse en **módulos nuevos** (p. ej. `ml/src/tbxtrain/phase5_explainability.py`, `ml/src/tbxtrain/phase5_evaluation.py`) y un **notebook nuevo** (p. ej. `ml/notebooks/07_fase5_gradcam_seleccion_y_test.ipynb`) con su propio ZIP, conservando ejecutados en Drive los notebooks `02` a `06`, `04b` y `04c`.

---

## 4. Ubicación de artefactos locales y en Google Drive

### En local (`ml/`)
- Evidencias previas (3 modelos + preparación Fase 5): `ml/reports/phase5_readiness/drive_evidence/` y `ml/reports/phase5_review/drive_evidence/`
- Evidencias de `ViT-B/16`: `ml/reports/phase5_review/vit_evidence/` (`vit_report_payload.json`, `vit_summary.json`, `vit_history.csv`, `vit_validation_predictions.csv`, figuras y `04b_entrenamiento_vit_comparacion_ejecutado.ipynb`)
- Evidencias de `Híbrido ConvNeXt + ViT-B/16`: `ml/reports/phase5_review/hybrid_vit_evidence/` (`hybrid_vit_report_payload.json`, `hybrid_vit_summary.json`, `hybrid_vit_history.csv`, `hybrid_vit_validation_predictions.csv`, figuras y `04c_entrenamiento_hibrido_convnext_vit_ejecutado.ipynb`)
- Informe vigente: `ml/docs/Informe_Avance_Fase_5.docx` y `ml/docs/Informe_Avance_Fase_5.md`, generados por `ml/scripts/build_phase5_report.py`.

### En Google Drive (`MyDrive/Proyecto_Tuberculosis/`)
- Carpeta raíz: `https://drive.google.com/drive/folders/1eN1n7g3-f8qxCS_DjDPBFmhz2qRdI-zG`
- Experimentos: `https://drive.google.com/drive/folders/1oShA2XqeladDmRowawLN0IQ0QAMsYb3k`
- Checkpoints: `https://drive.google.com/drive/folders/1StwQrnh-cNYjMDSvAZa7LSidrRV1CZy7`
- Resultados: `https://drive.google.com/drive/folders/17_bjqSSNiebmgXsYR56ODjdcPG0Tvo1b`
- Notebook 06 (preparación Fase 5 ejecutada): `https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD`
- Notebook 04b (`ViT-B/16` ejecutado): `https://colab.research.google.com/drive/1tvKNsTzirSa4G7wXO_l7xZqp3YtoM0fU`
- Exportación ViT-B/16: `https://drive.google.com/drive/folders/1m5WrwO8KpmZq5XYJEt3KO22iOcb1Igh6`
- Notebook 04c (`Híbrido ConvNeXt+ViT` ejecutado): `https://colab.research.google.com/drive/1hr1r-vRpuHDOUV9KmRFlh9AzL-k4Tbvw`
- Corrida `Híbrido ConvNeXt+ViT`: `https://drive.google.com/drive/folders/1h3QICmhofofrIFbhMY3tOSdkBrx8xVOT`
- Exportación 5 modelos (`Híbrido ConvNeXt+ViT`): `https://drive.google.com/drive/folders/1NsRcHaE7VCyz2drTjTIAF_JasZvgE0tF`

---

## 5. Qué falta para completar la Fase 5 al 100 %

1. **Diagnóstico del detector en `validation` (`RUN_DETECTOR_VALIDATION`):** ejecutar inferencia de `detector_yolo11n_20261103T032033865005Z` sobre las `1.241` imágenes de validación, generar la tabla de umbrales (`detector_threshold_table`) y la tabla de concordancia exploratoria con los 3 modelos punteros. Decidir si YOLO11n queda como baseline o requiere una mejora antes de congelar.
2. **Revisión visual de errores y cajas:** inspeccionar falsos negativos TB, falsos positivos y casos con discrepancia entre clasificador y detector.
3. **Explicabilidad verificada (Grad-CAM / mapas de relevancia):**
   - `ConvNeXt-Tiny`: Grad-CAM sobre `features[-1]` (`16×16`).
   - `VMamba-Tiny` / `Híbrido ConvNeXt+VMamba`: verificar primero qué capas espaciales admiten gradientes limpios sin romper `oflex` (`conv` `16×16` y mapa previo a `classifier` en `mamba`, además de pesos de `CrossFusion`).
   - `ViT-B/16` y `Híbrido ConvNeXt+ViT-B/16`: Attention Rollout / Grad-CAM espacial sobre los `32×32` tokens del encoder de ViT + Grad-CAM sobre `conv` (`16×16`) y atención cruzada en `CrossFusion`.
   - Comprobar numéricamente gradientes finitos, no nulidad y contraste frente a aleatorización/clase (un mapa visual por sí solo no valida localización de lesiones).
4. **Coste computacional comparable:** medir parámetros totales/entrenables, VRAM pico y latencia media por imagen (`batch=1` y batch fijo en GPU Colab y/o CPU) para los 3 candidatos principales y el detector.
5. **Incertidumbre por grupos:** calcular intervalos de confianza (bootstrap estratificado/agrupado por `group_id`) en validación para exactitud, Macro-F1, sensibilidad TB, especificidad y AUC.
6. **Selección final y congelamiento inmutable:** elegir el clasificador ganador entre `Híbrido (C+VM)`, `ViT-B/16` e `Híbrido (C+ViT)`, fijar umbral operativo `p_TB`, umbral de detección, NMS y regla definitiva de concordancia (conservando la alerta del clasificador cuando no haya cajas). Generar contrato JSON inmutable con hashes de código, pesos y configuración antes de tocar `test`.
7. **Evaluación única sobre `test` interno (`1.241` imágenes) y externos:** ejecutar una sola vez el contrato congelado sobre `manifests/test.csv` (sin reajustar nada después) y, si se evalúan `Montgomery`/`Shenzhen`, revisar primero procedencia, solapamientos y mapeo binario compatible.
8. **Informe final de Fase 5 y exportación:** actualizar el documento Word final, reemplazar si corresponde la versión en Drive y exportar pesos/contrato para `api/` y `web/`.

---

## 6. Entornos locales disponibles

- **Pruebas y scripts ML (CPU):** `ml/tmp/phase3_test_env/Scripts/python.exe` (`torch`, `torchvision`, `timm`, `matplotlib`, `pandas`).
- **Generación de `.docx`:** `py -3` (Python 3.14 con `python-docx`).
- **Google Drive CLI:** plugin `gdrive` conectado (`andyrodriguez0604@gmail.com`); leer `C:\Users\andyr\.gemini\config\plugins\gdrive\skills\gdrive\SKILL.md` antes de usarlo.

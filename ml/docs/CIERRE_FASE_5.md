# Cierre de Fase 5 (100 % completada)

Fecha de cierre: 16 de noviembre de 2026 (UTC).

## 1. Contrato congelado e inmutable
- Archivo de contrato: `ml/configs/phase5_frozen_contract.json` (`schema: phase5_frozen_contract_v1`, `decisions_frozen: true`).
- Clasificador seleccionado: **ViT-B/16** (`phase3_vit_b_16_20261116T014808Z_2e8a27`, época 11, SHA-256 `2873fbcff636ad151b41451022b0fc34eeb197866cf5acd200e3f764f4f3d50d`), umbral operativo `p_TB >= 0.30` (conservando `argmax` y probabilidades originales).
- Detector auxiliar: **YOLO11n** (`detector_yolo11n_20261103T032033865005Z`, SHA-256 `3e28e82987c8c2c84ca503f639354c7da1130472165adae8642737c7b5263dd0`), `conf >= 0.25`, `imgsz=512, iou=0.7, max_det=300, half=False, rect=False`.
- Regla de concordancia: `additive_4state_no_suppression_v1` (`both_alert`, `classifier_only_review`, `detector_only_review`, `neither_alert`), sin supresión de alertas del clasificador ni multiplicación de scores.

## 2. Resultados resumidos
- **Validación (`1.241` imágenes, `1.237` grupos `group_id`)** — `ml/reports/phase5_diag/phase5_diag_20261116T042213Z_ac3bcb/`:
  - Empate estadístico entre `Híbrido (C+VM)` (`0.9645`), `ViT-B/16` (`0.9644`) e `Híbrido (C+ViT)` (`0.9630`); todos los IC 95 % de deltas pareados cruzan `0`.
  - Explicabilidad verificada (`60/60` chequeos de finitud, gradientes no nulos, contraste y aleatorización superados); `ViT-B/16` logró el mayor enriquecimiento en cajas GT (`1.42×`).
  - Coste NVIDIA L4 (`batch=1`, FP32): `ViT-B/16` `31.3 ms` (`389 MB` VRAM), `YOLO11n` `19.4 ms` (`41.9 MB` VRAM).
- **Test interno reservado (`1.241` imágenes, evaluación única)** — `ml/reports/phase5_final/phase5_final_20261116T051424Z_f5c880/test_evaluation/`:
  - `ViT-B/16` (`argmax`): Exactitud **`97,18 %`**, Macro-F1 **`0,9551`**, TB ROC AUC **`0,9941`**, TB AP **`0,9683`**.
  - `ViT-B/16` (`p_TB >= 0,30`): Sensibilidad TB `90,00 %`, Especificidad TB `99,11 %`.
  - `YOLO11n` (`conf >= 0,25`): Recall por lesión `48,37 %`, sensibilidad por imagen (cualquier caja) `73,33 %`, falsas alarmas no-TB `0,18 %` (`1` `health`, `1` `sick`).
  - Concordancia combinada (`p_TB >= 0,30` o `YOLO11n >= 0,25`): Sensibilidad TB **`95,83 %` (`115/120`)**, revisión en no-TB **`0,98 %` (`11/1.121`)**.
- **Externos (`Montgomery` `138` y `Shenzhen` `662`, `0` solapamiento con `train`/`validation`)** — `ml/reports/phase5_final/phase5_final_20261116T051424Z_f5c880/external_evaluation/`:
  - `ViT-B/16` (`p_TB >= 0,30`) sufrió *domain shift* hospitalario en especificidad (`25,00 %` en Montgomery, `12,88 %` en Shenzhen; `p_health` colapsó en radiografías normales externas).
  - `YOLO11n` aislado (`conf >= 0,25`) mantuvo **AUC `0,811`** (especificidad **`100,0 %`**, `0/80` FP) en Montgomery y **AUC `0,804`** (especificidad **`97,55 %`**, `8/326` FP) en Shenzhen.

## 3. Paquete exportado para `api/` y `web/`
Ubicado y verificado en `ml/models/frozen_phase5/`:
- `vit_b_16_best.pt`
- `yolo11n_best.pt`
- `phase5_frozen_contract.json`
- `data_config.json`
- `deployment_manifest.json`

## 4. Recomendaciones para futura Fase 6 (fuera del contrato de Fase 5)
- Incorporar segmentación/recorte pulmonar, normalización de contraste controlada (ablación CLAHE) y aumentos de dominio para reducir atajos de adquisición hospitalaria.
- Validar generalización inter-institucional con partición *Leave-One-Hospital-Out* (LOHO) y calibración por sitio antes de desplegar fuera del dominio TBX11K.

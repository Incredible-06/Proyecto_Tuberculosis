# Contexto para continuar el sistema web

Actualizado el 16 de noviembre de 2026. Este documento resume el estado técnico
del sistema tras la integración del contrato congelado de Fase 5 y el
refinamiento de la interfaz web.

## Estado general

La Fase 5 de ML está cerrada y congelada en `ml/models/frozen_phase5/`
(`vit_b_16_best.pt`, `yolo11n_best.pt`, `phase5_frozen_contract.json`,
`data_config.json`, `deployment_manifest.json`). El motor de inferencia ya está
integrado en `api/` y `web/` sin modificar los archivos protegidos por
`provenance` en `ml/src/`.

La aplicación es apoyo a la decisión clínica (no diagnóstico definitivo) e
incluye advertencia explícita sobre la limitación de dominio `TBX11K` y el
*domain shift* observado en cohortes externas (`Montgomery` y `Shenzhen`).

## Organización actual

```text
api/       FastAPI, migraciones SQLite (0001–0003), motor congelado y 43 pruebas.
web/       React 19 + Vite 7 + TypeScript + Tailwind CSS 4, servido por FastAPI.
docs/      Backlog, criterios de aceptación y contexto del producto.
ml/        Workspace ML congelado (datos, entrenamiento, reportes y frozen_phase5).
.local/    Entornos (.venv), SQLite, radiografías y mapas generados (ignorado por Git).
run.ps1    Arranque local unificado (dependencias base + PyTorch CPU/Ultralytics + Vite build + Uvicorn).
```

## Capacidades implementadas

### Identidad y permisos (`HU01–HU03`)
- Sesiones por cookie `HttpOnly` (`SameSite=Strict`), bloqueo por intentos y auditoría.
- Matriz de permisos exacta para `administrador`, `tecnologo_radiologia`,
  `medico_programa_tb`, `medico_radiologo` y `coordinador_estrategia_tb`.
- Confirmación modal accesible (`<dialog>`) al desactivar usuarios y guardas para
  no perder la última cuenta administradora activa ni autodesactivarse.

### Evaluaciones, carga y calidad (`HU05–HU08`)
- Alta con código anonimizado `TB-AAAAMMDD-XXXXXX` y filtro anti-PII en notas.
- Carga PNG/JPG (rechazo DICOM, límite configurable 10 MB) y reemplazo controlado
  en estado `imagen_cargada`.
- Validación de calidad `HU07`: decodificación real con Pillow/NumPy, mínimo
  `224 px` (recomendado `512 px`), relación de aspecto, desviación estándar y
  rango dinámico (`p99 - p01`); promueve a `lista_para_analisis`.
- Búsqueda y filtrado en servidor (`status`, `search`, `review_required`, `concordance_state`).

### Inferencia congelada, explicabilidad y revisión (`HU09–HU15`, `HU22`)
- **Versionado (`HU22`)**: `sync_frozen_model_version` verifica el manifiesto y
  registra la versión activa en `model_versions`; cada predicción referencia
  `model_version_id`.
- **Motor (`api/app/inference.py`)**: verifica SHA-256 de contrato y pesos antes
  de cargar; aplica `letterbox(512)` + normalización `imagenet_rgb`; ejecuta
  `ViT-B/16` (`argmax`, `p_health`, `p_sick`, `p_TB`, corte `0.30`), `YOLO11n`
  (`conf >= 0.25`, `imgsz=512`, `iou=0.7`) con proyección inversa a coordenadas
  originales, concordancia aditiva de 4 estados (`both_alert`,
  `classifier_only_review`, `detector_only_review`, `neither_alert`), alerta de
  baja confianza (`HU12`), `Grad-CAM` sobre tokens espaciales (`layers[-2]`,
  clase `TB`) y `Attention Rollout` (`12` bloques encoder), exportando PNG RGBA
  recortados al área útil bajo `.local/runtime/api/explainability/`.
- **Seguridad clínica**: `POST /analyze` accesible para tecnólogo y médicos;
  lectura de predicción y mapas restringida a `predictions.read` (`Cache-Control:
  private, no-store`).
- **Observaciones (`HU15`)**: registro médico con filtro anti-PII (`observations.write`)
  y transición `analizada` → `revisada`; lectura para coordinador (`observations.read`).

## Rutas de la API

| Ruta | Comportamiento |
|---|---|
| `GET /api/v1/health` | Disponibilidad de la API. |
| `POST /api/v1/auth/login`, `logout`, `GET /me` | Autenticación y sesión vigente. |
| `GET/POST /api/v1/users`, `PATCH /api/v1/users/{id}`, `GET /api/v1/roles` | Gestión de cuentas y roles. |
| `GET /api/v1/model-versions` | Versiones de modelo registradas (`HU22`). |
| `GET/POST /api/v1/evaluations`, `GET /api/v1/evaluations/{id}` | Listado filtrado y detalle. |
| `POST/PUT/GET /api/v1/evaluations/{id}/radiograph` | Carga, reemplazo pre-validación y descarga. |
| `POST /api/v1/evaluations/{id}/radiograph/validate` | Control de calidad `HU07`. |
| `POST /api/v1/evaluations/{id}/analyze` | Ejecución del motor congelado Fase 5 (`HU09`). |
| `GET /api/v1/evaluations/{id}/prediction` | Probabilidades, concordancia, cajas y versión (`HU10–HU13`, `HU22`). |
| `GET /api/v1/evaluations/{id}/prediction/explainability/{kind}` | Mapas `gradcam` y `rollout` en PNG RGBA (`HU14`). |
| `GET/POST /api/v1/evaluations/{id}/observations` | Observaciones clínicas y cierre `revisada` (`HU15`). |

## Comandos de verificación

```powershell
# Arrancar todo el sistema localmente
.\run.ps1

# Pruebas de API (43 pruebas)
.local/runtime/api/.venv/Scripts/python.exe -m pytest api/tests -q --basetemp .local/runtime/api/pytest-tmp-direct -o cache_dir=.local/runtime/api/pytest-cache

# Frontend
npm run typecheck --prefix web
npm run build --prefix web
```

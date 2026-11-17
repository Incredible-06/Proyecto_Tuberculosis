# API

Implementación de HU01–HU03, HU05–HU15 y HU22: autenticación, sesiones
revocables, matriz de permisos, administración de cuentas, evaluaciones
anonimizadas, carga y validación de calidad de radiografías (HU07), inferencia
congelada de Fase 5 (`ViT-B/16` + `YOLO11n` + concordancia aditiva en 4 estados +
mapas Grad-CAM/Rollout), versionado de modelo y observaciones clínicas.

## Ejecutar localmente

La forma recomendada es ejecutar `.\run.ps1` desde la raíz del repositorio.
El script crea el entorno, instala dependencias base y ML (PyTorch CPU +
Ultralytics si faltan), genera el secreto local, construye el frontend y arranca
Uvicorn.

Para hacerlo manualmente:

```powershell
# El venv existente puede no incluir pip; use el Python del sistema.
py -m pip --python .local/runtime/api/.venv install -r api/requirements.txt
py -m pip --python .local/runtime/api/.venv install --index-url https://download.pytorch.org/whl/cpu torch torchvision
py -m pip --python .local/runtime/api/.venv install -r api/requirements-ml.txt

$env:APP_SESSION_SECRET = 'genere-un-secreto-aleatorio-de-al-menos-32-caracteres'
.local/runtime/api/.venv/Scripts/python.exe -m app.bootstrap_admin --username admin.local --display-name 'Administrador inicial'
.local/runtime/api/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir api --reload
```

La interfaz queda disponible en `http://127.0.0.1:8000/`, la API en
`/api/v1/` y el contrato interactivo en `/docs`. Si `web/dist` no existe, la
raíz responde con un mensaje para ejecutar `npm run dev` dentro de `web/`.

Variables relevantes:

| Variable | Uso |
|---|---|
| `APP_SESSION_SECRET` | Secreto de sesiones; mínimo 32 caracteres. |
| `APP_DATABASE_PATH` | Ruta de SQLite; por defecto `.local/runtime/api/tuberculosis.sqlite3`. |
| `APP_STORAGE_ROOT` | Raíz de radiografías y mapas; por defecto `.local/runtime/api/`. |
| `APP_MAX_UPLOAD_BYTES` | Tamaño máximo de imagen; por defecto 10 MB. |
| `APP_FROZEN_MODEL_DIR` | Paquete congelado de Fase 5; por defecto `ml/models/frozen_phase5`. |
| `APP_INFERENCE_DEVICE` | Dispositivo (`auto`, `cpu`, `cuda`); por defecto `auto`. |
| `APP_MIN_IMAGE_DIMENSION` | Resolución mínima HU07; por defecto `224` px. |
| `APP_RECOMMENDED_IMAGE_DIMENSION` | Resolución recomendada HU07; por defecto `512` px. |
| `APP_LOW_CONFIDENCE_THRESHOLD` | Umbral fijo HU12; por defecto `0.70`. |
| `APP_COOKIE_SECURE` | Cookies `Secure`; activar en producción con TLS. |
| `APP_ENV` | `development` o `production`. |

En producción, configure `APP_ENV=production`, un secreto persistente, TLS en
el despliegue y `APP_COOKIE_SECURE=true`.

## Rutas

- `GET /api/v1/health`
- `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`
- `GET/POST /api/v1/users`, `PATCH /api/v1/users/{user_id}`, `GET /api/v1/roles`
- `GET /api/v1/model-versions`
- `GET/POST /api/v1/evaluations`, `GET /api/v1/evaluations/{id}`
- `POST/PUT/GET /api/v1/evaluations/{id}/radiograph`
- `POST /api/v1/evaluations/{id}/radiograph/validate`
- `POST /api/v1/evaluations/{id}/analyze`
- `GET /api/v1/evaluations/{id}/prediction`
- `GET /api/v1/evaluations/{id}/prediction/explainability/{gradcam|rollout}`
- `GET/POST /api/v1/evaluations/{id}/observations`

## Pruebas

```powershell
.local/runtime/api/.venv/Scripts/python.exe -m pytest api/tests -q --basetemp .local/runtime/api/pytest-tmp-direct -o cache_dir=.local/runtime/api/pytest-cache
```

La primera cuenta se crea solo mediante el comando local y no tiene contraseña
por defecto. Después, las rutas de administración exigen una sesión con los
permisos `users.read`, `users.manage` o `roles.read`, según corresponda.

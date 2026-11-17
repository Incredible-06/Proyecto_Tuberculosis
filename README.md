# Sistema web de apoyo para tuberculosis

Sistema web para apoyar el cribado y la revisión clínica de radiografías de
tórax mediante el motor congelado de Fase 5 (`ViT-B/16` + `YOLO11n`). Es una
herramienta de apoyo a la decisión clínica y no sustituye el diagnóstico médico
ni las pruebas confirmatorias.

## Estructura

```text
api/            # FastAPI, migraciones SQLite, motor congelado, demo_data/ y pruebas
web/            # React 19 + Vite 7 + TypeScript + Tailwind CSS 4
docs/           # backlog, criterios de aceptación y contexto técnico
ml/             # código, configuraciones y paquete congelado ml/models/frozen_phase5/
.local/         # entornos (.venv), SQLite y archivos generados en ejecución (ignorado por Git)
run.ps1         # arranque local automatizado
```

## Estado actual

- **HU01–HU03**: autenticación por cookie `HttpOnly`, administración de cuentas y
  matriz de permisos para los 5 roles.
- **HU05–HU08**: registro anonimizado (`TB-AAAAMMDD-XXXXXX`), carga y reemplazo
  controlado de PNG/JPG, validación de calidad (`HU07`) y búsqueda/triaje en servidor (`HU08`).
- **HU09–HU14 y HU22**: inferencia congelada (`ViT-B/16` con `p_TB >= 0.30` +
  `YOLO11n` con `conf >= 0.25`), concordancia aditiva en 4 estados, alerta de
  baja confianza, zonas focales, mapas `Grad-CAM` / `Attention Rollout`,
  advertencia explícita de dominio `TBX11K` y trazabilidad de versión del modelo.
- **HU15**: registro de observaciones clínicas médicas y cierre en estado `revisada`.

Guía de demostración y cuentas: [DEMO_ESTADO_ACTUAL.md](DEMO_ESTADO_ACTUAL.md).

## Clonar y ejecutar en otro equipo

Requisitos: **Git + Git LFS**, **Python 3.12+** y **Node.js 20+**.

```powershell
git clone https://github.com/Incredible-06/Proyecto_Tuberculosis.git
cd Proyecto_Tuberculosis
git lfs pull
.\run.ps1
```

Al ejecutar `.\run.ps1` por primera vez:
1. Verifica los pesos en `ml/models/frozen_phase5/` (`vit_b_16_best.pt` y `yolo11n_best.pt` vía Git LFS).
2. Crea `.local/runtime/api/.venv` e instala dependencias base y ML (`PyTorch CPU` + `Ultralytics`).
3. Instala dependencias de `web/` y compila `web/dist`.
4. Si no existe `.local/runtime/api/tuberculosis.sqlite3`, restaura automáticamente la base y estudios de muestra desde `api/demo_data/`.
5. Levanta el sistema en `http://127.0.0.1:8000/`.

Para desarrollo con recarga y Vite:

```powershell
# Terminal 1: API
.local/runtime/api/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir api --reload

# Terminal 2: interfaz
npm run dev --prefix web
```

Consulte [api/README.md](api/README.md) y [web/README.md](web/README.md) para
más detalles.

## Pruebas

```powershell
.local/runtime/api/.venv/Scripts/python.exe -m pytest api/tests -q --basetemp .local/runtime/api/pytest-tmp-direct -o cache_dir=.local/runtime/api/pytest-cache
npm run typecheck --prefix web
npm run build --prefix web
```

## Workspace de ML

[ml/README.md](ml/README.md) conserva el historial, código de entrenamiento y
evidencias de Fase 5. Los datasets completos (`TBX11K`, `Montgomery`, `Shenzhen`)
permanecen fuera de Git.

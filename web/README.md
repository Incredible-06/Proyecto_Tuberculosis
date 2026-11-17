# Aplicación web

Interfaz React 19 + Vite 7 + TypeScript + Tailwind CSS 4 para el sistema de
apoyo al cribado radiográfico de tuberculosis. Consume `/api/v1/*` mediante
cookie `HttpOnly`, sin guardar credenciales ni tokens en `localStorage`.

## Desarrollo

```powershell
npm install --prefix web
npm run dev --prefix web
```

Vite queda disponible en `http://127.0.0.1:5173/` y redirige `/api` a
`http://127.0.0.1:8000`.

## Compilación integrada

```powershell
npm run typecheck --prefix web
npm run build --prefix web
```

`web/dist` es servido por FastAPI en `http://127.0.0.1:8000/` con fallback SPA.

## Rutas y componentes principales

- `/login` (`LoginPage.tsx`): acceso institucional con validación accesible.
- `/` (`DashboardPage.tsx`): resumen operativo por rol, cola de pendientes y estado del motor de análisis.
- `/evaluaciones` (`EvaluationsPage.tsx`): registro anonimizado, búsqueda en servidor por código/estado/prioridad y listado clínico.
- `/evaluaciones/:id` (`EvaluationDetailPage.tsx`):
  - `RadiographViewer.tsx`: visor con zoom (`100%–300%`), paneo, cajas SVG `YOLO11n` y superposición de mapas `Grad-CAM` / `Attention Rollout` con opacidad ajustable.
  - `PredictionPanel.tsx`: concordancia en 4 estados, probabilidades con corte `p_TB >= 0.30`, alerta de confianza baja, advertencia clínica/dominio `TBX11K` y trazabilidad técnica colapsable.
  - `ObservationsPanel.tsx`: observaciones médicas anti-PII y cierre a `revisada`.
- `/usuarios` (`UsersPage.tsx`): administración de cuentas, roles y diálogo nativo de confirmación.

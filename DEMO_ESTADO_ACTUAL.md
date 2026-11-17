# Demo y estado actual del sistema TB

Corte: 16 de noviembre de 2026.

Este documento describe el flujo completo disponible para demostración, las
cuentas recomendadas por rol y los límites operativos vigentes.

El sistema es una herramienta de apoyo a la decisión clínica. No emite
diagnóstico médico definitivo, no reemplaza el criterio clínico ni las pruebas
confirmatorias (baciloscopía, GeneXpert o cultivo) y no debe usarse con datos
identificables de pacientes reales.

## 1. Qué se puede hacer hoy

- Iniciar y cerrar sesión con cookie `HttpOnly` y bloqueo por intentos fallidos.
- Administrar cuentas y aplicar en el servidor la matriz de permisos de los 5 roles.
- Registrar evaluaciones con código interno anonimizado (`TB-AAAAMMDD-XXXXXX`).
- Buscar evaluaciones en servidor por código interno, estado y prioridad clínica (`HU08`).
- Cargar y reemplazar (antes de validar calidad) radiografías PNG o JPG (`HU06`).
- Validar resolución mínima (`224 px`), decodificación y legibilidad (`HU07`).
- Ejecutar el motor congelado de Fase 5 (`ml/models/frozen_phase5/`) con verificación
  criptográfica SHA-256 (`HU09`, `HU22`):
  - Clasificador `ViT-B/16` (`health` / `sick` / `TB` y alerta operativa `p_TB >= 0.30`).
  - Detector `YOLO11n` (`conf >= 0.25`, `imgsz=512`, `iou=0.7`) proyectado al espacio de la imagen original.
  - Regla de concordancia aditiva en 4 estados sin supresión (`both_alert`, `classifier_only_review`, `detector_only_review`, `neither_alert`).
  - Alerta de baja confianza / caso limítrofe (`HU12`).
  - Mapas visuales superpuestos con control de opacidad y zoom (`Grad-CAM` clase `TB` y `Attention Rollout` de `ViT-B/16`).
- Mostrar advertencia clínica explícita y limitación de dominio `TBX11K` (efectos de *domain shift* observados en Montgomery/Shenzhen).
- Registrar observaciones clínicas médicas con validación anti-PII y cerrar la evaluación en estado `revisada` (`HU15`).

## 2. Estado de historias de usuario

| Historia | Estado | Qué se puede demostrar |
|---|---|---|
| HU01 | Implementada | Login, logout, sesión revocable, bloqueo por intentos. |
| HU02 | Implementada | Alta de cuentas por administrador; sin registro público. |
| HU03 | Implementada | Cinco roles, permisos aplicados en el servidor. |
| HU04 | No implementada | Sin recuperación de contraseña desde interfaz. |
| HU05 | Implementada | Creación anonimizada con código `TB-AAAAMMDD-XXXXXX`. |
| HU06 | Implementada | Carga y reemplazo controlado de PNG/JPG; rechazo DICOM/corruptos. |
| HU07 | Implementada | Validación de decodificación, resolución mínima (`224 px`), contraste y legibilidad. |
| HU08 | Implementada | Búsqueda en servidor por código, estado y filtro prioritario. |
| HU09–HU14 | Implementada | Motor congelado Fase 5: inferencia, probabilidades, advertencias, concordancia, cajas `YOLO11n` y capas `Grad-CAM` / `Rollout`. |
| HU15 | Implementada | Registro de observaciones clínicas médicas y transición a `revisada`. |
| HU22 | Implementada | Registro automático y trazabilidad de versión/hashes del modelo en cada predicción. |
| HU16–HU21, HU23–HU24 | Fuera de alcance demo | Reportes PDF, configuración dinámica de umbrales, visor de auditoría y respaldos automáticos. |

Flujo completo de estados:
`pendiente_imagen` → `imagen_cargada` → `lista_para_analisis` → `analizada` → `revisada`.

## 3. Cómo arrancar

Desde la raíz del repositorio:

```powershell
.\run.ps1
```

Si PowerShell bloquea la ejecución:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

El script prepara `.local/runtime/api/.venv` (incluyendo PyTorch CPU y
Ultralytics si faltan), construye `web/dist` si hace falta, solicita crear
`admin.local` en el primer arranque y levanta el servidor en:

```text
http://127.0.0.1:8000/
```

## 4. Cuentas sugeridas para la demostración

Desde `admin.local` cree estas cuentas en **Usuarios** (contraseña de 12+ caracteres, por ejemplo `DemoTB-2026-Segura!` solo para demo local):

| Cuenta | Nombre visible | Rol |
|---|---|---|
| `admin.local` | Administrador inicial | Administrador del sistema |
| `tecnologo.rayos` | Tecnóloga de radiología | Tecnólogo médico en radiología |
| `medico.radiologo` | Médico radiólogo | Médico radiólogo |
| `medico.programa` | Médico del programa TB | Médico del programa de TB |
| `coordinador.tb` | Coordinador TB | Coordinador de la Estrategia Sanitaria de TB |

## 5. Guion corto de demostración

1. **Administrador (`admin.local`)**:
   - Verifique en **Inicio** el estado activo del motor (`ViT-B/16` + `YOLO11n`, hash SHA-256 verificado).
   - En **Usuarios**, muestre la gestión de roles, alta de cuentas y el diálogo modal de confirmación al desactivar.
2. **Tecnólogo (`tecnologo.rayos`)**:
   - En **Estudios de tórax**, pulse **Nueva evaluación** (`pendiente_imagen`).
   - Cargue una radiografía PNG/JPG (`imagen_cargada`). Puede probar subir una imagen muy pequeña (`< 224 px`) o plana para demostrar el rechazo en **Verificar calidad** y luego **Reemplazar imagen**.
   - Con una radiografía válida, pulse **Verificar calidad** (`lista_para_analisis`).
   - Pulse **Ejecutar análisis** (`analizada`; en CPU toma ~25–30 s). Verifique que el tecnólogo no accede a probabilidades, cajas ni mapas clínicos.
3. **Médico (`medico.radiologo` o `medico.programa`)**:
   - En **Inicio** o **Estudios de tórax**, ubique el caso (pruebe el filtro **Solo revisión prioritaria** o la búsqueda por código).
   - Muestre la advertencia de apoyo clínico y limitación de dominio `TBX11K`, la concordancia operativa en 4 estados, probabilidades con corte `p_TB >= 30%`, cajas `YOLO11n`, selector **Original / Grad-CAM / Atención ViT**, opacidad y zoom.
   - Registre una impresión médica en **Evaluación médica y observaciones** dejando marcada la casilla para pasar a **`Revisada`**.
4. **Coordinador (`coordinador.tb`)**:
   - Verifique que puede consultar listado, predicción, mapas y observaciones en modo lectura, sin permiso para crear evaluaciones ni registrar observaciones.

## 6. Validación técnica verificada

- **43 pruebas automatizadas de API aprobadas** (`pytest api/tests`), incluida ejecución real de inferencia con `ml/models/frozen_phase5/`.
- **`npm run typecheck --prefix web` y `npm run build --prefix web` aprobados**.
- **Prueba HTTP E2E real con Uvicorn aprobada**: flujo completo de bootstrap, alta por roles, carga, calidad, inferencia congelada, descarga de mapas con `Cache-Control: private, no-store`, búsqueda y cierre clínico en `revisada`.
- **Integridad ML**: los 17 hashes de procedencia (`code_hashes` en `phase5_frozen_contract.json`) permanecen intactos.

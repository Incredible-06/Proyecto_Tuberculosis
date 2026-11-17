# Backlog y arquitectura de producto

Actualizado el 9 de noviembre de 2026. Este documento traduce el backlog
priorizado a una secuencia técnica verificable; no sustituye el refinamiento ni
la aceptación del Product Owner antes de iniciar cada historia.

## Fuente vigente

- Fuente archivada localmente, fuera de Git:
  `.local/backlog_sources/LAB05 - Backlogs Priorizados-20261110T030733Z-1-001.zip`.
- Se toma EP01 versión 2.1 y EP02–EP07 versión 2.0 como versión vigente.
- Los cierres de sprint que figuran en los documentos son planificación
  histórica; el orden de dependencias y las prioridades siguen siendo útiles.
- Los documentos priorizan y estiman las historias, pero indican expresamente
  que los criterios de aceptación se definen durante el refinamiento.

## Roles y límites

Los cinco roles son administrador del sistema, tecnólogo médico en radiología,
médico del programa de TB, médico radiólogo y coordinador de la Estrategia
Sanitaria de TB. Las cuentas las crea el administrador; no hay registro público
ni inicio de sesión institucional dentro del alcance actual. Cada evaluación
usa un código interno anonimizado: no se deben guardar identificadores del
paciente. PNG y JPG están admitidos; DICOM está fuera de alcance.

El producto es apoyo a la decisión, no diagnóstico. El clasificador provisional
no se presenta como definitivo, y test interno, Shenzhen y Montgomery siguen
reservados hasta documentar el congelamiento de Fase 5.

## Mapa priorizado

| Épica | Historias en orden | Prioridad |
|---|---|---|
| EP01 · Acceso y usuarios | HU01 iniciar sesión; HU02 registrar usuario; HU03 roles y permisos; HU04 recuperar contraseña | Must, Must, Must, Should |
| EP02 · Evaluaciones y radiografías | HU05 registrar evaluación; HU06 cargar radiografía; HU07 validar imagen; HU08 buscar por código | Must, Must, Must, Should |
| EP03 · Análisis Deep Learning | HU09 ejecutar modelo; HU10 categoría/probabilidades; HU11 confianza/advertencia; HU12 alerta baja confianza | Must, Must, Must, Should |
| EP04 · Interpretación visual | HU13 zonas sospechosas; HU14 Grad-CAM; HU15 observaciones | Must, Should, Could |
| EP05 · Historial | HU16 historial; HU18 detalle; HU17 filtros | Must, Must, Should |
| EP06 · Reportes | HU19 generar PDF; HU20 descargar reporte | Must, Should |
| EP07 · Administración | HU22 versión de modelo; HU23 auditoría; HU21 parámetros; HU24 respaldos | Must, Should, Should, Could |

## Secuencia técnica

1. **Fundación:** HU01–HU02, esquema de usuarios, migraciones, sesión segura,
   administrador inicial configurado fuera del repositorio y pruebas de API.
2. **Flujo de entrada:** HU03, HU05–HU07 y HU06; control de roles, evaluación
   anonimizada, carga segura y validación de PNG/JPG.
3. **Contrato de inferencia:** HU09–HU11, HU13–HU14 y HU22; FastAPI, errores,
   versión de modelo y respuestas simuladas mientras Fase 5 permanezca abierta.
4. **Consulta y operación:** HU08, HU12, HU15–HU21, HU23–HU24, según las
   dependencias de las historias y la capacidad disponible.

La base de dominio prevista separa `User`/`Role`, `Evaluation`/`Radiograph`,
`Prediction`/`ModelVersion`, `Detection`, `Observation`, `Report` y
`AuditEvent`. Las imágenes y reportes se almacenarán fuera del repositorio; la
base de datos solo conserva metadatos, rutas controladas y trazabilidad.

## Refinamiento inicial: HU01 y HU02

Antes de implementarlas, el Product Owner debe validar como mínimo los campos
de una cuenta, política de contraseña, caducidad de sesión y mecanismo de
bloqueo por intentos fallidos. Como propuesta verificable:

- **HU01:** una cuenta activa con credenciales válidas inicia sesión; una
  credencial inválida recibe un mensaje genérico; ningún secreto se devuelve ni
  se registra; rutas protegidas rechazan sesiones ausentes o vencidas; los
  intentos fallidos se limitan y quedan auditables.
- **HU02:** solo el administrador inicial puede crear cuentas; el identificador
  de acceso es único; la contraseña se almacena exclusivamente como hash; el
  rol inicial pertenece a los cinco roles aprobados; duplicados y datos
  incompletos se rechazan sin crear una cuenta parcial.

La matriz completa de permisos se implementa en HU03. Hasta entonces, el
privilegio de alta queda limitado al administrador inicial para cumplir HU02
sin adelantar la matriz transversal.

## Estructura local

`.local/` está ignorado por Git y agrupa `data/` (TBX11K, fuentes externas y
ZIP), `generated/` (dist y resultados), `work/` (temporales), `runtime/`
(base de datos, cargas, reportes y entorno de pruebas) y `backlog_sources/`.
El código y la evidencia de entrenamiento viven en `ml/`; allí se mantienen
los enlaces locales de compatibilidad a los datos y las salidas históricas. La
raíz conserva solo el sistema web/API y sus documentos de producto. No se
eliminó ningún dato durante esta reorganización.

# Criterios de aceptación — HU03, HU05 y HU06

Actualizado el 9 de noviembre de 2026. Este documento refina el segundo bloque
de implementación del sistema web. Complementa
`docs/CRITERIOS_ACEPTACION_HU01_HU02.md` y el backlog priorizado; no sustituye
la aceptación del Product Owner.

## Decisiones de producto de este bloque

- La autorización se aplica siempre en el servidor. La interfaz puede ocultar
  acciones, pero nunca es la única barrera.
- El principio por defecto es de mínimo privilegio: el administrador gestiona
  identidad, roles, parámetros y auditoría; no recibe acceso clínico general.
- Los cinco roles aprobados son los únicos válidos. No se crean roles nuevos en
  este bloque.
- Una evaluación se identifica solo por un código interno generado por el
  sistema. No se solicitan ni almacenan nombres, documentos, historia clínica,
  fechas de nacimiento ni contacto del paciente.
- Una evaluación tiene como máximo una radiografía en este bloque. La imagen
  original no se versiona en Git y se guarda bajo `.local/runtime/api/`.
- No se ejecuta inferencia ni se muestran resultados clínicos. HU07, HU09 y las
  historias posteriores permanecen fuera de este bloque.
- No se usa test interno, Shenzhen, Montgomery ni ningún artefacto de Fase 5
  para construir o probar estas historias.

## Matriz de permisos de HU03

Los permisos se nombran con la forma `recurso.acción` y se evalúan contra el rol
persistido de la cuenta activa.

| Permiso | Administrador | Tecnólogo médico en radiología | Médico del programa de TB | Médico radiólogo | Coordinador de la Estrategia Sanitaria de TB |
|---|:---:|:---:|:---:|:---:|:---:|
| `users.read` | Sí | No | No | No | No |
| `users.manage` | Sí | No | No | No | No |
| `roles.read` | Sí | No | No | No | No |
| `evaluations.create` | No | Sí | No | No | No |
| `evaluations.read` | No | Sí | Sí | Sí | Sí |
| `radiographs.upload` | No | Sí | No | No | No |
| `radiographs.read` | No | Sí | Sí | Sí | Sí |
| `predictions.read` | No | No | Sí | Sí | Sí |
| `observations.read` | No | No | Sí | Sí | Sí |
| `observations.write` | No | No | Sí | Sí | No |
| `reports.read` | No | No | Sí | Sí | Sí |
| `reports.generate` | No | No | Sí | Sí | Sí |
| `settings.manage` | Sí | No | No | No | No |
| `audit.read` | Sí | No | No | No | No |
| `model_versions.manage` | Sí | No | No | No | No |

Los permisos de resultados, observaciones y reportes quedan definidos para las
historias posteriores, pero todavía no habilitan rutas clínicas de EP03–EP06.

## HU03 — Asignar roles y permisos

**Historia:** Como administrador del sistema, quiero asignar el rol y el estado
de cada cuenta para que cada integrante acceda únicamente a las funciones que le
corresponden.

### Criterios de aceptación

1. Toda ruta protegida exige una sesión activa y un permiso concreto. Una
   solicitud sin sesión recibe `401`; una sesión válida sin el permiso recibe
   `403`, sin revelar detalles internos.
2. La matriz anterior se aplica a los cinco roles y se expone al frontend en los
   datos públicos de la sesión para construir la navegación.
3. Solo una cuenta con `users.manage` puede listar cuentas, cambiar el rol o
   activar/desactivar una cuenta. Las demás cuentas reciben `403`.
4. El cambio de rol acepta únicamente uno de los cinco roles aprobados y deja
   un evento de auditoría `user.updated` con los campos modificados, sin
   secretos.
5. No se puede desactivar la última cuenta administradora activa ni quitarle el
   rol administrador. Tampoco se puede desactivar la propia cuenta desde la API.
   Estas reglas evitan dejar el sistema sin administración.
6. El catálogo de roles y permisos se puede consultar con `GET /api/v1/roles`
   usando `roles.read`; el frontend no duplica la fuente de verdad de permisos.
7. La desactivación de una cuenta invalida sus sesiones en la siguiente
   solicitud, porque el rol y el estado se leen de la base en cada petición.
8. Los cambios de rol no alteran la contraseña ni crean cuentas nuevas.

### Pruebas de aceptación

- Matriz exacta para los cinco roles, incluyendo rutas permitidas y denegadas.
- Alta y cambio de rol por administrador con evento de auditoría.
- Rechazo de cambio de rol o desactivación sin `users.manage`.
- Rechazo de desactivar la última administradora y de autodesactivarse.
- Invalidación efectiva de una sesión tras desactivar la cuenta.

## HU05 — Registrar evaluación con código interno

**Historia:** Como tecnólogo médico en radiología, quiero registrar una
evaluación con un código interno anonimizado para asociarle después una
radiografía sin registrar datos personales del paciente.

### Criterios de aceptación

1. Solo una sesión con `evaluations.create` puede crear una evaluación. En este
   bloque ese permiso pertenece al tecnólogo médico en radiología.
2. El sistema genera el código interno con el formato `TB-AAAAMMDD-XXXXXX`, con
   sufijo aleatorio y unicidad verificada. El usuario no puede imponer ni editar
   el código.
3. El formulario solo admite una nota interna opcional de hasta 200 caracteres.
   Se rechazan correos y secuencias de ocho o más dígitos para reducir el riesgo
   de datos personales. La interfaz advierte que no se deben ingresar datos del
   paciente.
4. La evaluación nace con estado `pendiente_imagen`, cuenta creadora, fechas de
   creación y actualización, y un evento `evaluation.created` sin nota ni datos
   personales.
5. `GET /api/v1/evaluations` y `GET /api/v1/evaluations/{id}` requieren
   `evaluations.read`; las cuentas sin permiso reciben `403`. La lista se puede
   filtrar por estado y por código interno para preparar HU08.
6. La respuesta incluye el código, estado, nota, metadatos de la radiografía si
   existe y la identidad visible de la cuenta creadora. No incluye rutas físicas
   ni secretos.
7. La creación es atómica: si falla la auditoría o la persistencia, no queda una
   evaluación parcial.

### Pruebas de aceptación

- Creación válida por tecnólogo con código generado y auditoría.
- Rechazo de creación por administrador, médicos y coordinador.
- Rechazo de notas con correo o documento de ocho dígitos.
- Lectura permitida para los cinco roles clínicos definidos en la matriz y
  denegada para el administrador.
- Búsqueda y filtro por código/estado sin exponer rutas internas.

## HU06 — Cargar radiografía de tórax

**Historia:** Como tecnólogo médico en radiología, quiero cargar una radiografía
PNG o JPG asociada a una evaluación para que quede disponible de forma segura
para las etapas siguientes.

### Criterios de aceptación

1. Solo una sesión con `radiographs.upload` puede cargar la imagen. La
   evaluación debe existir y no tener ya una radiografía; un segundo intento
   recibe `409`.
2. Se admiten únicamente PNG y JPG/JPEG. El tipo declarado, la extensión y la
   firma binaria del archivo deben coincidir. Cualquier otro formato, incluido
   DICOM, recibe `415`.
3. El tamaño máximo por archivo es de 10 MB, configurable por entorno. Un
   archivo mayor recibe `413` sin persistirse.
4. Un archivo truncado, sin dimensiones válidas o con firma incoherente recibe
   `422` y no deja archivos ni filas parciales.
5. La imagen se guarda con un nombre aleatorio bajo
   `.local/runtime/api/radiographs/{evaluation_id}/`, fuera de Git. No se
   conserva el nombre original del archivo para evitar arrastrar datos
   personales. La base solo guarda metadatos y una ruta relativa controlada.
6. La carga registra tamaño, tipo, dimensiones, SHA-256, fecha, cuenta que carga
   y evento `radiograph.uploaded`; no registra el contenido ni el nombre
   original.
7. La evaluación pasa a estado `imagen_cargada` y actualiza su fecha. La
   descarga o vista previa exige `radiographs.read`; sin permiso responde `403`.
8. La respuesta y la interfaz muestran una vista previa autenticada, el estado y
   los metadatos, sin inferencia clínica. La validación de calidad de HU07 queda
   pendiente y se informa como tal.
9. Si la escritura del archivo o la transacción fallan, se limpia el archivo o
   la fila parcial antes de responder.

### Pruebas de aceptación

- Carga PNG y JPG válidos con metadatos, archivo en almacenamiento temporal y
  auditoría.
- Rechazo de texto, extensión/tipo incoherente, DICOM, archivo truncado y
  archivo sobre el límite, sin residuos.
- Rechazo de segunda carga sobre la misma evaluación.
- Descarga permitida a roles con `radiographs.read` y denegada a otros.
- Verificación de que la base y la auditoría no guardan el nombre original ni
  el contenido de la imagen.

## Límites explícitos

- HU07 definirá resolución mínima y criterios de legibilidad con tecnólogo y
  radiólogo; en este bloque solo se registran dimensiones.
- HU09–HU14 no se implementan: no hay modelo, probabilidades, Grad-CAM ni
  resultados clínicos.
- La interfaz es un prototipo de apoyo a la decisión. No emite diagnóstico ni
  sustituye el criterio profesional.
- Test interno, Shenzhen y Montgomery siguen reservados hasta el congelamiento
  documentado de Fase 5.

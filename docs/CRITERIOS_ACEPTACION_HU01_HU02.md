# Criterios de aceptación — HU01 y HU02

Actualizado el 9 de noviembre de 2026. Este documento completa el refinamiento
del backlog vigente para el primer bloque de implementación. Los criterios se
mantienen junto al backlog porque definen la condición objetiva de aceptación
de cada historia, sus pruebas y sus límites.

## Decisiones de producto

- El sistema no ofrece registro público: el administrador del sistema crea las
  cuentas del personal autorizado.
- Una cuenta contiene identificador de acceso, nombre visible, rol y secreto.
  No se requiere correo electrónico ni se almacenan datos del paciente.
- Los roles válidos son `administrador`, `tecnologo_radiologia`,
  `medico_programa_tb`, `medico_radiologo` y `coordinador_estrategia_tb`.
- La primera cuenta administrativa se crea mediante un comando local con un
  secreto proporcionado fuera del repositorio. No existen credenciales por
  defecto en código, documentación ni datos de prueba.
- La sesión es opaca, revocable y de ocho horas. Se transporta por cookie
  `HttpOnly` y `SameSite=Strict`; producción exige TLS y cookie `Secure`.
- Las contraseñas se almacenan con `scrypt`, un salt aleatorio y parámetros
  versionados. Nunca se registran, devuelven ni guardan en texto plano.

## HU01 — Iniciar sesión

**Historia:** Como integrante autorizado del sistema, quiero iniciar sesión con
mis credenciales para acceder únicamente a las funciones que me correspondan.

### Criterios de aceptación

1. Dada una cuenta activa y credenciales válidas, al iniciar sesión el sistema
   crea una sesión revocable con vencimiento de ocho horas y devuelve los datos
   públicos de la cuenta, sin contraseña ni token en el cuerpo de la respuesta.
2. Dadas credenciales inválidas, una cuenta inexistente o una cuenta inactiva,
   el sistema responde con el mismo mensaje genérico y no revela cuál dato falló.
3. Tras cinco intentos fallidos consecutivos, la cuenta queda bloqueada durante
   quince minutos. Mientras dure el bloqueo, incluso la contraseña correcta se
   rechaza; un inicio exitoso posterior reinicia el contador.
4. Las rutas protegidas rechazan solicitudes sin sesión, con sesión vencida,
   revocada o de una cuenta inactiva.
5. Cerrar sesión invalida la sesión del servidor y elimina la cookie del cliente;
   esa misma sesión no puede volver a utilizarse.
6. Cada inicio exitoso, fallo, bloqueo y cierre queda registrado como evento de
   auditoría sin incluir contraseñas, tokens ni datos clínicos.
7. La contraseña de entrada admite entre 1 y 128 caracteres para no alterar su
   significado al verificarla; la política fuerte se exige al crear la cuenta.

### Pruebas de aceptación

- Inicio válido seguido de consulta de identidad autenticada.
- Error indistinguible para usuario inexistente y contraseña incorrecta.
- Bloqueo al quinto fallo, rechazo durante la ventana y restauración tras ella.
- Rechazo de rutas protegidas sin sesión y después del cierre de sesión.
- Inspección de auditoría y de base de datos que confirme ausencia de secretos.

## HU02 — Registrar usuario

**Historia:** Como administrador del sistema, quiero registrar una cuenta del
personal autorizado para que pueda acceder al sistema según su rol inicial.

### Criterios de aceptación

1. Solo una sesión activa con rol `administrador` puede crear una cuenta; una
   solicitud sin sesión recibe `401` y una sesión de otro rol recibe `403`.
2. El formulario exige identificador de acceso único, nombre visible no vacío,
   rol permitido y contraseña de 12 a 128 caracteres. El identificador acepta
   letras, números, punto, guion y guion bajo, y se compara sin distinguir
   mayúsculas de minúsculas.
3. Una contraseña no puede estar vacía, ser menor de 12 caracteres, superar 128
   caracteres, consistir solo en espacios ni coincidir con el identificador de
   acceso. Se permiten frases de paso sin imponer reglas de composición frágiles.
4. Ante un identificador duplicado, el sistema responde `409` y no crea ningún
   registro parcial. Ante un dato inválido, responde `422` con el campo que debe
   corregirse y tampoco crea registros parciales.
5. La nueva cuenta queda activa por defecto, conserva exactamente uno de los
   cinco roles aprobados y devuelve únicamente datos públicos de la cuenta.
6. La contraseña se persiste exclusivamente como hash `scrypt` con salt; una
   inspección directa de la base de datos no puede recuperar el secreto original.
7. El alta genera un evento `user.created` con el administrador actor y la nueva
   cuenta como sujeto, sin incluir contraseña ni token.

### Pruebas de aceptación

- Alta correcta realizada por administrador y posterior inicio de sesión de la
  cuenta creada.
- Rechazo de alta para solicitudes anónimas y para los otros cuatro roles.
- Rechazo de duplicados que solo difieren por mayúsculas/minúsculas.
- Rechazo de cada regla de validación sin filas residuales.
- Verificación del hash, el rol inicial y el evento de auditoría.

## Límites de este bloque

HU03 completará la matriz de permisos sobre evaluaciones, imágenes, resultados,
reportes y configuración. HU01–HU02 solo establecen la restricción mínima de
que el administrador inicial crea cuentas. La recuperación de contraseña se
mantiene en HU04 y no se simula en este bloque.

La autenticación se implementa sin usar test interno, datos externos, pesos ni
resultados clínicos de Fase 5.

# Fase 3: primer clasificador y ciclo de experimentación

Estado al 26 de octubre de 2026: pipeline y notebook preparados para Colab; no hay
una corrida de entrenamiento Colab verificada en esta entrega de código. No se
han generado métricas ni pesos de TBX11K durante su preparación. La fase queda
abierta. Los resultados de las fases anteriores no se reemplazan.

## Qué ya se comprobó

La revisión local del 21 de octubre contiene notebooks 00/01 ejecutados sin
errores, particiones congeladas y 12.278 hashes comprobados. En la consulta a
Drive del 26 de octubre se encontraron el paquete de Fases 1/2, TBX11K.zip y las
carpetas de resultados correspondientes. `experiments/` estaba vacía.

Las clases vigentes son **health, sick y TB**. El plan de la imagen menciona
cuatro clases; el pipeline existente agrupa los subtipos TB en una sola clase y
conserva sus metadatos. Cambiar a cuatro salidas requiere resolver esa definición
y revisar las etiquetas: no se cambia silenciosamente en este avance.

## Alcance de la primera corrida

El notebook incluye **ambos recorridos**, en celdas separadas:

- **Smoke:** dos imágenes por clase de train y dos de validation (6 + 6), mismo
  tamaño 512 px y modelo real. Comprueba GPU, AMP, gradientes, actualización de
  pesos y escritura/recarga del checkpoint. Sus carpetas llevan `smoke_`, no
  cuentan como baseline y no aportan métricas para presentar calidad.
- **Pipeline del modelo base:** usa todo train y validation, entrena durante el
  presupuesto configurado, selecciona y guarda el mejor modelo, permite reanudar,
  inspeccionar errores y comenzar un ajuste posterior desde ese modelo.

El baseline vuelve a los pesos ImageNet: no hereda pesos de la prueba smoke.
La limitación inicial de tres épocas es un presupuesto preliminar configurable,
no una limitación del pipeline a unos pocos lotes. No hay un corte oculto de datos.

Se implementa **ConvNeXt-Tiny con pesos ImageNet IMAGENET1K_V1**, que ya estaba
previsto como línea base de rasgos locales en el plan. Es la referencia necesaria
para comparar más adelante con Vision Mamba y el híbrido; su calidad sobre TBX11K
todavía debe medirse. No se elige por resultados que aún no existen.

Se conserva el protocolo de datos: RGB 512×512 con normalización ImageNet,
letterbox, brillo/contraste ±8 % y rotación ±5° solo en train; sin CLAHE ni flips.
La resolución y conservación del campo difieren del preset resize/crop 224 de
los pesos oficiales. Esta adaptación queda explícita en cada experimento.

| Parámetro inicial | Valor |
|---|---|
| Train / validation | 5.792 / 1.241 imágenes, sin crear otras particiones |
| Presupuesto | 3 épocas completas, terminables antes por early stopping |
| Primera época | Cabeza entrenable; extractor congelado en modo evaluación |
| Épocas 2 y 3 | Cabeza y última etapa del extractor entrenables |
| Optimizador | AdamW, weight decay 0,01, sin scheduler en esta corrida corta |
| Learning rate | Cabeza 1e-3; última etapa 1e-5 |
| Desbalance | Cross entropy ponderada; pesos calculados solo en train |
| Memoria | Batch 8, acumulación 4, AMP, clipping 1,0 |
| Selección | Macro-F1 de validation, mejora mínima 0,001 |
| Early stopping | Paciencia 3, guardado del mejor estado |
| GPU | La GPU CUDA disponible en Colab, incluida L4; no requiere A100 |

El objetivo de tres épocas es una referencia preliminar revisable. No basta
para afirmar convergencia. Con este presupuesto, normalmente se agotarán las
tres épocas antes de que la paciencia detenga la corrida.

## Cómo ejecutarlo

1. En Drive, conserve `Proyecto_Tuberculosis/TBX11K.zip` donde ya está.
2. Guarde allí el nuevo `tbx11k_fase3_baseline_colab.zip`, separado del ZIP anterior.
3. Abra `02_convnext_tiny_baseline.ipynb` con Google Colaboratory y guárdelo en
   Drive. Seleccione GPU L4/T4/A100 disponible. No cambie ni reinstale CUDA.
4. Ejecute las celdas en orden. El montaje de Drive requiere autorización en su
   sesión. La verificación de hashes se repite al copiar el dataset al runtime.
5. Ejecute **F3-SMOKE** para validar el recorrido técnico. Si pasa, ejecute
   **F3-E3** para entrenar el modelo base completo. La prueba smoke por sí sola
   no demuestra la calidad de un modelo preliminar.
6. Revise las curvas y errores, complete la bitácora y guarde el notebook con
   **Ctrl+S** para conservar también las salidas visibles en Colab.

Si falta memoria, cree una nueva corrida con batch 4 y acumulación 8 y registre
el motivo. No altere datos o resolución para resolver un OOM. La duración depende
del dispositivo asignado y del estado del runtime; no se promete un tiempo fijo.

## Qué se guarda y cuándo

```text
MyDrive/Proyecto_Tuberculosis/
  02_convnext_tiny_baseline.ipynb       # guardar las salidas mediante Colab
  tbx11k_fase3_baseline_colab.zip
  experiments/<run_id>/
    config.json, data_config.json, environment.json, provenance.json
    train.csv, validation.csv, class_weights.json
    initial_validation.json            # referencia previa a actualizar pesos
    status.json                       # running / interrupted / failed / completed_preliminary
    history.csv                       # métricas por época completa
    best_validation.json
    validation_predictions.csv        # al completar/exportar la corrida
    figures/                          # curvas, matriz, ROC; errores al inspeccionarlos
    review.json                       # al completar la bitácora humana
    summary.json                      # solo después de completar/exportar
    notebook_source_unexecuted.ipynb   # fuente, sin fingir salidas ejecutadas
    transfer_integrity.json            # después de retornar la celda de entrenamiento
  checkpoints/<run_id>/
    last.pt                           # optimizador, AMP, RNG, historial y mejor estado
    best.pt                           # modelo para inspección/inferencia posterior
```

`last.pt` se guarda al terminar cada época y contiene también el mejor estado
para reconstruir `best.pt` si una publicación se interrumpe. Los checkpoints se
serializan temporalmente en Colab y se copian a Drive antes de publicar su nombre.
Pueden ocupar varios cientos de MB; quedan fuera de Git y del paquete pequeño.

Para reanudar, copie el nombre de la carpeta a `RESUME_RUN` y ejecute con la misma
configuración y código. Se repite la época parcial. Cambiar épocas, learning rate
o batch es un experimento nuevo. Si solo falló la exportación final, reanudar
reconstruye los resultados desde `last.pt` sin repetir las épocas completas.
Use exclusivamente checkpoints propios de esta carpeta.

## Mostrar un ciclo real sin completar toda la fase

1. **Hipótesis v01:** adaptar cabeza y última etapa con el protocolo congelado.
2. **Entrenamiento:** completar la primera corrida y guardar sus checkpoints.
3. **Revisión:** contrastar pérdida, macro-F1 y sensibilidad TB, observar la matriz
   y algunos errores de validation, y registrar la conclusión real en `review.json`.
4. **Ajuste v02:** formular el siguiente cambio. El notebook trae una segunda
   corrida opcional desactivada; puede quedar pendiente para la próxima sesión.

Una corrida contiene varias épocas. Tres épocas **no son tres modelos comparados**.
El bloque opcional v02 continúa desde el mejor modelo, con optimizador nuevo y
tasas menores: se presenta como ajuste exploratorio. No es una ablación aislada,
porque cambian tasas y presupuesto y parte de pesos ya adaptados.

Cada nueva corrida tiene ID independiente y referencia a su corrida de origen.
La tabla compara resultados que realmente existen; no crea filas ni números de
experimentos pendientes. Para comparar arquitecturas después harán falta el
mismo protocolo, presupuesto y selección, más repeticiones y control de semillas.

## Capturas útiles para esta entrega parcial

| Código del notebook | Evidencia |
|---|---|
| F3-E1 | GPU y versiones de torch/torchvision |
| F3-E2 | Particiones y verificación de integridad |
| F3-SMOKE | Forward/backward, actualización y recarga con pocas muestras; prueba técnica |
| F3-E3 | Épocas entrenadas y confirmación de guardado en Drive |
| F3-E4 | Curvas, matriz y métricas de validation |
| F3-E5 | Errores inspeccionados y decisión registrada |
| F3-E6 | Tabla de corridas y alcance pendiente |

Después de ejecutar, la última celda redacta una descripción del avance usando
el número real de épocas y el macro-F1 obtenido. Antes de ejecutar, describa el
estado como «pipeline de entrenamiento implementado y listo para ejecutar».
Solo con actualizaciones guardadas es correcto decir «entrenamiento preliminar
iniciado». No afirme que la Fase 3 está terminada.

## Límites que se conservan

No se evalúa el test interno, el oficial ni fuentes externas. La selección usa
validation y por ello sus métricas no son evaluación final. Sensibilidad TB y
especificidad usan argmax multiclase; no se optimiza un umbral con test. Las
métricas iniciales son por imagen, sin intervalos de confianza ni independencia
garantizada por paciente. Los aumentos no cambian las etiquetas ni las cajas.

Quedan pendientes Vision Mamba, híbrido con atención cruzada, comparación completa,
repeticiones, ajustes adicionales justificados, incertidumbre y evaluación final.
Tampoco se desarrolla en este avance el detector, la API o la interfaz web.

## Reproducir la verificación del código

```powershell
python scripts/create_training_notebook.py
python scripts/check_training_notebook.py
python -m unittest discover -s tests -p test_training_protocol.py -v
python scripts/check_training_tests.py
python scripts/package_training_colab.py
```

La validación estática y las pruebas de fixtures no equivalen a una ejecución
Colab. Los registros de verificación distinguen esas dos cosas.

La prueba opcional `tests/test_training_torch_fixture.py` comprueba en CPU
acumulación ponderada exacta (incluido el último grupo parcial), forward/backward
de ConvNeXt real con datos sintéticos de 32 px y recarga del checkpoint. No
descarga pesos ni usa TBX11K. La resolución 512 y AMP/CUDA se verifican en la
celda smoke real de Colab, no en esa prueba sintética.
Si el intérprete local no tiene torch/torchvision o Matplotlib, esas pruebas
opcionales quedan marcadas como omitidas; el registro informa sus conteos.

Fuentes: [pesos y arquitectura oficiales de TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
[AMP en PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) y
[runtimes y almacenamiento Colab](https://research.google.com/colaboratory/faq.html).

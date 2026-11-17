# Figuras para informe — historiales y validation

PNG a 300 dpi; PDF/SVG vectoriales; CSV de datos por figura. Sin inferencia nueva ni acceso a test.
Las comparaciones usan una seed y validation reutilizada. No incluyen intervalos de confianza.
Las matrices conservan argmax softmax, filas reales y columnas predichas.

## Historial de ConvNeXt-Tiny

Epochs reales; línea en mejor época. Train se mide durante updates con aumentos; validation sin aumentos. No se rellenan épocas tras early stopping.

[png](convnext_tiny_training.png) | [pdf](convnext_tiny_training.pdf) | [svg](convnext_tiny_training.svg) | [csv](convnext_tiny_training.csv)

## Matriz de confusión de ConvNeXt-Tiny

Filas reales, columnas predichas; argmax original; normalización por fila.

[png](convnext_tiny_confusion.png) | [pdf](convnext_tiny_confusion.pdf) | [svg](convnext_tiny_confusion.svg) | [csv](convnext_tiny_confusion.csv)

## ROC multiclase de ConvNeXt-Tiny

One-vs-rest, scores originales; no corresponde a test.

[png](convnext_tiny_roc.png) | [pdf](convnext_tiny_roc.pdf) | [svg](convnext_tiny_roc.svg) | [csv](convnext_tiny_roc.csv)

## Historial de VMamba-Tiny

Epochs reales; línea en mejor época. Train se mide durante updates con aumentos; validation sin aumentos. No se rellenan épocas tras early stopping.

[png](vmamba_tiny_training.png) | [pdf](vmamba_tiny_training.pdf) | [svg](vmamba_tiny_training.svg) | [csv](vmamba_tiny_training.csv)

## Matriz de confusión de VMamba-Tiny

Filas reales, columnas predichas; argmax original; normalización por fila.

[png](vmamba_tiny_confusion.png) | [pdf](vmamba_tiny_confusion.pdf) | [svg](vmamba_tiny_confusion.svg) | [csv](vmamba_tiny_confusion.csv)

## ROC multiclase de VMamba-Tiny

One-vs-rest, scores originales; no corresponde a test.

[png](vmamba_tiny_roc.png) | [pdf](vmamba_tiny_roc.pdf) | [svg](vmamba_tiny_roc.svg) | [csv](vmamba_tiny_roc.csv)

## Historial de Híbrido de atención cruzada

Epochs reales; línea en mejor época. Train se mide durante updates con aumentos; validation sin aumentos. No se rellenan épocas tras early stopping.

[png](hybrid_cross_attention_training.png) | [pdf](hybrid_cross_attention_training.pdf) | [svg](hybrid_cross_attention_training.svg) | [csv](hybrid_cross_attention_training.csv)

## Matriz de confusión de Híbrido de atención cruzada

Filas reales, columnas predichas; argmax original; normalización por fila.

[png](hybrid_cross_attention_confusion.png) | [pdf](hybrid_cross_attention_confusion.pdf) | [svg](hybrid_cross_attention_confusion.svg) | [csv](hybrid_cross_attention_confusion.csv)

## ROC multiclase de Híbrido de atención cruzada

One-vs-rest, scores originales; no corresponde a test.

[png](hybrid_cross_attention_roc.png) | [pdf](hybrid_cross_attention_roc.pdf) | [svg](hybrid_cross_attention_roc.svg) | [csv](hybrid_cross_attention_roc.csv)

## Comparación de los tres clasificadores

Una seed y validation reutilizada para selección; sin intervalos ni prueba de superioridad estadística.

[png](classifier_comparison.png) | [pdf](classifier_comparison.pdf) | [svg](classifier_comparison.svg) | [csv](classifier_comparison.csv)

## ROC y precisión–recall TB comparativas

AP es average precision, no área trapezoidal. Scores TB originales; ningún umbral queda seleccionado.

[png](TB_roc_precision_recall.png) | [pdf](TB_roc_precision_recall.pdf) | [svg](TB_roc_precision_recall.svg) | [csv](TB_roc_precision_recall.csv)

## Entrenamiento del detector YOLO11n

La línea marca el mejor mAP50–95 del historial. El resultado final de best.pt procede del resumen y puede diferir del CSV de entrenamiento. P/R no son sensibilidad por imagen.

[png](detector_training.png) | [pdf](detector_training.pdf) | [svg](detector_training.svg) | [csv](detector_training.csv)

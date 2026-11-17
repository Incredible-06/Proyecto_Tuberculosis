# Resultados de validación y avance de la fase 5

La comparación controlada incorpora ahora cinco clasificadores: ConvNeXt Tiny, VMamba Tiny, el híbrido original (ConvNeXt + VMamba), ViT-B/16 y el nuevo híbrido ConvNeXt + ViT-B/16. El híbrido original mantiene el primer lugar por el criterio predefinido de Macro F1 (0,9645 frente a 0,9644 de ViT-B/16 y 0,9630 del híbrido ConvNeXt + ViT-B/16). ViT-B/16 obtuvo la mayor sensibilidad TB por argmax (110 de 120 imágenes, 91,67 %), mientras el nuevo híbrido ConvNeXt + ViT-B/16 empató la mayor exactitud global (97,82 %) y alcanzó la mayor precisión TB (97,22 %), ROC AUC (0,9975) y Average Precision (0,9806). La fase 5 sigue en desarrollo; aún faltan decisiones y pruebas para cerrar la evaluación.

La revisión integra el notebook 06 ejecutado en Colab (preparación de fase 5), el notebook 04b (ViT-B/16) y el notebook 04c (híbrido ConvNeXt + ViT-B/16), ambos ejecutados en GPU NVIDIA L4 bajo el mismo protocolo controlado (semilla 20261019, 512 píxeles y hasta 12 épocas). La validación contiene 1.241 radiografías: 570 sin enfermedad, 551 con otras enfermedades y 120 con tuberculosis. Las figuras a color son las imágenes originales exportadas por las ejecuciones. El conjunto de prueba permanece reservado.

## Comparación de los clasificadores

La exactitud indica cuántas imágenes se clasificaron correctamente. Macro F1 resume el desempeño dando el mismo peso a las tres clases. Para TB también se muestra cuántas de sus 120 imágenes fueron reconocidas.

| Modelo | Exactitud | Macro F1 | TB reconocidas |
|---|---|---|---|
| ConvNeXt Tiny | 97,10 % | 0,9427 | 94 de 120 |
| VMamba Tiny | 96,86 % | 0,9497 | 101 de 120 |
| Híbrido (ConvNeXt+VMamba) | 97,82 % | 0,9645 | 107 de 120 |
| ViT-B/16 | 97,74 % | 0,9644 | 110 de 120 |
| Híbrido (ConvNeXt+ViT-B/16) | 97,82 % | 0,9630 | 105 de 120 |

Figura 1. Comparación de los cinco clasificadores en validación.

![Comparación de los cinco clasificadores en validación.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\hybrid_vit_evidence\classifier_comparison.png)

Los tres modelos superiores (híbrido ConvNeXt+VMamba, ViT-B/16 e híbrido ConvNeXt+ViT-B/16) superan con claridad a ConvNeXt Tiny y VMamba Tiny. El híbrido original y ViT-B/16 quedan prácticamente empatados en Macro F1 (0,9645 frente a 0,9644), con mayor sensibilidad TB en ViT-B/16 (110 frente a 107). Por su parte, el híbrido ConvNeXt + ViT-B/16 iguala la exactitud máxima (97,82 %) con solo tres falsas alarmas TB y logra la mejor separación probabilística de TB (ROC AUC de 0,9975 y AP de 0,9806), lo que abre margen para ajustar su umbral operativo. Todavía falta comprobar el desempeño en el conjunto de prueba.

## Resultados de ConvNeXt Tiny

ConvNeXt obtuvo 97,10 % de exactitud y Macro F1 de 0,9427. Completó 12 épocas de entrenamiento y su mejor versión fue la época 11. Una época equivale a una pasada por los datos de entrenamiento.

Figura 2. Pérdidas y métricas de ConvNeXt durante el entrenamiento.

![Pérdidas y métricas de ConvNeXt durante el entrenamiento.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\convnext_tiny_training.png)

Reconoció 94 de las 120 imágenes TB, una sensibilidad de 78,33 %. Omitió 26: cuatro se clasificaron como health y 22 como sick. Además, tres imágenes sick recibieron la etiqueta TB. Su principal dificultad fue distinguir TB de otras enfermedades.

Figura 3. Errores de ConvNeXt con conteos y porcentajes por clase real.

![Errores de ConvNeXt con conteos y porcentajes por clase real.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\convnext_tiny_confusion.png)

En las matrices, las filas indican la clase real y las columnas la predicción. Health corresponde a sin enfermedad; sick, a otras enfermedades. Los valores de la diagonal son aciertos.

## Resultados de VMamba Tiny

VMamba obtuvo 96,86 % de exactitud y Macro F1 de 0,9497. Completó 12 épocas y su mejor versión fue la época 11. Aunque su exactitud general fue ligeramente menor que la de ConvNeXt, mejoró el reconocimiento de TB.

Figura 4. Pérdidas y métricas de VMamba durante el entrenamiento.

![Pérdidas y métricas de VMamba durante el entrenamiento.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\vmamba_tiny_training.png)

Reconoció 101 de las 120 imágenes TB, una sensibilidad de 84,17 %. Omitió 19: 12 se clasificaron como health y siete como sick. Cuatro imágenes sick recibieron la etiqueta TB. El balance entre clases mejoró, pero todavía hubo omisiones importantes de TB.

Figura 5. Errores de VMamba con conteos y porcentajes por clase real.

![Errores de VMamba con conteos y porcentajes por clase real.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\vmamba_tiny_confusion.png)

## Resultados del modelo híbrido (ConvNeXt + VMamba)

El híbrido original obtuvo 97,82 % de exactitud y Macro F1 de 0,9645. Terminó en la época 7 por parada anticipada y se conservó la época 3, que fue la mejor. Las curvas muestran que la pérdida de validación aumentó después, por lo que se mantiene esa versión.

Figura 6. Pérdidas y métricas del híbrido ConvNeXt + VMamba durante el entrenamiento.

![Pérdidas y métricas del híbrido ConvNeXt + VMamba durante el entrenamiento.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\hybrid_cross_attention_training.png)

Acertó 1.214 de las 1.241 imágenes y reconoció 107 de las 120 TB, una sensibilidad de 89,17 %. Omitió 13: seis se clasificaron como health y siete como sick. Además, cuatro imágenes sick recibieron la etiqueta TB. Sigue siendo el candidato principal por Macro F1, aunque esos errores todavía requieren revisión.

Figura 7. Errores del híbrido ConvNeXt + VMamba con conteos y porcentajes por clase real.

![Errores del híbrido ConvNeXt + VMamba con conteos y porcentajes por clase real.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\drive_evidence\hybrid_cross_attention_confusion.png)

## Resultados de ViT-B/16

ViT-B/16 obtuvo 97,74 % de exactitud y Macro F1 de 0,9644. Se entrenó con pesos oficiales ImageNet adaptados mediante interpolación bicúbica de embeddings posicionales de 224 a 512 píxeles; completó 12 épocas y su mejor versión fue la época 11.

Figura 8. Pérdidas y métricas de ViT-B/16 durante el entrenamiento.

![Pérdidas y métricas de ViT-B/16 durante el entrenamiento.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\vit_evidence\vit_b_16_training.png)

Acertó 1.213 de las 1.241 imágenes y reconoció 110 de las 120 TB, la sensibilidad por argmax más alta entre los cinco clasificadores (91,67 %). Omitió 10: tres se clasificaron como health y siete como sick. Ninguna imagen health fue confundida con TB y siete imágenes sick recibieron la etiqueta TB.

Figura 9. Errores de ViT-B/16 con conteos y porcentajes por clase real.

![Errores de ViT-B/16 con conteos y porcentajes por clase real.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\vit_evidence\vit_b_16_confusion.png)

## Resultados del híbrido ConvNeXt + ViT-B/16

El nuevo híbrido ConvNeXt-Tiny + ViT-B/16 con atención cruzada bidireccional obtuvo 97,82 % de exactitud y Macro F1 de 0,9630. Terminó en la época 6 por parada anticipada y conservó la época 2 (fin del warmup de la fusión y la norma final), pues al descongelar las últimas etapas de ambas ramas en la época 3 la pérdida de validación aumentó de 0,159 a 0,522.

Figura 10. Pérdidas y métricas del híbrido ConvNeXt + ViT-B/16 durante el entrenamiento.

![Pérdidas y métricas del híbrido ConvNeXt + ViT-B/16 durante el entrenamiento.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\hybrid_vit_evidence\hybrid_convnext_vit_training.png)

Acertó 1.214 de las 1.241 imágenes y reconoció 105 de las 120 TB (87,50 % de sensibilidad), omitiendo 15 (cuatro health y 11 sick). Generó solo tres falsas alarmas TB (una health y dos sick), logrando la mayor precisión TB (97,22 %), especificidad (99,73 %), ROC AUC (0,9975) y Average Precision (0,9806) de los cinco modelos.

Figura 11. Errores del híbrido ConvNeXt + ViT-B/16 con conteos y porcentajes por clase real.

![Errores del híbrido ConvNeXt + ViT-B/16 con conteos y porcentajes por clase real.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\hybrid_vit_evidence\hybrid_convnext_vit_confusion.png)

## Umbrales y estado de la evaluación

Se exploró la probabilidad mínima p_TB para emitir una alerta TB en los tres modelos punteros (híbrido ConvNeXt+VMamba, ViT-B/16 e híbrido ConvNeXt+ViT-B/16), por separado de la clase original por argmax. Los valores siguientes son ejemplos de validación y todavía no están fijados.

| Modelo | Umbral | TB identificadas | TB omitidas | Falsas alarmas |
|---|---|---|---|---|
| Híbrido (C+VM) | 0,10 | 111 | 9 | 5 |
| Híbrido (C+VM) | 0,50 | 107 | 13 | 4 |
| Híbrido (C+VM) | 0,90 | 100 | 20 | 1 |
| ViT-B/16 | 0,10 | 115 | 5 | 22 |
| ViT-B/16 | 0,50 | 110 | 10 | 6 |
| ViT-B/16 | 0,90 | 100 | 20 | 2 |
| Híbrido (C+ViT) | 0,10 | 111 | 9 | 15 |
| Híbrido (C+ViT) | 0,30 | 108 | 12 | 5 |
| Híbrido (C+ViT) | 0,50 | 105 | 15 | 2 |
| Híbrido (C+ViT) | 0,90 | 88 | 32 | 0 |

En el híbrido ConvNeXt+VMamba, bajar de 0,50 a 0,10 elevó a 111 los casos TB con cinco falsas alarmas. En ViT-B/16, el umbral 0,50 identifica 110 casos con seis falsas alarmas y bajar a 0,10 alcanza 115 de 120 TB con 22 falsas alarmas. En el híbrido ConvNeXt+ViT-B/16, bajar a 0,30 recupera 108 casos TB con solo cinco falsas alarmas y subir a 0,90 elimina las falsas alarmas a costa de omitir 32 casos. Falta acordar el equilibrio entre omisiones, carga de revisión y la selección final del clasificador.

## Detector de lesiones y siguientes pasos

YOLO11n conserva los resultados de la fase anterior: mAP50 de 0,484 y mAP50–95 de 0,216. Estas medidas evalúan las cajas y su ubicación, no el porcentaje de pacientes diagnosticados. En el punto reportado por la herramienta recuperó cerca del 45,1 % de las lesiones anotadas; su capacidad de localización sigue siendo limitada.

El diagnóstico nuevo del detector no se activó en estas ejecuciones. Falta revisar cajas y errores, comprobar las explicaciones visuales y el coste computacional de los candidatos, elegir el modelo final entre los tres punteros, fijar los umbrales y evaluar la concordancia. La regla propuesta conserva la alerta del clasificador aunque no haya cajas y envía esa discrepancia a revisión.

Antes de abrir el test deben quedar fijados el modelo, los pesos, el preprocesamiento, los umbrales y la concordancia. La comparación usa una sola semilla y validación reutilizada, sin intervalos de confianza ni independencia por paciente garantizada. Por ello, la fase 5 sigue abierta y los conjuntos de prueba y externos permanecen reservados.

## Fuentes revisadas

[Notebook 06 ejecutado en Google Colab](https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD)

[Resultados y registros de la ejecución de fase 5](https://drive.google.com/drive/folders/17JBPtxHx5jL0SEnPY-T56MEt6YM7URgW)

[Figuras y tablas exportadas de fase 5](https://drive.google.com/drive/folders/1r0NumlzRBzGZiQicnYeHEw_k7RWAXjZS)

[Notebook 04b (ViT-B/16) ejecutado en Google Colab](https://colab.research.google.com/drive/1tvKNsTzirSa4G7wXO_l7xZqp3YtoM0fU)

[Corrida y registros de ViT-B/16 en Drive](https://drive.google.com/drive/folders/1yqZz22Wm6EQgV9zekJtt0mHJTxuWGoF8)

[Figuras y tablas exportadas de ViT-B/16](https://drive.google.com/drive/folders/1m5WrwO8KpmZq5XYJEt3KO22iOcb1Igh6)

[Notebook 04c (Híbrido ConvNeXt + ViT-B/16) ejecutado en Google Colab](https://colab.research.google.com/drive/1hr1r-vRpuHDOUV9KmRFlh9AzL-k4Tbvw)

[Corrida y registros de Híbrido ConvNeXt + ViT-B/16 en Drive](https://drive.google.com/drive/folders/1h3QICmhofofrIFbhMY3tOSdkBrx8xVOT)

[Figuras y tablas exportadas de Híbrido ConvNeXt + ViT-B/16 y comparación de 5 modelos](https://drive.google.com/drive/folders/1NsRcHaE7VCyz2drTjTIAF_JasZvgE0tF)

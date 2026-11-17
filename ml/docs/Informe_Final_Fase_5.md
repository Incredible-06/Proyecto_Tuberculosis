# Informe final de fase 5: selección, explicabilidad, evaluación congelada y generalización

La fase 5 completa el cierre experimental del sistema de apoyo diagnóstico para tuberculosis en radiografías de tórax. En primer lugar se completó el diagnóstico integral sobre el conjunto de validación (1.241 imágenes en 1.237 grupos de contenido) para los cinco clasificadores entrenados bajo protocolo controlado (ConvNeXt-Tiny, VMamba-Tiny, Híbrido ConvNeXt+VMamba, ViT-B/16 e Híbrido ConvNeXt+ViT-B/16) y el detector baseline YOLO11n, incorporando intervalos de confianza por bootstrap estratificado agrupado por group_id, coste computacional comparable en GPU NVIDIA L4, revisión visual determinista de errores y explicabilidad verificada con Grad-CAM, Attention Rollout y CrossFusion.

Con esa evidencia se congeló el contrato inmutable antes de abrir el conjunto de prueba reservado: clasificador ViT-B/16 (umbral operativo p_TB >= 0,30 conservando argmax y probabilidades originales), detector auxiliar YOLO11n (umbral de confianza >= 0,25) y regla de concordancia aditiva en cuatro estados sin supresión de alertas del clasificador. En la evaluación única sobre test interno (1.241 radiografías), el sistema congelado alcanzó 97,18 % de exactitud, Macro-F1 de 0,9551, TB ROC AUC de 0,9941 y una sensibilidad combinada TB de 95,83 % (115 de 120 casos), con apenas 0,98 % de falsas alertas en imágenes no TB. Por contraste, la evaluación externa en Montgomery (138) y Shenzhen (662) evidenció un fuerte domain shift hospitalario sobre el clasificador global (especificidad de 25,00 % y 12,88 %), mientras el detector localizado YOLO11n mantuvo AUC de 0,811 y 0,804 con especificidad de 100,0 % y 97,5 %.

## 1. Comparación de validación e incertidumbre por grupos (group_id)

Sobre las 1.241 imágenes de validación se verificaron los cinco checkpoints contra sus hashes de procedencia originales (5/5 pruebas smoke superadas con delta máximo de probabilidad de 5,71e-4 en NVIDIA L4 con backend oflex). Para cuantificar incertidumbre respetando los pares de duplicados cercanos, se aplicó bootstrap estratificado por group_id (1.237 grupos, 2.000 réplicas, semilla 20261019).

| Modelo | Exactitud | Macro F1 (IC 95 %) | Sensibilidad TB argmax (IC 95 %) | TB AUC |
|---|---|---|---|---|
| ConvNeXt-Tiny | 97,10 % | 0,9427 [0,9224, 0,9594] | 78,33 % [70,83 %, 85,00 %] | 0,9958 |
| VMamba-Tiny | 96,86 % | 0,9497 [0,9321, 0,9648] | 84,17 % [77,50 %, 90,00 %] | 0,9934 |
| Híbrido (ConvNeXt+VMamba) | 97,82 % | 0,9645 [0,9484, 0,9786] | 89,17 % [83,33 %, 94,17 %] | 0,9958 |
| ViT-B/16 | 97,74 % | 0,9644 [0,9484, 0,9775] | 91,67 % [86,65 %, 95,83 %] | 0,9969 |
| Híbrido (ConvNeXt+ViT-B/16) | 97,82 % | 0,9630 [0,9473, 0,9762] | 87,50 % [80,83 %, 92,50 %] | 0,9975 |

Figura 1. Comparación de los cinco clasificadores en validación.

![Comparación de los cinco clasificadores en validación.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_review\hybrid_vit_evidence\classifier_comparison.png)

Los deltas pareados de bootstrap entre los tres candidatos punteros (Híbrido C+VM, ViT-B/16 e Híbrido C+ViT) muestran que sus intervalos de confianza del 95 % cruzan cero (por ejemplo, Híbrido C+VM menos ViT-B/16: Δ Macro-F1 = +0,00003 [-0,0174, +0,0162] y Δ sensibilidad TB = -2,50 pp [-8,33, +3,33]), confirmando un empate estadístico en Macro-F1 sobre esta validación con una sola semilla.

## 2. Diagnóstico del detector YOLO11n, concordancia exploratoria y coste

Se ejecutó YOLO11n sobre las 1.241 radiografías de validación (512 px, FP32, NMS IoU=0,7). En el punto de confianza 0,25 recuperó 56,82 % de las 176 lesiones anotadas (IoU >= 0,5), detectó al menos una caja en 74,17 % de los pacientes TB y mantuvo 0 falsas alarmas en health (0/570) y solo 7/551 en sick (0,62 % en no-TB).

| Umbral conf | Recall lesión | Precisión caja | Sens. imagen (cualq.) | Sens. imagen (loc.) | Falsa alarma no-TB |
|---|---|---|---|---|---|
| 0,10 | 65,91 % | 32,86 % | 85,00 % | 75,00 % | 1,43 % |
| 0,25 | 56,82 % | 43,67 % | 74,17 % | 65,00 % | 0,62 % |
| 0,50 | 39,20 % | 71,88 % | 55,83 % | 47,50 % | 0,00 % |

Figura 2. Curvas por umbral y FROC exploratoria de YOLO11n en validación.

![Curvas por umbral y FROC exploratoria de YOLO11n en validación.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_diag\phase5_diag_20261116T042213Z_ac3bcb\detector_diagnostic.png)

La medición comparable en GPU NVIDIA L4 (FP32, 512×512) mostró que Híbrido (C+VM) y ViT-B/16 tienen latencia prácticamente idéntica por imagen (30,0 ms y 31,3 ms en batch=1), mientras Híbrido (C+ViT) requiere 40,7 ms y 115,2 M de parámetros. YOLO11n añade 19,4 ms y 2,59 M de parámetros.

| Componente | Parámetros | Peso disco | Latencia b=1 | VRAM pico b=1 | Throughput b=8 |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 27,82 M | 106,2 MB | 8,7 ms | 186 MB | 104,0 img/s |
| VMamba-Tiny | 29,48 M | 112,6 MB | 22,8 ms | 250 MB | 71,0 img/s |
| Híbrido (ConvNeXt+VMamba) | 58,22 M | 222,2 MB | 30,0 ms | 359 MB | 42,0 img/s |
| ViT-B/16 | 86,44 M | 329,8 MB | 31,3 ms | 389 MB | 33,4 img/s |
| Híbrido (ConvNeXt+ViT-B/16) | 115,18 M | 439,5 MB | 40,7 ms | 519 MB | 25,0 img/s |
| YOLO11n | 2,59 M | 5,2 MB | 19,4 ms | 42 MB | 94,9 img/s |

## 3. Explicabilidad verificada y revisión visual de errores

Se implementó Grad-CAM por rama para los cinco modelos, Attention Rollout (32×32) para las ramas ViT y mapas de atención cruzada (8×8) en CrossFusion. Los 60 chequeos ejecutados (12 casos representativos de validación × 5 arquitecturas) superaron el 100 % de las pruebas de finitud, no nulidad de gradientes (incluso a través del kernel oflex de VMamba), contraste entre clase TB y health (correlación negativa entre -0,06 y -0,33) y aleatorización de la cabeza/fusión sobre copia en memoria (correlación cercana a cero o negativa frente al checkpoint entrenado).

En la alineación auxiliar frente a cajas GT en los siete casos TB evaluados, ViT-B/16 obtuvo el mayor enriquecimiento promedio de relevancia dentro de lesiones anotadas (1,42× frente a área uniforme), por encima de ConvNeXt-Tiny (1,31×), Híbrido C+ViT (1,22×), Híbrido C+VM (1,13×) y VMamba-Tiny (0,86×). La revisión visual de los 67 casos seleccionados (FN, FP y discrepancias) mostró que en errores compartidos difíciles (como tb0003.png) los mapas se desvían hacia regiones basales/diafragmáticas en lugar de las cajas apicales.

Figura 3. Comparación de mapas de explicabilidad en acierto TB (tb0378.png).

![Comparación de mapas de explicabilidad en acierto TB (tb0378.png).](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_diag\phase5_diag_20261116T042213Z_ac3bcb\explainability_panels\explain_01_tb0378.png)

Figura 4. Revisión visual de discrepancias clasificador–detector con cajas GT (verde).

![Revisión visual de discrepancias clasificador–detector con cajas GT (verde).](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_diag\phase5_diag_20261116T042213Z_ac3bcb\visual_review_panels\visual_review_page_01.png)

## 4. Contrato congelado y evaluación única sobre test reservado

Con base en el empate estadístico en Macro-F1, la mayor sensibilidad TB por argmax (91,67 %) y combinada (96,67 % a p_TB >= 0,30), el mayor enriquecimiento en cajas GT (1,42×) y la portabilidad en PyTorch estándar sin dependencia CUDA custom (oflex) para api/ y web/, se seleccionó y congeló ViT-B/16 (época 11) con umbral operativo p_TB >= 0,30 junto a YOLO11n (conf >= 0,25) bajo la regla aditiva de cuatro estados (configs/phase5_frozen_contract.json). La evaluación se ejecutó una sola vez sobre las 1.241 radiografías reservadas de manifests/test.csv.

| Métrica en test reservado (n = 1.241) | Valor puntual | IC 95 % (bootstrap group_id) |
|---|---|---|
| Exactitud ViT-B/16 (argmax) | 97,18 % | [96,22 %, 98,06 %] |
| Macro-F1 ViT-B/16 (argmax) | 0,9551 | [0,9378, 0,9702] |
| TB ROC AUC / Average Precision | 0,9941 / 0,9683 | [0,9885, 0,9981] |
| Sensibilidad TB clasificador (argmax) | 87,50 % | [80,83 %, 93,33 %] |
| Sensibilidad TB clasificador (p_TB >= 0,30) | 90,00 % | [84,17 %, 95,00 %] |
| Especificidad TB clasificador (p_TB >= 0,30) | 99,11 % | [98,48 %, 99,64 %] |
| YOLO11n (conf >= 0,25) recall lesión / sens. caja | 48,37 % / 73,33 % | [40,96 %, 55,87 %] |
| Sensibilidad TB combinada (concordancia) | 95,83 % (115/120) | [92,48 %, 99,17 %] |
| Tasa de revisión en no-TB (concordancia) | 0,98 % (11/1.121) | [0,45 %, 1,61 %] |

Figura 5. Matriz de confusión del contrato congelado (ViT-B/16) en test reservado.

![Matriz de confusión del contrato congelado (ViT-B/16) en test reservado.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_final\phase5_final_20261116T051424Z_f5c880\test_evaluation\test_confusion_matrix.png)

Figura 6. Curvas ROC OVR y Precisión-Recall TB en test reservado.

![Curvas ROC OVR y Precisión-Recall TB en test reservado.](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_final\phase5_final_20261116T051424Z_f5c880\test_evaluation\test_roc_pr.png)

Figura 7. Distribución de estados de concordancia en test reservado (p_TB >= 0,30 y YOLO11n >= 0,25).

![Distribución de estados de concordancia en test reservado (p_TB >= 0,30 y YOLO11n >= 0,25).](C:\Users\andyr\Desktop\Proyecto_Tuberculosis\ml\reports\phase5_final\phase5_final_20261116T051424Z_f5c880\test_evaluation\test_concordance.png)

En el test reservado, YOLO11n rescató 7 casos TB adicionales que el clasificador había dejado por debajo de 0,30, llevando la detección combinada a 115/120 (95,83 %) con solo 5 casos TB omitidos, 2 falsas alertas en health (0,35 %) y 9 en sick (1,63 %). La consistencia entre validación (96,67 % combinado) y test (95,83 % combinado) confirma que el contrato congelado no sufrió sobreajuste.

## 5. Evaluación externa (Montgomery y Shenzhen) y análisis de domain shift

La auditoría previa confirmó 0 duplicados visuales entre Montgomery (138) / Shenzhen (662) y las particiones train/validation de TBX11K (las coincidencias exactas se encuentran en extra/mc+shenzhen y en test oficial no etiquetado, excluidos del entrenamiento). Dado que estos conjuntos solo disponen de etiquetas binarias (normal frente a anormal compatible con TB), se evaluaron p_TB >= 0,30, la concordancia combinada y la referencia 1 - p_health, comparándolos además con el detector YOLO11n aislado.

| Conjunto externo | ViT-B/16 p_TB>=0,30 (Sens / Espec / AUC) | YOLO11n conf>=0,25 solo (Sens / Espec / AUC) |
|---|---|---|
| Montgomery (n = 138: 58 TB, 80 normal) | 74,14 % / 25,00 % / 0,580 | 43,10 % / 100,00 % (0/80 FP) / 0,811 |
| Shenzhen (n = 662: 336 TB, 326 normal) | 94,94 % / 12,88 % / 0,647 | 49,11 % / 97,55 % (8/326 FP) / 0,804 |

El contraste entre ambos componentes es un hallazgo relevante: en imágenes normales externas, ViT-B/16 emitió p_health cercana a cero (media 0,0046 en Montgomery y 0,0440 en Shenzhen), clasificando la mayoría de radiografías normales como sick o TB (especificidad de 25,00 % y 12,88 %). Esto refleja un domain shift clásico de TBX11K, donde las clases health, sick y TB provienen de cohortes hospitalarias y formatos distintos (contraste, proporciones y bordes tras letterbox), induciendo atajos de dominio en el clasificador de imagen completa. En contraste, YOLO11n —entrenado exclusivamente con cajas locales de lesión TB y sin distinguir entre dominios negativos— conservó AUC externa de 0,811 y 0,804 y especificidad de 100,0 % en Montgomery y 97,55 % en Shenzhen a conf >= 0,25.

Por rigor metodológico, la fase 5 se cierra con estos resultados sin reabrir el test interno. Para despliegue fuera del dominio TBX11K se documenta la necesidad de calibración por sitio o una futura fase 6 basada en segmentación pulmonar, normalización de contraste controlada (p. ej. ablación con CLAHE) y entrenamiento multi-hospital.

## 6. Exportación final y fuentes verificadas

El paquete congelado quedó exportado y verificado por SHA-256 en ml/models/frozen_phase5/ (vit_b_16_best.pt: 2873fbcff636ad15..., yolo11n_best.pt: 3e28e82987c8c2c8..., contrato: phase5_frozen_contract_v1) para su integración en api/ y web/.

[Notebook 07 ejecutado (diagnóstico, Grad-CAM, coste e incertidumbre)](https://colab.research.google.com/drive/1YeJmyXyk9Iwg6_GS5BAh6GvQFtkrR3gY)

[Resultados Parte 1 en Drive (phase5_diag)](https://drive.google.com/drive/folders/12zsM2dgbgWwsh2iJOxH7Q-ObVJsjIoBB)

[Notebook 08 ejecutado (evaluación única test y externos)](https://colab.research.google.com/drive/1bLNyXqWGrUULrfAWZmxCrTQhzZGbBveP)

[Resultados finales Parte 2 en Drive (phase5_final)](https://drive.google.com/drive/folders/1FwpwLNKmJryjRYkXdFYvOcIs7PfpbuSR)

[Notebook 06 ejecutado (preparación inicial fase 5)](https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD)

[Notebook 04b (ViT-B/16) ejecutado en Colab](https://colab.research.google.com/drive/1tvKNsTzirSa4G7wXO_l7xZqp3YtoM0fU)

[Notebook 04c (Híbrido ConvNeXt + ViT-B/16) ejecutado en Colab](https://colab.research.google.com/drive/1hr1r-vRpuHDOUV9KmRFlh9AzL-k4Tbvw)

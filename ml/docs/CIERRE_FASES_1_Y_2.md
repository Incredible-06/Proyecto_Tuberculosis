# Estado revisado de las fases 1 y 2

Revisión del 21 de octubre de 2026, America/Lima. Fase 1 corresponde al entorno;
Fase 2, al dataset. No incluye entrenamientos ni integración web.

## Fase 1

La nueva entrega del 21 de octubre confirma 9 celdas del entorno y 20 de datos
ejecutadas en orden y sin errores, con el código preparado sin cambios. Ambas
ejecuciones registraron los 12.278 hashes completos y cero cruces conocidos.
La ejecución de entorno guardó resultados a las 18:49 y la de datos a las 18:54
(Lima). Registro: `reports/colab_20261021/verification.json`.

Comprobada con la ejecución histórica de Colab del 20 de octubre a las 00:11:
A100, PyTorch 2.11.0+cu130, operación CUDA pequeña aprobada, Drive persistente
y comparación SHA-256 completa de las 12.278 imágenes. El contenido de los JSON
en Drive coincide con la copia local. experiments y checkpoints están vacías
porque se prepararon para la etapa de modelos.

## Fase 2

TBX11K preparado con 8.274 contenidos únicos, 8.251 grupos y particiones
5.792/1.241/1.241. Doce cruces internos conocidos en cero, nueve pruebas ligeras
aprobadas y reproducción de manifiestos idéntica. Se conservan 1.211 cajas COCO.
La evidencia histórica Colab registra un lote CPU 4×3×512×512.

Las tres clases son una decisión explícita del clasificador: health, sick y TB.
Se estudiaron las cuatro categorías descriptivas y se conservaron los subtipos;
30 imágenes TB tienen ambos subtipos y una carece de cajas anotadas. Mantener
TB agrupada evita una asignación exclusiva arbitraria. Un clasificador de cuatro
salidas sigue siendo un objetivo distinto que necesita tratar los casos mixtos.

La adquisición externa se amplió a las 138 imágenes Montgomery y 662 Shenzhen
originales, con sus 800 lecturas y documentos de uso NLM. Los originales están
en external_data, fuera de Git y de los manifiestos internos congelados. Sus
etiquetas binarias se basan en la convención documentada por la fuente.

La preparación externa no equivale a independencia: TBX11K contiene versiones
de fuentes externas. Los candidatos por dHash están registrados y todas las
imágenes siguen reservadas hasta revisar solapamientos y definir la evaluación.

## Informe y capturas

- Vigente: docs/Informe_Fases_1_y_2_Revisado.docx. Usa la versión editada por el usuario y conserva la carátula sin el año, según su indicación.
- Texto versionable: docs/Informe_Fases_1_y_2.md.
- Guía exacta de celdas: docs/GUIA_CAPTURAS.md.
- Notebooks revisados: 00 entorno, 01 datos; bloques F1-E1/F1-E2 y F2-E1 a F2-E6.
- Paquete nuevo para subir a Drive: dist/tbx11k_fases_1_y_2_colab.zip; sin imágenes ni pesos.
- Auditoría integrada: reports/revision_20261021/verification.json.
- Fuentes externas: reports/external_sources/preparation_summary.json y manifests/external.

El informe recupera el gráfico de barras original del notebook y conserva las
cinco imágenes que seleccionó el usuario. Incluye seis figuras, con explicaciones
para revisar los resultados sin abrir los notebooks. La redacción es más directa
y no presenta fechas u horas en el contenido.

Las versiones revisadas tienen 9 y 20 celdas de código y ahora cuentan también
con ejecución Colab verificada. Las capturas aportadas ya se incorporaron al
informe y los PNG exportados coinciden con los mostrados por el notebook.
Los notebooks históricos ejecutados no se modificaron.
La versión anterior de su código se conserva en reports/revision_20261021/referencias.

## Límites del cierre

La preparación técnica permite pasar a la etapa de modelos. No se garantiza
independencia por paciente ni búsqueda exhaustiva de duplicados. Las cajas se
preservan como estructura de detección; transformar cajas junto con imágenes y
adaptarlas a YOLO/RT-DETR corresponde a esa etapa. La prueba CUDA básica no
valida ConvNeXt o Vision Mamba. Las condiciones de uso se registran en
FUENTES_Y_LICENCIAS.md, incluida la discrepancia de declaraciones TBX11K.

La nueva entrega acredita el uso de las versiones revisadas. No hace falta
repetir la ejecución para sustentar las dos fases. Los cambios de código y
documentación locales aún no están publicados en GitHub.

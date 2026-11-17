# Guía para las capturas del informe

**Evidencias recibidas y verificadas el 21 de octubre de 2026.** Las capturas ya se incorporaron al informe. No hace falta repetir la ejecución para respaldar estas dos fases. El registro está en `reports/colab_20261021/verification.json`.

Use los notebooks de la carpeta `notebooks`, incluidos en el ZIP nuevo de `dist`.
Los nombres 00 y 01 son el orden de ejecución; las fases del curso son 1 entorno y 2 datos.
La columna “celda total” cuenta todas las celdas desde 1, incluidas las de texto.
La columna “código” cuenta solo las celdas ejecutables. **Busque el identificador del título**:
el número `[n]` que muestra Colab es un contador de ejecución y puede ser distinto.

## Para repetir capturas en futuras ejecuciones

1. Sustituya en Drive el paquete pequeño `tbx11k_fases_1_y_2_colab.zip` por el de `dist`.
   El dataset `TBX11K.zip` permanece igual. El paquete no incluye imágenes ni pesos.
2. Abra los notebooks revisados en Colab. En 00 seleccione GPU; 01 puede ejecutarse en CPU.
3. Ejecute todas las celdas en orden. Las nuevas salidas deben pertenecer a esa ejecución.
4. Capture el título y la salida completa del bloque; puede ocultar el código para legibilidad.
   Descargue también el `.ipynb` ejecutado y conserve la carpeta nueva de resultados en Drive.

## Celdas exactas

| Notebook | Bloque | Celda total | Código | Qué debe verse |
|---|---|---:|---:|---|
| 00 | F1-E1 | 18 | 8 | GPU, PyTorch, CUDA passed, seed y ausencia de entrenamiento |
| 00 | F1-E2 | 20 | 9 | 12.278 hashes completos, cero cruces y ruta persistente en Drive |
| 01 | F2-E1 | 32 | 15 | Tres clases y motivo de agrupación de los subtipos TB |
| 01 | F2-E2 | 34 | 16 | 5.792 / 1.241 / 1.241 y doce cruces conocidos en cero |
| 01 | F2-E3 | 36 | 17 | Muestras de train y cajas originales; se puede usar la imagen exportada |
| 01 | F2-E4 | 38 | 18 | Aumentos solo en train, normalización y comparación visual |
| 01 | F2-E5 | 40 | 19 | Seis muestras, lote CPU 4×3×512×512, 1.211 cajas y tb1199 |
| 01 | F2-E6 | 42 | 20 | 138 Montgomery + 662 Shenzhen y reserva externa; resumen registrado localmente |

## Selección breve recomendada

Use F1-E1, F1-E2, F2-E2, F2-E3, F2-E4 y F2-E6. F2-E1 y F2-E5 son complementarias.
Las dos capturas de Drive ya aportadas pueden acompañar la evidencia del entorno.
El informe revisado contiene recortes de las capturas de Colab entregadas y salidas gráficas de la nueva ejecución del 21 de octubre.
No se han fabricado capturas de Colab ni insertado resultados históricos en celdas nuevas.

F2-E6 muestra la preparación externa registrada y comprueba los manifiestos del paquete;
no afirma que las 800 imágenes estén cargadas o verificadas en la sesión actual de Colab.
Para reconstruir esas fuentes desde sus originales oficiales, ejecute `python scripts/prepare_external.py --download`
en un entorno con red y NumPy/Pillow. Descarga las fuentes aparte y regenera únicamente los manifiestos externos.
El paquete pequeño no incluye los originales externos; están en `external_data` local.

## Qué resultados no deben confundirse

- Las 7 y 14 celdas históricas corresponden a versiones anteriores, conservadas por trazabilidad.
- Esta revisión contiene 9 y 20 celdas de código. Se probaron localmente y se verificó la nueva entrega Colab: todas las celdas se ejecutaron en orden y sin errores.
- Si CUDA no muestra passed o los hashes completos no llegan a 12.278, esa nueva ejecución no acredita esos puntos.
- experiments y checkpoints vacíos son normales antes del entrenamiento.
- Cero cruces de grupos conocidos no significa independencia por paciente ni ausencia exhaustiva de duplicados.

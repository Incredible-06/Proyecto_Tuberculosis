# Modelos

Espacio reservado para definiciones y configuraciones de modelos de la etapa de
entrenamiento. No hay modelos entrenados en las fases 1 y 2 del plan del curso.
Los pesos y checkpoints se guardarán en Drive, fuera de Git.

El primer entrenamiento de Fase 3 está implementado en `src/tbxtrain/` y en
`notebooks/02_convnext_tiny_baseline.ipynb`. La celda smoke verifica el camino
técnico con pocas muestras; la celda de baseline usa todo train/validation y
guarda `best.pt`/`last.pt` por corrida. No se han verificado pesos entrenados
en Colab al preparar este código. Consulte `docs/AVANCE_FASE_3.md`.

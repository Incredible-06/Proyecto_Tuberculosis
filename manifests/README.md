# Manifiestos congelados

`train.csv`, `validation.csv` y `test.csv` contienen un representante por contenido
RGB exacto, agrupando las variantes cercanas confirmadas en el mismo split.
Todas las rutas son relativas a la raíz de TBX11K (`imgs/...`).

- `all_labeled_members.csv`: índice de los 8.400 archivos originales y su representante/split.
- `official_test_unlabeled.csv`: 3.302 imágenes, sin clases o etiquetas inferidas.
- `reserved_external.csv`: 576 imágenes excluidas de clasificadores, sin etiquetas inferidas.
- `external_overlap_report.csv`: rutas externas con duplicados conocidos, incluidos cruces con el test oficial.
- `manifest_metadata.json`: seed, algoritmo, conteos, cuotas y fingerprints de entradas/salidas.

`group_id` identifica contenido compartido, no un paciente. Los solapamientos con
fuentes reservadas son información de auditoría y no autorizan asignar etiquetas
al test oficial. Los metadatos `boxes_json` siguen en las coordenadas de la imagen
original; no corresponden a un tensor aumentado del lector de clasificación.

No edite manualmente estos CSV. Para reproducir use `scripts/build_manifests.py`
con otra carpeta de salida y compare fingerprints. Las imágenes originales se
conservan intactas. El test interno se reserva para la evaluación final.

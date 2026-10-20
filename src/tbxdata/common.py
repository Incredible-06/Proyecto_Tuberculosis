import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def safe_path(root, relative):
    """Resuelve rutas de manifiestos/ZIP sin permitir escapes de la raíz."""
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Ruta relativa no válida: {relative}')
    root = Path(root).resolve()
    path = (root / str(rel)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'Ruta fuera de la raíz: {relative}')
    return path

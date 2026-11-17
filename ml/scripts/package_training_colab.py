"""Paquete nuevo de Fase 3, sin modificar el ZIP histórico de Fases 1 y 2."""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxdata.common import sha256
from tbxtrain.protocol import atomic_json


def main():
    relative_files = ['README.md', 'requirements-data.txt', 'requirements-training.txt',
        'configs/data.json', 'configs/convnext_tiny_baseline.json',
        'notebooks/02_convnext_tiny_baseline.ipynb', 'docs/AVANCE_FASE_3.md',
        'reports/initial_audit/image_inventory.csv', 'reports/initial_audit/metadata_sha256.json',
        'reports/near_duplicates/review_decisions.csv', 'reports/near_duplicates/refined_pairs.csv',
        'scripts/create_training_notebook.py', 'scripts/check_training_notebook.py',
        'scripts/check_training_tests.py', 'scripts/package_training_colab.py',
        'tests/test_training_protocol.py', 'tests/test_training_torch_fixture.py', 'tests/test_training_review.py']
    for folder, pattern in [('src', '*.py'), ('manifests', '*.csv'), ('manifests', '*.json'),
                             ('reports/phase3', '*.json')]:
        relative_files.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob(pattern)
                              if '__pycache__' not in p.parts)
    files = sorted(set(relative_files))
    checksums = {name: sha256(ROOT/name) for name in files}
    destination = ROOT/'dist/tbx11k_fase3_baseline_colab.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            if name.startswith(('TBX11K/', 'external_data/', 'results/', 'checkpoints/')) or Path(name).suffix in {'.pt', '.pth'}:
                raise ValueError('Originales o pesos no deben entrar en el paquete.')
            archive.write(ROOT/name, name)
        archive.writestr('phase3_package_checksums.json', json.dumps(checksums, indent=2)+'\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError('ZIP corrupto.')
        import hashlib
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Bytes incorrectos en el paquete: '+name)
    info = {'file': destination.name, 'bytes': destination.stat().st_size,
        'sha256': sha256(destination), 'files': len(files)+1,
        'contains_original_images': False, 'contains_weights': False,
        'training_executed': False}
    atomic_json(ROOT/'dist/phase3_package_info.json', info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

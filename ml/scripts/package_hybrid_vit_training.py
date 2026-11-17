"""Paquete reproducible para entrenamiento controlado del híbrido ConvNeXt-Tiny + ViT-B/16."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tbxdata.common import sha256
from tbxtrain.protocol import atomic_json


def main():
    names = [
        'requirements-data.txt',
        'requirements-training.txt',
        'reports/initial_audit/image_inventory.csv',
        'reports/initial_audit/metadata_sha256.json',
        'reports/near_duplicates/review_decisions.csv',
        'reports/near_duplicates/refined_pairs.csv',
        'scripts/create_hybrid_vit_notebook.py',
        'scripts/package_hybrid_vit_training.py',
        'tests/test_hybrid_vit_experiment.py',
    ]
    for short in ['convnext', 'vmamba', 'hybrid']:
        for suffix in ['summary.json', 'history.csv', 'validation_predictions.csv']:
            names.append(f'reports/phase5_readiness/drive_evidence/{short}_{suffix}')
    for suffix in ['summary.json', 'history.csv', 'validation_predictions.csv']:
        names.append(f'reports/phase5_review/vit_evidence/vit_{suffix}')
    for folder, pattern in [('src', '*.py'), ('configs', '*.json'), ('manifests', '*.csv'), ('manifests', '*.json')]:
        names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob(pattern) if '__pycache__' not in p.parts)

    hashes = {n: sha256(ROOT / n) for n in sorted(set(names))}
    target = ROOT / 'dist/tbx11k_hibrido_convnext_vit_colab.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
        archive.writestr('hybrid_vit_package_checksums.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip():
            raise ValueError('ZIP corrupto.')
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(name)

    info = dict(
        file=target.name,
        bytes=target.stat().st_size,
        sha256=sha256(target),
        files=len(hashes) + 1,
        contains_original_images=False,
        contains_weights=False,
        contains_prior_notebooks=False,
        test_evaluated=False,
        colab_executed=False,
    )
    atomic_json(ROOT / 'dist/hybrid_vit_package_info.json', info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

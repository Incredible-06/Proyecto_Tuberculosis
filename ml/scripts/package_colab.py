"""Empaqueta solo el proyecto pequeño; nunca incluye TBX11K ni pesos."""
import hashlib
import json
import argparse
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='tbx11k_fases_1_y_2_colab.zip')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or not args.name.endswith('.zip'):raise ValueError('Nombre ZIP no válido.')
    destination=root/'dist'/args.name
    destination.parent.mkdir(parents=True,exist_ok=True)
    folders=['src','configs','manifests','notebooks','scripts','tests','docs','api','web','models',
             'reports/initial_audit','reports/near_duplicates','reports/phase1',
             'reports/environment/colab_review','reports/revision_20261021',
             'reports/external_sources','reports/colab_20261021']
    files=[root/name for name in ['README.md','requirements-data.txt','requirements-audit.txt','pyproject.toml','.gitignore','.gitattributes']]
    for folder in folders:
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix.lower()!='.docx' and not any(part in ['__pycache__','test_tmp','_qa','_tmp','assets'] for part in p.parts):files.append(p)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(set(files)):
            relative=p.relative_to(root).as_posix()
            if relative.startswith('TBX11K/'):
                raise ValueError('El dataset original nunca debe entrar en este ZIP.')
            archive.write(p,relative)
    with zipfile.ZipFile(destination) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('ZIP corrupto: '+bad)
    h=hashlib.sha256(destination.read_bytes()).hexdigest()
    info={'file':destination.name,'bytes':destination.stat().st_size,'sha256':h,
          'contains_original_dataset':False,'contains_weights':False}
    (destination.parent/'package_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()

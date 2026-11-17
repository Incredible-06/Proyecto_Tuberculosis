"""Paquete reproducible Fases 3/4, sin datos originales ni pesos ni entornos."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tbxdata.common import sha256
from tbxtrain.protocol import atomic_json


def main():
    names=['README.md','requirements-data.txt','requirements-training.txt',
        'requirements-phase3-suite.txt','requirements-detection.txt','docs/CIERRE_FASE_3_Y_TRANSICION_4.md',
        'docs/AVANCE_FASE_3.md','docs/CONTINUACION_V02.md',
        'reports/initial_audit/image_inventory.csv','reports/initial_audit/metadata_sha256.json',
        'reports/near_duplicates/review_decisions.csv','reports/near_duplicates/refined_pairs.csv']
    for folder,pattern in [('src','*.py'),('configs','*.json'),('manifests','*.csv'),('manifests','*.json'),
                           ('notebooks','*.ipynb'),('scripts','*.py'),('tests','*.py'),('reports/phase3','*.json'),
                           ('reports/phase3_completion','*.json')]:
        names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob(pattern) if '__pycache__' not in p.parts)
    names += ['vendor/vmamba/'+n for n in ['vmamba.py','csms6s.py','LICENSE','source.json']]
    hashes={n:sha256(ROOT/n) for n in sorted(set(names))}
    output=ROOT/'dist/tbx11k_fases_3_y_4_colab.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n in hashes: z.write(ROOT/n,n)
        z.writestr('phase3_package_checksums.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip(): raise ValueError('ZIP corrupto')
        for n,h in hashes.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h: raise ValueError(n)
    report={'file':output.name,'bytes':output.stat().st_size,'sha256':sha256(output),
        'files':len(hashes)+1,'contains_original_images':False,'contains_weights':False,
        'colab_suite_executed':False,'phase3_complete':False,'phase4_complete':False}
    atomic_json(ROOT/'dist/phase3_completion_package_info.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

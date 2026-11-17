"""Archiva capturas aportadas y recorta solo las salidas para facilitar su lectura."""
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets/colab_20261021'
SCREENSHOTS = [
    ('dataset_stage', 'c2ef3245-96fd-4ced-be71-149ba73b0c4e'),
    ('environment_details', 'a0e7b37e-4213-4e4a-bbd7-fbbfd3caa6e5'),
    ('transfer_integrity', 'bf0bbb66-48a7-40b5-b218-35d737e850f8'),
    ('F1-E1', '9aee6cc0-672a-4804-aa3a-8407c94379d3'),
    ('dataset_audit', '56010c3b-7d0f-47dc-a603-fb2bf49aba20'),
    ('class_weights', 'fa82d1d0-f142-415b-b56a-ba28b17c243c'),
    ('figures_code', 'bbb28be2-448f-47e1-892c-16803c86224a'),
    ('distribution', '20918751-ec44-497c-ade6-9171b3bc70c4'),
    ('training_samples', '66245334-aa53-415f-b766-547687b86e79'),
    ('F2-E2', 'cff12fba-20de-4da0-8cea-7f96fbed85f3'),
]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    originals = ROOT / 'results/colab_20261021/screenshots'; originals.mkdir(parents=True, exist_ok=True)
    proof = {'source': 'Capturas aportadas por el usuario el 21 de octubre de 2026', 'originals': {}, 'crops': {}}
    for name, identifier in SCREENSHOTS:
        source = Path(r'C:\Users\andyr\AppData\Local\Temp') / f'codex-clipboard-{identifier}.png'
        target = originals / (name + '.png')
        if source.is_file():
            shutil.copy2(source, target)
        elif not target.is_file():
            raise FileNotFoundError('No se encuentra la captura original: ' + name)
        proof['originals'][name] = {'file': target.relative_to(ROOT).as_posix(),
                                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    for name, box in [('F1-E1', (73, 400, 730, 569)),
                      ('transfer_integrity', (32, 287, 635, 336)),
                      ('F2-E2', (48, 444, 720, 716))]:
        with Image.open(originals / (name + '.png')) as image:
            image.crop(box).save(ASSETS / (name + '_output.png'))
        proof['crops'][name] = {'box': box, 'changes': 'Solo recorte rectangular; sin alterar texto o resultados.'}
    source = ROOT / 'reports/colab_20261021/figures'
    with Image.open(source / 'training_samples.png') as image:
        panel = Image.new('RGB', (1020, 330), 'white')
        for i, box in enumerate([(5, 5, 335, 335), (685, 5, 1015, 335), (345, 385, 675, 715)]):
            panel.paste(image.crop(box), (i * 345, 0))
        panel.save(ASSETS / 'training_samples_three.png')
    with Image.open(source / 'training_transformations.png') as image:
        image.crop((0, 0, 1280, 355)).save(ASSETS / 'training_transformations_row.png')
    proof['plot_source'] = 'reports/colab_20261021/figures; verificado contra los PNG mostrados por el notebook entregado.'
    (ROOT / 'reports/colab_20261021/captures_provenance.json').write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print('Capturas originales archivadas:', len(SCREENSHOTS))


if __name__ == '__main__': main()

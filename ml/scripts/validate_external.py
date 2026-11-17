"""Inventario y reserva de fuentes externas oficiales, separado del protocolo TBX11K."""
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PIL import Image
from tbxdata.common import read_csv, sha256
from prepare_external import SOURCES


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def dhash(image):
    small = image.convert('L').resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def validate_external(project):
    project = Path(project)
    out = project / 'reports/external_sources'; out.mkdir(parents=True, exist_ok=True)
    inventory = read_csv(project / 'reports/initial_audit/image_inventory.csv')
    tbx_hashes = [(r['path'], int(r['dhash64'], 16)) for r in inventory]
    # Una coincidencia perceptual es un candidato a revisar, nunca una etiqueta.
    # Se conserva toda fuente externa en reserva, incluso sin candidatos.
    rows, candidates, summaries = [], [], {}
    for name, source in SOURCES.items():
        directory = project / 'external_data' / name
        paths = sorted((directory / 'CXR_png').glob('*.png'))
        assert len(paths) == source['expected'], (name, len(paths))
        index = (directory / 'CXR_png/index.html').read_text(encoding='utf-8')
        expected_sizes = dict((a, int(b)) for a, b in re.findall(
            r"href='([^']+\.png)'[^<]*>[^<]+</a></td><td class='size'>(\d+)", index))
        assert len(expected_sizes) == source['expected']
        local = []
        for i, path in enumerate(paths, 1):
            assert path.stat().st_size == expected_sizes[path.name], path
            match = re.fullmatch(r'(MCUCXR|CHNCXR)_(\d+)_(0|1)\.png', path.name)
            assert match, path
            label = int(match[3])
            reading = directory / 'ClinicalReadings' / (path.stem + '.txt')
            assert reading.is_file() and reading.stat().st_size > 0, reading
            # Descompresión completa, una imagen por vez; no genera imágenes de evaluación.
            with Image.open(path) as image:
                image.load()
                mode, width, height = image.mode, image.width, image.height
                # P contiene una paleta PNG; convert('RGB') la aplica sin
                # truncar intensidades, a diferencia de originales de 16 bits.
                assert mode in {'L', 'RGB', 'P'}, (path, mode, 'Requiere conversión de intensidad explícita')
                code = dhash(image)
                small = image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
                h = hashlib.sha256(b'512x512:RGB:'); h.update(small.tobytes())
                resized_rgb = h.hexdigest()
            matches = [(p, (code ^ value).bit_count()) for p, value in tbx_hashes if (code ^ value).bit_count() <= 4]
            for p, distance in matches:
                candidates.append({'external_dataset': name, 'external_image': path.name,
                                   'tbx11k_path': p, 'dhash_hamming': distance,
                                   'status': 'candidate_requires_review_not_confirmed'})
            row = {'dataset': name, 'relative_path': path.relative_to(project / 'external_data').as_posix(),
                   'source_url': source['base'] + 'CXR_png/' + path.name,
                   'binary_label': label, 'label_meaning': 'normal' if label == 0 else 'abnormal_tb_consistent',
                   'label_source': source['readme'] + ': documented filename convention 0/1',
                   'split': 'external_reserved', 'width': width, 'height': height, 'mode': mode,
                   'bytes': path.stat().st_size, 'sha256': sha256(path),
                   'resized512_bilinear_rgb_sha256': resized_rgb,
                   'dhash64': f'{code:016x}', 'clinical_reading_sha256': sha256(reading),
                   'dhash_candidates_tbx11k': len(matches), 'eligible_for_evaluation': False,
                   'eligibility_reason': 'Reserved until overlap review and evaluation protocol approval'}
            local.append(row); rows.append(row)
            if i % 100 == 0 or i == len(paths):
                print(name, 'validadas', i, '/', len(paths), flush=True)
            time.sleep(0.01)
        counts = dict(Counter(r['binary_label'] for r in local))
        assert counts == source['counts'], (name, counts)
        summaries[name] = {'images': len(local), 'normal': counts[0], 'abnormal_tb_consistent': counts[1],
                           'clinical_readings': len(local), 'readme_sha256': sha256(directory / source['readme']),
                           'images_with_overlap_candidates': sum(r['dhash_candidates_tbx11k'] > 0 for r in local),
                           'source_url': source['base'], 'image_modes': dict(Counter(r['mode'] for r in local)),
                           'unique_file_sha256': len({r['sha256'] for r in local})}
        write_csv(project / 'manifests/external' / f'{name}.csv', local)
    write_csv(out / 'image_inventory.csv', rows)
    if candidates:
        write_csv(out / 'overlap_candidates.csv', candidates)
    summary = {'status': 'prepared_and_reserved', 'datasets': summaries, 'total_images': len(rows),
               'total_clinical_readings': len(rows), 'decoding_errors': 0,
               'all_download_sizes_match_official_index': True,
               'excluded_from_tbx11k_splits': True, 'training_executed': False,
               'eligible_for_evaluation': False,
               'overlap_search': 'All 800 external dHash64 vs 12278 frozen TBX11K hashes; distance <=4; candidates only, non-exhaustive.',
               'known_limitation': 'TBX11K includes resized external sources. Do not claim independent evaluation until overlap review.',
               'outputs_sha256': {p.name: sha256(p) for p in sorted((project / 'manifests/external').glob('*.csv'))}}
    (out / 'preparation_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    validate_external(ROOT)

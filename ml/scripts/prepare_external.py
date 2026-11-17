"""Descarga reanudable de las fuentes NLM originales; sin modelos ni entrenamiento.

Solo descarga URLs del directorio oficial de cada conjunto. Conserva lecturas y
documentos de uso. Los originales quedan separados del TBX11K congelado.
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/'
SOURCES = {
    'montgomery': {'base': BASE + 'Montgomery-County-CXR-Set/MontgomerySet/',
                   'readme': 'NLM-MontgomeryCXRSet-ReadMe.pdf', 'expected': 138,
                   'counts': {0: 80, 1: 58}},
    'shenzhen': {'base': BASE + 'Shenzhen-Hospital-CXR-Set/',
                 'readme': 'NLM-ChinaCXRSet-ReadMe.docx', 'expected': 662,
                 'counts': {0: 326, 1: 336}},
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.hrefs.extend(v for k, v in attrs if k == 'href')


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                expected = response.headers.get('Content-Length')
                partial = target.with_suffix(target.suffix + '.partial')
                with partial.open('wb') as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if expected and partial.stat().st_size != int(expected):
                    raise ValueError('Descarga incompleta: ' + url)
                partial.replace(target)
            return target
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def listing(base, destination, suffix):
    page = fetch(base + 'index.html', destination / 'index.html')
    parser = Links(); parser.feed(page.read_text(encoding='utf-8'))
    links = []
    for href in parser.hrefs:
        url = urljoin(base, href)
        name = unquote(urlparse(url).path.rsplit('/', 1)[-1])
        if url.startswith(base) and name.lower().endswith(suffix) and '/' not in name and '\\' not in name:
            links.append((url, destination / name))
    return sorted(set(links))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    tasks = []
    for name, source in SOURCES.items():
        directory = ROOT / 'external_data' / name
        if args.download or args.metadata_only:
            fetch(source['base'] + 'index.html', directory / 'source_index.html')
            fetch(source['base'] + source['readme'], directory / source['readme'])
            images = listing(source['base'] + 'CXR_png/', directory / 'CXR_png', '.png')
            readings = listing(source['base'] + 'ClinicalReadings/', directory / 'ClinicalReadings', '.txt')
            assert len(images) == source['expected'], (name, len(images))
            assert len(readings) == source['expected'], (name, len(readings))
            print(name, len(images), 'radiografías y', len(readings), 'lecturas oficiales', flush=True)
            if args.download:
                tasks.extend(images + readings)
    if args.metadata_only:
        return
    if tasks:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, url, path) for url, path in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 40 == 0 or i == len(tasks):
                    print('Descargas', i, '/', len(tasks), flush=True)
    from validate_external import validate_external
    validate_external(ROOT)


if __name__ == '__main__':
    main()

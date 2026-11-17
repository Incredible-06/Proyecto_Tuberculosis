"""Contrasta las nuevas entregas de Colab sin ejecutar notebooks, modelos o CUDA."""
import argparse
import base64
import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'reports/colab_20261021'
ARCHIVE = ROOT / 'results/colab_20261021'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stdout(cell):
    return ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', []) if o['output_type'] == 'stream')


def local_time(run):
    raw = run.split('_', 1)[1]
    return datetime.strptime(raw, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc).astimezone(
        timezone(timedelta(hours=-5))).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming', type=Path, default=Path(r'C:\Users\andyr\Desktop'))
    args = parser.parse_args()
    ARCHIVE.mkdir(parents=True, exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    incoming = args.incoming
    notebook_results, notebooks = {}, {}
    for name in ['00_configuracion_entorno.ipynb', '01_exploracion_preprocesamiento.ipynb']:
        source = incoming / name
        destination = ARCHIVE / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        nb = load(destination); reference = load(ROOT / 'notebooks' / name)
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        expected = [c for c in reference['cells'] if c['cell_type'] == 'code']
        assert len(code) == len(expected), name
        assert [''.join(c['source']).strip() for c in code] == [''.join(c['source']).strip() for c in expected], name
        assert [c['execution_count'] for c in code] == list(range(1, len(code) + 1)), name
        assert all(c.get('outputs') for c in code), name
        assert not any(o['output_type'] == 'error' for c in code for o in c['outputs']), name
        notebooks[name] = {c['id']: c for c in code}
        notebook_results[name] = {'code_cells': len(code), 'executed_in_order': True,
                                  'code_matches_current_reference': True, 'errors': 0,
                                  'sha256': digest(destination.read_bytes())}
    source_zip = incoming / 'drive-download-20261022T000213Z-1-001.zip'
    target_zip = ARCHIVE / source_zip.name
    if source_zip.resolve() != target_zip.resolve(): shutil.copy2(source_zip, target_zip)
    meta = load(ROOT / 'manifests/manifest_metadata.json')
    with zipfile.ZipFile(target_zip) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all('..' not in Path(name).parts and not Path(name).is_absolute() for name in names)
        env_run = next(name.split('/')[0] for name in names if name.endswith('/environment_setup.json'))
        data_run = next(name.split('/')[0] for name in names if name.endswith('/data_checks.json'))
        def read(run, name): return json.loads(archive.read(run + '/' + name))
        env = read(env_run, 'environment_setup.json')
        env_integrity = read(env_run, 'transfer_integrity.json')
        data_integrity = read(data_run, 'integrity.json')
        environment = read(data_run, 'environment.json')
        checks = read(data_run, 'data_checks.json')
        external = read(data_run, 'external_preparation_summary.json')
        assert env['in_colab'] and env['cuda_available'] and env['cuda_probe']['status'] == 'passed'
        assert env['cuda_probe']['executed'] and env['training_executed'] is False
        assert environment['in_colab'] and not environment['cuda_available']
        assert checks['dataloader_check'] == {'executed': True, 'shape': [4, 3, 512, 512], 'device': 'cpu'}
        assert len(checks['checked_samples']) == 6
        for integrity in [env_integrity, data_integrity]:
            assert integrity['status'] == 'passed' and integrity['all_image_hashes_verified'] == 12278
            assert integrity['counts'] == meta['counts']
            assert len(integrity['overlap_checks']) == 12 and all(v == 0 for v in integrity['overlap_checks'].values())
        verified_csv = []
        for name, expected in meta['output_sha256'].items():
            raw = archive.read(data_run + '/manifests/' + name)
            assert digest(raw) == expected == digest((ROOT / 'manifests' / name).read_bytes())
            verified_csv.append(name)
        for run in [env_run, data_run]:
            assert read(run, 'config.json') == load(ROOT / 'configs/data.json')
            assert read(run, 'manifest_metadata.json' if run == env_run else 'manifests/manifest_metadata.json') == meta
        assert read(data_run, 'external_preparation_summary.json') == load(ROOT / 'reports/external_sources/preparation_summary.json')
        for name, expected in external['outputs_sha256'].items():
            assert digest(archive.read(data_run + '/manifests/external/' + name)) == expected
            assert digest((ROOT / 'manifests/external' / name).read_bytes()) == expected
        assert archive.read(data_run + '/review_decisions.csv') == (ROOT / 'reports/near_duplicates/review_decisions.csv').read_bytes()
        assert environment['manifest_metadata_sha256'] == digest((ROOT / 'manifests/manifest_metadata.json').read_bytes())
        assert environment['manifest_csv_sha256'] == meta['output_sha256']
        env_cells = notebooks['00_configuracion_entorno.ipynb']
        data_cells = notebooks['01_exploracion_preprocesamiento.ipynb']
        assert json.loads(stdout(env_cells['env-010'])) == {k: v for k, v in env.items() if k != 'output_paths'}
        assert json.loads(stdout(data_cells['cell-014'])) == data_integrity
        assert json.loads(stdout(data_cells['cell-006'])) == {k: v for k, v in environment.items()
                                                               if not k.startswith('manifest_')}
        assert env_run in stdout(env_cells['F1-E2']) and data_run in stdout(data_cells['F2-E5'])
        assert 'Prueba CUDA: passed' in stdout(env_cells['F1-E1'])
        assert 'Hashes completos: 12278' in stdout(data_cells['F2-E2'])
        # Los PNG mostrados por los notebooks deben ser los de la carpeta exportada.
        figure_checks = {}
        for filename, identifier, output_index in [
            ('class_distribution.png', 'cell-018', 0),
            ('training_samples.png', 'F2-E3', 0),
            ('training_transformations.png', 'F2-E4', 0),
        ]:
            outputs = [o['data']['image/png'] for o in data_cells[identifier]['outputs']
                       if 'image/png' in o.get('data', {})]
            raw_display = outputs[output_index]
            raw_display = ''.join(raw_display) if isinstance(raw_display, list) else raw_display
            raw_export = archive.read(data_run + '/figures/' + filename)
            with Image.open(io.BytesIO(base64.b64decode(raw_display))) as a, Image.open(io.BytesIO(raw_export)) as b:
                assert a.size == b.size and ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None
            figure_checks[filename] = {'matches_displayed_pixels': True, 'sha256': digest(raw_export)}
            target = REVIEW / 'figures' / filename; target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw_export)
        for run, filenames in [(env_run, ['environment_setup.json', 'transfer_integrity.json', 'config.json', 'manifest_metadata.json']),
                               (data_run, ['environment.json', 'integrity.json', 'data_checks.json', 'external_preparation_summary.json'])]:
            folder = REVIEW / run; folder.mkdir(exist_ok=True)
            for filename in filenames: (folder / filename).write_bytes(archive.read(run + '/' + filename))
    report = {'status': 'passed', 'review_date_lima': '2026-10-21',
              'source_archive': target_zip.relative_to(ROOT).as_posix(), 'source_archive_sha256': digest(target_zip.read_bytes()),
              'notebooks': notebook_results, 'environment_run': env_run, 'environment_run_lima': local_time(env_run),
              'data_run': data_run, 'data_run_lima': local_time(data_run),
              'gpu': env['gpu']['name'], 'cuda_probe': 'passed',
              'complete_image_hashes_in_both_runs': 12278, 'internal_overlap_checks_zero': 12,
              'seven_frozen_csvs_match': verified_csv, 'external_manifest_hashes_match': external['outputs_sha256'],
              'external_summary_matches_local': True, 'external_original_images_checked_in_colab': False,
              'dataloader_check': checks['dataloader_check'], 'figures': figure_checks,
              'training_executed': False, 'local_notebook_or_cuda_execution_by_review': False,
              'scope': 'Archivos entregados por el usuario; ejecución Colab registrada, sin observación de una sesión en vivo.'}
    (REVIEW / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

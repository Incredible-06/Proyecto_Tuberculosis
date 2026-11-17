"""Revisa evidencias entregadas de Colab sin ejecutar el notebook o CUDA."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    incoming = ROOT / 'reports/temp_salida_notebook00'
    output = ROOT / 'reports/environment/colab_review'
    output.mkdir(parents=True, exist_ok=True)
    notebook = incoming / '00_configuracion_entorno.ipynb'
    archives = list(incoming.glob('*.zip'))
    assert len(archives) == 1, 'Se esperaba un ZIP de entorno.'
    archive = archives[0]
    delivered = json.loads(notebook.read_text(encoding='utf-8'))
    # La ejecución histórica debe contrastarse con su versión de código,
    # no con los resúmenes añadidos para futuras ejecuciones.
    reference = ROOT / 'reports/revision_20261021/referencias/00_configuracion_entorno.ipynb'
    original = json.loads(reference.read_text(encoding='utf-8'))
    code = [c for c in delivered['cells'] if c['cell_type'] == 'code']
    source = [c for c in original['cells'] if c['cell_type'] == 'code']
    assert len(code) == len(source) == 7
    assert [c['source'] for c in code] == [c['source'] for c in source]
    assert [c.get('execution_count') for c in code] == list(range(1, 8))
    assert not any(o['output_type'] == 'error' for c in code for o in c.get('outputs', []))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = {Path(n).name: n for n in z.namelist() if not n.endswith('/')}
        assert set(names) == {'config.json', 'manifest_metadata.json', 'environment_setup.json', 'transfer_integrity.json'}
        data = {key: json.loads(z.read(name)) for key, name in names.items()}
        run = Path(names['environment_setup.json']).parent.name
    config = json.loads((ROOT / 'configs/data.json').read_text(encoding='utf-8'))
    meta = json.loads((ROOT / 'manifests/manifest_metadata.json').read_text(encoding='utf-8'))
    assert data['config.json'] == config
    assert data['manifest_metadata.json'] == meta
    for name, expected in {**meta['input_sha256'], **{'manifests/' + k: v for k, v in meta['output_sha256'].items()}}.items():
        assert sha(ROOT / name) == expected, name
    env = data['environment_setup.json']
    integrity = data['transfer_integrity.json']
    stream = ''.join(''.join(o.get('text', [])) for o in code[4]['outputs'])
    assert json.loads(stream) == {k: v for k, v in env.items() if k != 'output_paths'}
    assert env['in_colab'] and env['cuda_available'] and env['cuda_probe']['executed']
    assert env['cuda_probe']['status'] == 'passed' and env['training_executed'] is False
    assert integrity['status'] == 'passed' and integrity['all_image_hashes_verified'] == 12278
    assert integrity['counts'] == meta['counts']
    assert integrity['supervised_images'] == meta['unique_labeled_images']
    assert integrity['supervised_groups'] == meta['duplicate_groups_internal']
    assert len(integrity['overlap_checks']) == 12 and all(v == 0 for v in integrity['overlap_checks'].values())
    for name, value in data.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    proof = {'status': 'passed', 'run': run, 'run_local_time': '2026-10-20 00:11:06 America/Lima',
             'source_notebook_sha256': sha(notebook), 'source_zip_sha256': sha(archive),
             'executed_code_cells': 7, 'code_matches_original': True, 'notebook_errors': 0,
             'reference_notebook': reference.relative_to(ROOT).as_posix(),
             'comparison_scope': 'Código de la versión histórica, anterior a la revisión de presentación del 21 de octubre.',
             'configuration_matches': True, 'manifest_metadata_matches': True,
             'local_frozen_inputs_and_csv_fingerprints_match': True,
             'gpu': env['gpu']['name'], 'cuda_probe': env['cuda_probe']['status'],
             'all_image_hashes_verified_in_colab': integrity['all_image_hashes_verified'],
             'internal_overlap_checks_zero': 12, 'local_training_or_cuda_executed_by_review': False,
             'scope': 'Revisión de archivos entregados; no observación de sesión en vivo ni validación de modelos.'}
    (output / 'verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

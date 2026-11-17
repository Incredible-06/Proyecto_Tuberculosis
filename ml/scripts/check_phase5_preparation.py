"""Verifica entrega local; no carga imágenes ni checkpoints, no usa GPU."""
import ast
import io
import json
import sys
import tempfile
import unittest
import zipfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tbxdata.common import sha256, write_csv
from tbxdata.validate import validate_manifests
from tbxtrain.phase5 import validation_rows, verify_provenance, read_validation_predictions, tb_threshold_table
from tbxtrain.protocol import atomic_json
from tbxtrain.phase5_figures import export_validation_figures


def main():
    output = ROOT/'reports/phase5_preparation'
    output.mkdir(parents=True,exist_ok=True)
    previous_notebooks = {p.name:sha256(p) for p in (ROOT/'notebooks').glob('0[2345]_*.ipynb')}
    notebook = ROOT/'notebooks/06_fase5_seleccion_explicabilidad_evaluacion.ipynb'
    payload = json.loads(notebook.read_text(encoding='utf-8'))
    for cell in payload['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),cell['id'],'exec')
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Esta revisión espera una plantilla sin ejecutar.')
    assignments = {}
    for cell in payload['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                    for target in node.targets:
                        if isinstance(target,ast.Name): assignments[target.id] = node.value.value
    info = json.loads((ROOT/'dist/phase5_package_info.json').read_text())
    archive = ROOT/'dist'/info['file']
    if sha256(archive) != info['sha256'] or assignments['EXPECTED_PACKAGE_SHA256'] != info['sha256']:
        raise ValueError('El notebook y el ZIP no son de la misma entrega.')
    summaries = [json.loads(p.read_text()) for p in
                 (ROOT/'reports/phase5_readiness/drive_evidence').glob('*_summary.json')
                 if p.name in {'convnext_summary.json','vmamba_summary.json','hybrid_summary.json'}]
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
        extracted = Path(temp).resolve()
        if (ROOT/'tmp').resolve() not in extracted.parents: raise ValueError('Temporal fuera del proyecto.')
        from tbxdata.colab import safe_extract_zip
        with zipfile.ZipFile(archive) as z:
            hashes = json.loads(z.read('phase5_package_checksums.json'))
            if set(z.namelist()) != set(hashes) | {'phase5_package_checksums.json'}:
                raise ValueError('Cobertura de ZIP incorrecta.')
            if any(n.endswith(('.ipynb','.pt','.png','.jpg')) for n in z.namelist()):
                raise ValueError('Paquete contiene artefactos excluidos.')
        safe_extract_zip(archive,extracted)
        for name,expected in hashes.items():
            if sha256(extracted/name) != expected: raise ValueError(name)
        validate_manifests(extracted)
        for s in summaries: verify_provenance(extracted,s['provenance'])
    rows = validation_rows(ROOT)
    comparisons = []
    for short in ['convnext','vmamba','hybrid']:
        folder = ROOT/'reports/phase5_readiness/drive_evidence'
        s = json.loads((folder/(short+'_summary.json')).read_text())
        _,y,p,m = read_validation_predictions(folder/(short+'_validation_predictions.csv'),rows)
        comparisons.append(dict(architecture=s['architecture'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                                TB_recall=m['per_class']['TB']['recall'],source='existing_validation_predictions'))
        if short == 'hybrid':
            write_csv(output/'hybrid_TB_thresholds_existing_validation.csv',tb_threshold_table(y,p))
    write_csv(output/'comparison_existing_validation.csv',comparisons)
    config = json.loads((ROOT/'configs/phase5_preparation.json').read_text())
    evidence = ROOT/'reports/phase5_readiness/drive_evidence'
    source_files = {}
    for short,spec in zip(['convnext','vmamba','hybrid'],config['classifiers']):
        source_files[spec['run_id']] = {key:evidence/(short+'_'+suffix) for key,suffix in
            [('summary','summary.json'),('history','history.csv'),('predictions','validation_predictions.csv')]}
    source_files[config['detector']['run_id']] = dict(history=evidence/'detector_results.csv',
        summary=evidence/'detector_detector_summary.json')
    figures_dir = output/('figures_existing_validation_'+uuid.uuid4().hex[:8])
    manifest_path = export_validation_figures(ROOT,ROOT,config['classifiers'],figures_dir,
        detector_run_id=config['detector']['run_id'],source_files=source_files)
    figure_manifest = json.loads(manifest_path.read_text())
    if len(figure_manifest['figures']) != 12: raise ValueError('Figuras faltantes.')
    for name,digest in figure_manifest['output_sha256'].items():
        if sha256(figures_dir/name) != digest: raise ValueError('Figura alterada.')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_phase5_preparation.py')
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    unchanged = previous_notebooks == {p.name:sha256(p) for p in (ROOT/'notebooks').glob('0[2345]_*.ipynb')}
    report = dict(scope='local_CPU_checks_existing_validation_only',tests_run=result.testsRun,
        tests_passed=result.wasSuccessful(),package_verified=True,original_provenance_verified_in_package=True,
        previous_local_notebooks_unchanged_during_check=unchanged,previous_local_notebook_sha256=previous_notebooks,
        new_notebook_sha256=sha256(notebook),package=info,source_predictions='reports/phase5_readiness/drive_evidence',
        figures_exported=len(figure_manifest['figures']),figures_directory=str(figures_dir),
        figure_formats=['png_300dpi','pdf','svg','csv'],
        new_gpu_inference_executed=False,colab_executed=False,decisions_frozen=False,
        test_evaluated=False,external_evaluated=False,phase5_complete=False)
    atomic_json(output/'validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'previous_local_notebook_sha256','package'}},indent=2))
    if not result.wasSuccessful() or not unchanged: raise SystemExit(1)


if __name__ == '__main__': main()

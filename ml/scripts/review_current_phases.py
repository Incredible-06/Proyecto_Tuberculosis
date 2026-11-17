"""Contrasta evidencia histórica, revisión local y fuentes externas sin entrenar."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tbxdata.validate import validate_manifests
from tbxdata.common import sha256


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    out = ROOT / 'reports/revision_20261021'; out.mkdir(parents=True, exist_ok=True)
    integrity = validate_manifests(ROOT, ROOT / 'TBX11K', hash_samples_per_class=3)
    historical = {}
    for name, executed_path in [
        ('00_configuracion_entorno.ipynb', 'reports/temp_salida_notebook00/00_configuracion_entorno.ipynb'),
        ('01_exploracion_preprocesamiento.ipynb', 'results/01_exploracion_preprocesamiento.ipynb'),
    ]:
        delivered = read(executed_path)
        reference = read('reports/revision_20261021/referencias/' + name)
        code = [c for c in delivered['cells'] if c['cell_type'] == 'code']
        original = [c for c in reference['cells'] if c['cell_type'] == 'code']
        assert [c['source'] for c in code] == [c['source'] for c in original]
        assert all(c['execution_count'] is not None for c in code)
        assert not any(o['output_type'] == 'error' for c in code for o in c['outputs'])
        historical[name] = {'executed_code_cells': len(code), 'code_matches_historical_reference': True,
                            'errors': 0, 'executed_file_sha256': sha256(ROOT / executed_path)}
    tests = read('reports/phase1/tests.json'); assert tests['passed'] and tests['tests_run'] == 9
    meta = read('manifests/manifest_metadata.json')
    historical_csv = []
    with zipfile.ZipFile(ROOT / 'results/results-20261020T043203Z-1-001.zip') as archive:
        assert archive.testzip() is None
        for name, expected in meta['output_sha256'].items():
            matches = [entry for entry in archive.namelist() if entry.endswith('/manifests/' + name)]
            assert len(matches) == 1, (name, matches)
            assert hashlib.sha256(archive.read(matches[0])).hexdigest() == expected
            historical_csv.append(name)
    external = read('reports/external_sources/preparation_summary.json')
    assert external['total_images'] == external['total_clinical_readings'] == 800
    for name, expected in external['outputs_sha256'].items():
        assert sha256(ROOT / 'manifests/external' / name) == expected
    assert read('reports/revision_20261021/drive_files/env_setup.json') == read('reports/environment/colab_review/environment_setup.json')
    assert read('reports/revision_20261021/drive_files/env_integrity.json') == read('reports/environment/colab_review/transfer_integrity.json')
    second = read('reports/revision_20261021/drive_files/data_integrity_0436.json')
    assert second['status'] == 'passed' and second['all_image_hashes_verified'] == 12278
    assert second['counts'] == integrity['counts']
    assert all(v == 0 for v in second['overlap_checks'].values())
    assert read('reports/revision_20261021/drive_files/data_checks_0436.json')['dataloader_check']['shape'] == [4, 3, 512, 512]
    guides = []
    revised = {}
    latest_path = ROOT / 'reports/colab_20261021/verification.json'
    latest = read('reports/colab_20261021/verification.json') if latest_path.is_file() else None
    if latest: assert latest['status'] == 'passed'
    for name in historical:
        nb = read('notebooks/' + name)
        assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        for cell in code:
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == [] and cell['execution_count'] is None
        revised[name] = {'code_cells': len(code), 'syntax_passed': True, 'new_colab_execution': False}
        if latest:
            delivered_path = ROOT / 'results/colab_20261021' / name
            delivered = json.loads(delivered_path.read_text(encoding='utf-8'))
            delivered_code = [c for c in delivered['cells'] if c['cell_type'] == 'code']
            assert [''.join(c['source']).strip() for c in code] == [''.join(c['source']).strip() for c in delivered_code]
            assert sha256(delivered_path) == latest['notebooks'][name]['sha256']
            revised[name]['new_colab_execution'] = True
            revised[name]['new_colab_errors'] = latest['notebooks'][name]['errors']
        execution = read('reports/' + ('environment' if name.startswith('00') else 'phase1') + '/notebook_validation.json')
        assert execution['local_execution'] and execution['code_cells_compiled'] == len(code)
        revised[name]['local_execution_passed'] = True
        for ordinal, cell in enumerate(nb['cells'], 1):
            if cell['cell_type'] == 'code' and 'evidencia' in cell['metadata'].get('tags', []):
                code_ordinal = sum(c['cell_type'] == 'code' for c in nb['cells'][:ordinal])
                guides.append((name, cell['id'], ordinal, code_ordinal))
    report = {'status': 'passed', 'review_date_lima': '2026-10-21',
              'phase1': 'verified_basic_environment', 'phase2': 'prepared_tbx11k_and_full_external_sources_reserved',
              'local_integrity': integrity, 'historical_notebooks': historical, 'revised_notebooks': revised,
              'local_tests': tests, 'historical_csv_hashes_match': historical_csv,
              'drive_environment_matches_local': True,
              'drive_second_data_run_full_hashes': 12278, 'external_images': 800,
              'external_independent_evaluation_enabled': False, 'training_executed': False,
              'latest_colab_review': 'reports/colab_20261021/verification.json' if latest else None,
              'limits': ['No patient independence guarantee.', 'Near-duplicate search is not exhaustive.',
                         'External overlap candidates require review before independent evaluation.',
                         'Classification loader only; detector geometric adaptation is a later step.'] +
                        ([] if latest else ['Revised notebooks require their own Colab execution for new screenshots.'])}
    (out / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    text = '''# Guía para las capturas del informe

Use los notebooks de la carpeta `notebooks`, incluidos en el ZIP nuevo de `dist`.
Los nombres 00 y 01 son el orden de ejecución; las fases del curso son 1 entorno y 2 datos.
La columna “celda total” cuenta todas las celdas desde 1, incluidas las de texto.
La columna “código” cuenta solo las celdas ejecutables. **Busque el identificador del título**:
el número `[n]` que muestra Colab es un contador de ejecución y puede ser distinto.

## Antes de capturar

1. Sustituya en Drive el paquete pequeño `tbx11k_fases_1_y_2_colab.zip` por el de `dist`.
   El dataset `TBX11K.zip` permanece igual. El paquete no incluye imágenes ni pesos.
2. Abra los notebooks revisados en Colab. En 00 seleccione GPU; 01 puede ejecutarse en CPU.
3. Ejecute todas las celdas en orden. Las nuevas salidas deben pertenecer a esa ejecución.
4. Capture el título y la salida completa del bloque; puede ocultar el código para legibilidad.
   Descargue también el `.ipynb` ejecutado y conserve la carpeta nueva de resultados en Drive.

## Celdas exactas

| Notebook | Bloque | Celda total | Código | Qué debe verse |
|---|---|---:|---:|---|
'''
    descriptions = {'F1-E1': 'GPU, PyTorch, CUDA passed, seed y ausencia de entrenamiento',
        'F1-E2': '12.278 hashes completos, cero cruces y ruta persistente en Drive',
        'F2-E1': 'Tres clases y motivo de agrupación de los subtipos TB',
        'F2-E2': '5.792 / 1.241 / 1.241 y doce cruces conocidos en cero',
        'F2-E3': 'Muestras de train y cajas originales; se puede usar la imagen exportada',
        'F2-E4': 'Aumentos solo en train, normalización y comparación visual',
        'F2-E5': 'Seis muestras, lote CPU 4×3×512×512, 1.211 cajas y tb1199',
        'F2-E6': '138 Montgomery + 662 Shenzhen y reserva externa; resumen registrado localmente'}
    for name, identifier, ordinal, code_ordinal in guides:
        text += f'| {name[:2]} | {identifier} | {ordinal} | {code_ordinal} | {descriptions[identifier]} |\n'
    text += '''
## Selección breve recomendada

Use F1-E1, F1-E2, F2-E2, F2-E3, F2-E4 y F2-E6. F2-E1 y F2-E5 son complementarias.
Las dos capturas de Drive ya aportadas pueden acompañar la evidencia del entorno.
El informe revisado contiene una captura real de Drive y salidas gráficas auténticas de la ejecución histórica.
No se han fabricado capturas de Colab ni insertado resultados históricos en celdas nuevas.

F2-E6 muestra la preparación externa registrada y comprueba los manifiestos del paquete;
no afirma que las 800 imágenes estén cargadas o verificadas en la sesión actual de Colab.
Para reconstruir esas fuentes desde sus originales oficiales, ejecute `python scripts/prepare_external.py --download`
en un entorno con red y NumPy/Pillow. Descarga las fuentes aparte y regenera únicamente los manifiestos externos.
El paquete pequeño no incluye los originales externos; están en `external_data` local.

## Qué resultados no deben confundirse

- Las 7 y 14 celdas históricas corresponden a versiones anteriores, conservadas por trazabilidad.
- Esta revisión contiene 9 y 20 celdas de código. Se probaron localmente; sus nuevas salidas Colab aún deben generarse.
- Si CUDA no muestra passed o los hashes completos no llegan a 12.278, esa nueva ejecución no acredita esos puntos.
- experiments y checkpoints vacíos son normales antes del entrenamiento.
- Cero cruces de grupos conocidos no significa independencia por paciente ni ausencia exhaustiva de duplicados.
'''
    if latest:
        text = text.replace('## Antes de capturar', '## Para repetir capturas en futuras ejecuciones')
        text = text.replace('El informe revisado contiene una captura real de Drive y salidas gráficas auténticas de la ejecución histórica.',
            'El informe revisado contiene recortes de las capturas de Colab entregadas y salidas gráficas de la nueva ejecución del 21 de octubre.')
        text = text.replace('Se probaron localmente; sus nuevas salidas Colab aún deben generarse.',
            'Se probaron localmente y se verificó la nueva entrega Colab: todas las celdas se ejecutaron en orden y sin errores.')
        text = text.replace('# Guía para las capturas del informe\n', '# Guía para las capturas del informe\n\n'
            '**Evidencias recibidas y verificadas el 21 de octubre de 2026.** Las capturas ya se incorporaron al informe. '
            'No hace falta repetir la ejecución para respaldar estas dos fases. El registro está en `reports/colab_20261021/verification.json`.\n')
    (ROOT / 'docs/GUIA_CAPTURAS.md').write_text(text, encoding='utf-8', newline='\n')
    print('Revisión aprobada; guía de', len(guides), 'celdas generada.')


if __name__ == '__main__':
    main()

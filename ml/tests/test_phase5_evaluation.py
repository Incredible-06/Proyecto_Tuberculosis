"""Pruebas unitarias locales para contrato congelado y evaluación final de Fase 5."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tbxdata.common import sha256, write_csv
from tbxtrain.phase5_evaluation import (
    _bootstrap_frozen_test,
    evaluate_detector_on_split,
    export_deployment_bundle,
    verify_frozen_contract,
)


class Phase5EvaluationTests(unittest.TestCase):
    def test_frozen_contract_verifies_and_detects_tampering(self):
        contract = verify_frozen_contract(ROOT)
        self.assertTrue(contract['decisions_frozen'])
        self.assertEqual(contract['classifier']['architecture'], 'vit_b_16')
        self.assertEqual(contract['classifier']['operational_tb_threshold'], 0.3)
        self.assertEqual(contract['detector']['confidence_threshold'], 0.25)
        self.assertFalse(contract['concordance_rule']['suppress_classifier_without_boxes'])

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            bad_path = Path(temp) / 'bad_contract.json'
            bad = json.loads(json.dumps(contract))
            bad['code_hashes']['src/tbxtrain/phase5_evaluation.py'] = '0' * 64
            bad_path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Código alterado'):
                verify_frozen_contract(ROOT, bad_path)

            bad2 = json.loads(json.dumps(contract))
            bad2['concordance_rule']['suppress_classifier_without_boxes'] = True
            bad_path.write_text(json.dumps(bad2), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'suprimir alertas'):
                verify_frozen_contract(ROOT, bad_path)

    def test_evaluate_detector_on_test_split_and_bootstrap(self):
        rows = [
            {'split': 'test', 'relative_path': 'imgs/health/h1.png', 'width': '100', 'height': '100',
             'class_name': 'health', 'label': '0', 'group_id': 'g1', 'boxes_json': '[]'},
            {'split': 'test', 'relative_path': 'imgs/sick/s1.png', 'width': '100', 'height': '100',
             'class_name': 'sick', 'label': '1', 'group_id': 'g2', 'boxes_json': '[]'},
            {'split': 'test', 'relative_path': 'imgs/tb/t1.png', 'width': '100', 'height': '100',
             'class_name': 'TB', 'label': '2', 'group_id': 'g3', 'boxes_json': json.dumps([{'bbox': [10, 10, 20, 20]}])},
            {'split': 'test', 'relative_path': 'imgs/tb/t2.png', 'width': '100', 'height': '100',
             'class_name': 'TB', 'label': '2', 'group_id': 'g4', 'boxes_json': json.dumps([{'bbox': [30, 30, 20, 20]}])},
        ]
        preds = {
            'imgs/health/h1.png': [],
            'imgs/sick/s1.png': [{'xyxy': [5, 5, 25, 25], 'confidence': 0.30}],
            'imgs/tb/t1.png': [{'xyxy': [10, 10, 30, 30], 'confidence': 0.80}],
            'imgs/tb/t2.png': [],
        }
        res = evaluate_detector_on_split(rows, preds, [0.25], expected_split='test', match_iou=0.5)[0]
        self.assertEqual(res['matched_boxes'], 1)
        self.assertEqual(res['false_positive_boxes'], 1)
        self.assertAlmostEqual(res['lesion_recall'], 0.5)
        self.assertAlmostEqual(res['health_alert_rate'], 0.0)
        self.assertAlmostEqual(res['sick_alert_rate'], 1.0)

        probs = np.array([
            [0.90, 0.08, 0.02],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
            [0.10, 0.55, 0.35],
        ])
        boot = _bootstrap_frozen_test(rows, probs, preds, cls_thresh=0.30, det_thresh=0.25, n_resamples=30, seed=11)
        self.assertEqual(boot['n_images'], 4)
        self.assertAlmostEqual(boot['operational_and_concordance']['cls_thresh_sensitivity']['point'], 1.0)
        self.assertAlmostEqual(boot['operational_and_concordance']['combined_tb_sensitivity']['point'], 1.0)

    def test_deployment_bundle_export_verifies_hashes(self):
        contract = verify_frozen_contract(ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            tmp = Path(temp)
            drive = tmp / 'drive'
            cls_file = drive / 'checkpoints' / contract['classifier']['run_id'] / 'best.pt'
            det_file = drive / 'experiments' / contract['detector']['run_id'] / 'training/weights/best.pt'
            cls_file.parent.mkdir(parents=True)
            det_file.parent.mkdir(parents=True)
            cls_file.write_bytes(b'cls-weights')
            det_file.write_bytes(b'det-weights')
            local_contract = json.loads(json.dumps(contract))
            local_contract['classifier']['checkpoint_sha256'] = sha256(cls_file)
            local_contract['detector']['weights_sha256'] = sha256(det_file)

            manifest = export_deployment_bundle(ROOT, drive, local_contract, tmp / 'bundle')
            self.assertEqual(manifest['classifier_sha256'], sha256(cls_file))
            self.assertEqual(manifest['detector_sha256'], sha256(det_file))
            self.assertTrue((tmp / 'bundle/phase5_frozen_contract.json').is_file())

    def test_external_audit_recorded_zero_train_val_overlap(self):
        audit_path = ROOT / 'reports/external_sources/train_val_overlap_audit.json'
        self.assertTrue(audit_path.is_file())
        summary = json.loads(audit_path.read_text(encoding='utf-8'))
        self.assertEqual(summary['montgomery']['total'], 138)
        self.assertEqual(summary['montgomery']['train_val_pixel_overlaps'], 0)
        self.assertEqual(summary['shenzhen']['total'], 662)
        self.assertEqual(summary['shenzhen']['train_val_pixel_overlaps'], 0)

    def test_notebook_08_compiles_and_matches_package(self):
        nb_path = ROOT / 'notebooks/08_fase5_evaluacion_final_test_y_externos.ipynb'
        pkg_path = ROOT / 'dist/phase5_evaluation_package_info.json'
        if not nb_path.is_file() or not pkg_path.is_file():
            self.skipTest('Ejecutar package_phase5_evaluation.py y create_phase5_evaluation_notebook.py primero.')
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        pkg = json.loads(pkg_path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertIn(pkg['sha256'], code)
        self.assertIn('verify_frozen_contract', code)
        self.assertIn('evaluate_frozen_test', code)
        self.assertNotIn('.train(', code)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()

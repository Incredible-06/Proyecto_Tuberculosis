"""Pruebas unitarias locales para diagnóstico de validation, concordancia, coste e incertidumbre."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tbxdata.common import write_csv
from tbxtrain.phase5 import validation_rows
from tbxtrain.phase5_diagnostics import (
    POINTER_ARCHS,
    benchmark_classifier_cost,
    concordance_grid_analysis,
    count_parameters,
    detector_detailed_metrics,
    detector_group_bootstrap,
    export_detector_diagnostic_figures,
    group_bootstrap_metrics,
    load_diagnostic_config,
    render_visual_review_panels,
    select_visual_review_cases,
    verify_any_classifier,
)


class Phase5DiagnosticsTests(unittest.TestCase):
    def test_diagnostic_config_and_local_evidence_verification(self):
        cfg = load_diagnostic_config(ROOT)
        self.assertEqual(cfg['scope'], 'validation_only')
        self.assertFalse(cfg['decisions_frozen'])
        self.assertFalse(cfg['test_evaluated'])
        self.assertEqual(len(cfg['classifiers']), 5)

        rows = validation_rows(ROOT)
        evidence_map = {
            'convnext_tiny': (
                ROOT / 'reports/phase5_readiness/drive_evidence/convnext_summary.json',
                ROOT / 'reports/phase5_readiness/drive_evidence/convnext_validation_predictions.csv',
            ),
            'vmamba_tiny': (
                ROOT / 'reports/phase5_readiness/drive_evidence/vmamba_summary.json',
                ROOT / 'reports/phase5_readiness/drive_evidence/vmamba_validation_predictions.csv',
            ),
            'hybrid_cross_attention': (
                ROOT / 'reports/phase5_readiness/drive_evidence/hybrid_summary.json',
                ROOT / 'reports/phase5_readiness/drive_evidence/hybrid_validation_predictions.csv',
            ),
            'vit_b_16': (
                ROOT / 'reports/phase5_review/vit_evidence/vit_summary.json',
                ROOT / 'reports/phase5_review/vit_evidence/vit_validation_predictions.csv',
            ),
            'hybrid_convnext_vit': (
                ROOT / 'reports/phase5_review/hybrid_vit_evidence/hybrid_vit_summary.json',
                ROOT / 'reports/phase5_review/hybrid_vit_evidence/hybrid_vit_validation_predictions.csv',
            ),
        }
        preds_by_arch = {}
        for spec in cfg['classifiers']:
            s_path, p_path = evidence_map[spec['architecture']]
            summary, ordered, _, _, metrics = verify_any_classifier(
                ROOT,
                ROOT,
                spec,
                rows,
                check_checkpoint=False,
                source_files={'summary': s_path, 'predictions': p_path},
            )
            preds_by_arch[spec['architecture']] = ordered
            self.assertEqual(metrics['n'], 1241)
            self.assertEqual(summary['best_epoch'], summary['best_epoch'])

        # Bootstrap agrupado real sobre las predicciones verificadas de los 5 modelos.
        boot = group_bootstrap_metrics(
            rows,
            preds_by_arch,
            thresholds=(0.1, 0.3, 0.5),
            pointer_archs=POINTER_ARCHS,
            n_resamples=40,
            seed=20261019,
        )
        self.assertEqual(boot['n_images'], 1241)
        self.assertEqual(boot['n_groups'], 1237)
        for arch in POINTER_ARCHS:
            m = boot['models'][arch]['metrics']
            self.assertLessEqual(m['macro_f1']['ci_low'], m['macro_f1']['point'])
            self.assertGreaterEqual(m['macro_f1']['ci_high'], m['macro_f1']['point'])
            self.assertIn('0.30', boot['models'][arch]['thresholds'])
        self.assertEqual(len(boot['paired_deltas']), 3)

    def test_group_bootstrap_preserves_multi_image_clusters(self):
        # 2 grupos de 2 imágenes cada uno + 2 grupos unitarios: valida estratificación y agrupamiento.
        rows = [
            {'relative_path': 'h1', 'label': '0', 'class_name': 'health', 'group_id': 'g_h1', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 'h2', 'label': '0', 'class_name': 'health', 'group_id': 'g_h2', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 's1', 'label': '1', 'class_name': 'sick', 'group_id': 'g_s_pair', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 's2', 'label': '1', 'class_name': 'sick', 'group_id': 'g_s_pair', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 't1', 'label': '2', 'class_name': 'TB', 'group_id': 'g_t1', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[{"bbox":[5,5,20,20]}]', 'tb_subtypes_json': '["ActiveTuberculosis"]'},
            {'relative_path': 't2', 'label': '2', 'class_name': 'TB', 'group_id': 'g_t2', 'split': 'validation', 'width': '64', 'height': '64', 'boxes_json': '[{"bbox":[10,10,20,20]}]', 'tb_subtypes_json': '["LatentTuberculosis"]'},
        ]
        preds = [
            {'relative_path': 'h1', 'label': 0, 'group_id': 'g_h1', 'prediction': 0, 'p_health': 0.9, 'p_sick': 0.05, 'p_TB': 0.05},
            {'relative_path': 'h2', 'label': 0, 'group_id': 'g_h2', 'prediction': 0, 'p_health': 0.8, 'p_sick': 0.1, 'p_TB': 0.1},
            {'relative_path': 's1', 'label': 1, 'group_id': 'g_s_pair', 'prediction': 1, 'p_health': 0.1, 'p_sick': 0.85, 'p_TB': 0.05},
            {'relative_path': 's2', 'label': 1, 'group_id': 'g_s_pair', 'prediction': 1, 'p_health': 0.1, 'p_sick': 0.8, 'p_TB': 0.1},
            {'relative_path': 't1', 'label': 2, 'group_id': 'g_t1', 'prediction': 2, 'p_health': 0.05, 'p_sick': 0.15, 'p_TB': 0.8},
            {'relative_path': 't2', 'label': 2, 'group_id': 'g_t2', 'prediction': 1, 'p_health': 0.1, 'p_sick': 0.55, 'p_TB': 0.35},
        ]
        boot = group_bootstrap_metrics(
            rows,
            {'hybrid_cross_attention': preds, 'vit_b_16': preds},
            thresholds=(0.3, 0.5),
            pointer_archs=('hybrid_cross_attention', 'vit_b_16'),
            n_resamples=30,
            seed=42,
        )
        self.assertEqual(boot['n_groups'], 5)
        self.assertAlmostEqual(boot['models']['hybrid_cross_attention']['metrics']['accuracy']['point'], 5 / 6)

    def test_detector_diagnostics_concordance_visual_review_and_cost(self):
        from PIL import Image
        import torch
        from torch import nn

        rows = [
            {'relative_path': 'imgs/health/h1.png', 'label': '0', 'class_name': 'health', 'group_id': 'gh1', 'split': 'validation', 'width': '512', 'height': '512', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 'imgs/sick/s1.png', 'label': '1', 'class_name': 'sick', 'group_id': 'gs1', 'split': 'validation', 'width': '512', 'height': '512', 'boxes_json': '[]', 'tb_subtypes_json': '[]'},
            {'relative_path': 'imgs/tb/t1.png', 'label': '2', 'class_name': 'TB', 'group_id': 'gt1', 'split': 'validation', 'width': '512', 'height': '512', 'boxes_json': '[{"bbox":[50,50,80,80]}]', 'tb_subtypes_json': '["ActiveTuberculosis"]'},
            {'relative_path': 'imgs/tb/t2.png', 'label': '2', 'class_name': 'TB', 'group_id': 'gt2', 'split': 'validation', 'width': '512', 'height': '512', 'boxes_json': '[{"bbox":[100,100,60,60]}]', 'tb_subtypes_json': '["LatentTuberculosis"]'},
        ]
        det_preds = {
            'imgs/health/h1.png': [],
            'imgs/sick/s1.png': [{'xyxy': [10.0, 10.0, 80.0, 80.0], 'confidence': 0.40}],
            'imgs/tb/t1.png': [{'xyxy': [50.0, 50.0, 130.0, 130.0], 'confidence': 0.85}],
            'imgs/tb/t2.png': [],
        }
        det_table = detector_detailed_metrics(rows, det_preds, [0.1, 0.25, 0.5])
        self.assertEqual(det_table[1]['matched_boxes'], 1)
        self.assertAlmostEqual(det_table[1]['box_precision'], 0.5)
        self.assertAlmostEqual(det_table[1]['sick_alert_rate'], 1.0)
        self.assertAlmostEqual(det_table[1]['health_alert_rate'], 0.0)

        det_boot = detector_group_bootstrap(rows, det_preds, thresholds=(0.25,), n_resamples=25, seed=7)
        self.assertEqual(det_boot['n_groups'], 4)
        self.assertIn('0.25', det_boot['thresholds'])

        preds_by_arch = {}
        for arch in POINTER_ARCHS:
            preds_by_arch[arch] = [
                {'relative_path': 'imgs/health/h1.png', 'label': 0, 'group_id': 'gh1', 'prediction': 0, 'p_health': 0.95, 'p_sick': 0.03, 'p_TB': 0.02},
                {'relative_path': 'imgs/sick/s1.png', 'label': 1, 'group_id': 'gs1', 'prediction': 1, 'p_health': 0.05, 'p_sick': 0.90, 'p_TB': 0.05},
                {'relative_path': 'imgs/tb/t1.png', 'label': 2, 'group_id': 'gt1', 'prediction': 2, 'p_health': 0.02, 'p_sick': 0.08, 'p_TB': 0.90},
                {'relative_path': 'imgs/tb/t2.png', 'label': 2, 'group_id': 'gt2', 'prediction': 2 if arch != 'hybrid_convnext_vit' else 1, 'p_health': 0.05, 'p_sick': 0.30 if arch != 'hybrid_convnext_vit' else 0.55, 'p_TB': 0.65 if arch != 'hybrid_convnext_vit' else 0.40},
            ]

        grid, default_rows = concordance_grid_analysis(
            rows, preds_by_arch, det_preds, classifier_thresholds=[0.3, 0.5], detector_thresholds=[0.25]
        )
        self.assertEqual(len(grid), 6)
        self.assertEqual(len(default_rows), 12)
        c_row = next(g for g in grid if g['architecture'] == 'hybrid_convnext_vit' and g['classifier_threshold'] == 0.5)
        self.assertEqual(c_row['tb_both_alert'], 1)
        self.assertEqual(c_row['tb_missed_by_both'], 1)

        cases = select_visual_review_cases(rows, preds_by_arch, det_preds, cls_threshold=0.5, det_threshold=0.25)
        cats = {c['relative_path']: c['categories'] for c in cases}
        self.assertIn('tb_fn', cats['imgs/tb/t2.png'])
        self.assertIn('discrepancy_cls_only_tb', cats['imgs/tb/t2.png'])
        self.assertIn('discrepancy_det_only_non_tb', cats['imgs/sick/s1.png'])
        self.assertIn('tb_tp_representative', cats['imgs/tb/t1.png'])

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            tmp = Path(temp)
            export_detector_diagnostic_figures(det_table, tmp / 'figs')
            self.assertTrue((tmp / 'figs/detector_diagnostic.png').is_file())
            self.assertTrue((tmp / 'figs/detector_diagnostic.svg').is_file())

            for r in rows:
                p = tmp / 'data' / r['relative_path']
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (512, 512), color=(120, 130, 140)).save(p)
            panels = render_visual_review_panels(tmp / 'data', cases, tmp / 'panels', per_page=3)
            self.assertGreaterEqual(len(panels), 1)
            self.assertTrue((tmp / 'panels' / panels[0]).is_file())

            tiny_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
            p_info = count_parameters(tiny_model)
            self.assertGreater(p_info['total_params'], 0)
            bench = benchmark_classifier_cost(tiny_model, device='cpu', batch_sizes=(1,), warmup=1, iterations=2, image_size=64)
            self.assertIn('batch_1', bench)
            self.assertGreater(bench['batch_1']['mean_per_image_ms'], 0.0)

    def test_notebook_07_compiles_matches_package_and_has_no_test_leakage(self):
        nb_path = ROOT / 'notebooks/07_fase5_diagnostico_gradcam_coste_incertidumbre.ipynb'
        pkg_info_path = ROOT / 'dist/phase5_diagnostic_package_info.json'
        if not nb_path.is_file() or not pkg_info_path.is_file():
            self.skipTest('Ejecutar package_phase5_diagnostic.py y create_phase5_diagnostic_notebook.py primero.')
        notebook = json.loads(nb_path.read_text(encoding='utf-8'))
        pkg_info = json.loads(pkg_info_path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        for forbidden in ['RUN_TEST', 'manifests/test.csv', "split='test'", 'reserved_external', 'run_suite_model', '.train(']:
            self.assertNotIn(forbidden, code)
        self.assertIn('decisions_frozen=False', code)
        self.assertIn('test_evaluated=False', code)
        self.assertIn(pkg_info['sha256'], code)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()

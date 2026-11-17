"""Pruebas de métricas y controles; no requieren torch ni entrenan TBX11K."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxtrain.metrics import binary_curve, classification_metrics
from tbxtrain.protocol import check_warm_start, provenance, validate_config
from tbxtrain.train import run_experiment, run_smoke


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/convnext_tiny_baseline.json').read_text())

    def test_baseline_and_invalid_configs(self):
        self.assertEqual(validate_config(self.config)['epochs'], 3)
        for changes in [{'batch_size': 0}, {'num_workers': -1}, {'warmup_epochs': 3},
                        {'weights': 'DEFAULT'}, {'selection_metric': 'test_accuracy'},
                        {'head_lr': float('nan')}, {'hypothesis': ''}, {'amp': 'true'}]:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                validate_config({**self.config, **changes})

    def test_perfect_and_known_confusions(self):
        classes = ['health', 'sick', 'TB']
        m = classification_metrics(np.array([0, 1, 2]), np.eye(3), classes)
        self.assertEqual(m['accuracy'], 1)
        self.assertEqual(m['macro_f1'], 1)
        self.assertEqual(m['macro_roc_auc_ovr'], 1)
        # cm=[[1,1,0],[0,1,0],[1,0,1]]: verifica TB TP/FN/TN/FP.
        y = np.array([0, 0, 1, 2, 2])
        p = np.eye(3)[[0, 1, 1, 0, 2]]
        m = classification_metrics(y, p, classes)
        self.assertEqual(m['confusion_matrix'], [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(m['accuracy'], .6)
        self.assertAlmostEqual(m['macro_f1'], (.5+2/3+2/3)/3)
        self.assertEqual(m['per_class']['TB']['recall'], .5)
        self.assertEqual(m['per_class']['TB']['specificity'], 1)

    def test_auc_ap_ties_and_no_positive_class(self):
        curve = binary_curve([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertAlmostEqual(curve['roc_auc'], .75)
        self.assertAlmostEqual(curve['average_precision'], 5/6)
        tied = binary_curve([0, 1, 0, 1], [.5]*4)
        self.assertEqual(tied['roc_auc'], .5)
        self.assertEqual(tied['average_precision'], .5)
        self.assertIsNone(binary_curve([0, 0], [.1, .2])['roc_auc'])

    def test_rejects_invalid_probabilities_and_labels(self):
        for y, p in [(np.array([3]), [[0, 0, 1]]), (np.array([0]), [[1, 1, 1]]),
                     (np.array([0]), [[float('nan'), 0, 1]]), (np.array([]), [])]:
            with self.assertRaises(ValueError):
                classification_metrics(y, p, ['health', 'sick', 'TB'])

    def test_warm_start_rejects_changed_data(self):
        current = provenance(ROOT)
        check_warm_start(current, current)
        for key in ['configs/data.json', 'manifests/train.csv', 'manifests/validation.csv', 'manifests/test.csv']:
            parent = copy.deepcopy(current)
            parent[key] = 'changed'
            with self.assertRaises(ValueError):
                check_warm_start(parent, current)

    def test_public_training_entry_rejects_local_execution(self):
        with self.assertRaisesRegex(RuntimeError, 'Google Colab'):
            run_experiment(ROOT, ROOT/'TBX11K', ROOT/'tmp', self.config)
        with self.assertRaisesRegex(RuntimeError, 'Google Colab'):
            run_smoke(ROOT, ROOT/'TBX11K', ROOT/'tmp', self.config)


if __name__ == '__main__':
    unittest.main()

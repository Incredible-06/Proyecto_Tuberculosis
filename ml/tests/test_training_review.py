"""Prueba de la exportación de evidencia con un fixture sintético identificado."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxdata.common import write_csv
from tbxtrain.metrics import classification_metrics
from tbxtrain.protocol import atomic_json
from tbxtrain.review import experiment_table, plot_run, record_review


class ReviewTests(unittest.TestCase):
    def test_smoke_and_failed_runs_do_not_appear_as_baselines(self):
        base = (ROOT/'tmp').resolve()
        with tempfile.TemporaryDirectory(dir=base) as temp:
            if base not in Path(temp).resolve().parents:
                raise ValueError('Temporal fuera del proyecto.')
            for name, status in [('smoke_fixture', 'completed_smoke'), ('failed_fixture', 'failed')]:
                atomic_json(Path(temp)/'experiments'/name/'summary.json',
                            {'status': status, 'training_executed': True})
            self.assertEqual(experiment_table(temp), [])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib requerido para verificar las figuras.')
    def test_figures_table_and_review_from_fixture(self):
        import matplotlib
        matplotlib.use('Agg')
        base = (ROOT/'tmp').resolve()
        with tempfile.TemporaryDirectory(dir=base) as temp:
            if base not in Path(temp).resolve().parents:
                raise ValueError('Temporal fuera del proyecto.')
            run = Path(temp)/'experiments'/'synthetic_fixture'
            run.mkdir(parents=True)
            history = [{'epoch': 1, 'train_loss': 1.2, 'validation_loss': 1.1,
                        'validation_macro_f1': .4, 'validation_tb_recall': .5},
                       {'epoch': 2, 'train_loss': .8, 'validation_loss': .9,
                        'validation_macro_f1': .6, 'validation_tb_recall': .7}]
            write_csv(run/'history.csv', history)
            labels = np.array([0, 1, 2, 2])
            probabilities = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.6, .1, .3]])
            metrics = classification_metrics(labels, probabilities, ['health', 'sick', 'TB'])
            atomic_json(run/'best_validation.json', metrics)
            write_csv(run/'validation_predictions.csv', [
                {'label': int(y), 'p_health': float(p[0]), 'p_sick': float(p[1]), 'p_TB': float(p[2])}
                for y, p in zip(labels, probabilities)])
            plot_run(run)
            for name in ['learning_curves.png', 'confusion_matrix.png', 'validation_roc.png']:
                self.assertGreater((run/'figures'/name).stat().st_size, 1000)
            config = {'parent_run': None, 'hypothesis': 'Fixture sintético', 'change_reason': 'Prueba, sin TBX11K'}
            atomic_json(run/'summary.json', {'run_id': run.name, 'status': 'completed_preliminary',
                'training_executed': True, 'validation': metrics, 'config': config,
                'completed_epochs': 2, 'best_epoch': 2})
            table = experiment_table(temp)
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]['run_id'], 'synthetic_fixture')
            record_review(run, 'Observación de prueba', 'Hipótesis de prueba', 'Decisión de prueba')
            review = json.loads((run/'review.json').read_text())
            self.assertEqual(review['basis'], 'validation; sin abrir test ni externos')


if __name__ == '__main__':
    unittest.main()

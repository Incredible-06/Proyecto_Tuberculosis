"""Controles de reserva de test, procedencia y diagnóstico de validation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxdata.common import write_csv
from tbxtrain.phase5 import (validation_rows, verify_provenance, read_validation_predictions,
                            tb_threshold_table, concordance, detector_threshold_table)


class PreparationTests(unittest.TestCase):
    def test_original_provenance_and_validation_metrics(self):
        rows = validation_rows(ROOT)
        evidence = ROOT/'reports/phase5_readiness/drive_evidence'
        for name in ['convnext', 'vmamba', 'hybrid']:
            summary = json.loads((evidence/(name+'_summary.json')).read_text())
            self.assertGreater(verify_provenance(ROOT, summary['provenance']), 20)
            _, _, _, metrics = read_validation_predictions(evidence/(name+'_validation_predictions.csv'), rows)
            self.assertEqual(metrics['confusion_matrix'], summary['validation']['confusion_matrix'])
            self.assertAlmostEqual(metrics['macro_f1'], summary['validation']['macro_f1'], places=12)

    def test_reject_test_split_and_changed_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root = Path(temp)
            write_csv(root/'manifests/validation.csv', [dict(split='test',relative_path='imgs/tb/x.png')])
            with self.assertRaisesRegex(ValueError, 'Solo se admite validation'):
                validation_rows(root)
            (root/'source.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Procedencia distinta'):
                verify_provenance(root, {'source.py':'0'*64})
            with self.assertRaises(ValueError):
                verify_provenance(root, {'../outside':'0'*64})

    def test_reject_predictions_from_other_partition_or_duplicated(self):
        rows = [dict(relative_path='a',label='2',group_id='g')]
        record = dict(relative_path='a',label=2,group_id='g',prediction=2,p_health=0.1,p_sick=0.1,p_TB=0.8)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            path = Path(temp)/'predictions.csv'
            for bad in [dict(record, relative_path='test_image'), dict(record,label=1),
                        dict(record,group_id='other'), dict(record,prediction=1), dict(record,p_TB=float('nan'))]:
                write_csv(path,[bad])
                with self.assertRaises(ValueError): read_validation_predictions(path, rows)
            write_csv(path,[record,record])
            with self.assertRaises(ValueError): read_validation_predictions(path, rows)

    def test_threshold_ties_boundaries_and_probability_preservation(self):
        y = np.array([2,0,2,1])
        p = np.array([[.3,.2,.5],[.4,.1,.5],[0.,0.,1.],[1.,0.,0.]])
        original = p.copy()
        table = tb_threshold_table(y,p,[0,.5,1])
        self.assertEqual((table[0]['tp'],table[0]['fp']), (2,2))
        self.assertEqual((table[1]['tp'],table[1]['fp']), (2,1))
        self.assertEqual((table[2]['tp'],table[2]['fp']), (1,0))
        np.testing.assert_array_equal(p,original)

    def test_concordance_preserves_classifier_alert_without_boxes(self):
        self.assertEqual(concordance(.8,[],.5,.25), dict(classifier_alert=True,detector_alert=False,
            review_required=True,state='classifier_only_review'))
        self.assertEqual(concordance(.1,[{'confidence':.8}],.5,.25)['state'], 'detector_only_review')
        self.assertTrue(concordance(.5,[{'confidence':.25}],.5,.25)['classifier_alert'])
        self.assertEqual(concordance(.1,[],.5,.25)['state'],'neither_alert')

    @staticmethod
    def detector_fixture():
        rows = [dict(split='validation',relative_path='a',width='100',height='100',class_name='TB',
                     boxes_json=json.dumps([{'bbox':[0,0,10,10]},{'bbox':[50,50,10,10]}])),
                dict(split='validation',relative_path='b',width='100',height='100',class_name='health',boxes_json='[]')]
        predictions = {'a':[dict(xyxy=[0,0,10,10],confidence=.9),dict(xyxy=[0,0,10,10],confidence=.8)],
                       'b':[dict(xyxy=[20,20,30,30],confidence=.7)]}
        return rows,predictions

    def test_detector_one_to_one_matching_and_image_denominators(self):
        rows, predictions = self.detector_fixture()
        result = detector_threshold_table(rows,predictions,[.5])[0]
        self.assertEqual(result['matched_boxes'],1)
        self.assertEqual(result['lesion_recall'],.5)
        self.assertEqual(result['image_sensitivity_any_box'],1.)
        self.assertEqual(result['fp_boxes_per_image'],1.)
        self.assertEqual(result['negative_image_alert_rate'],1.)
        # Una caja sin solapamiento activa imagen pero no localiza una lesión.
        predictions['a'] = [dict(xyxy=[20,20,30,30],confidence=.9)]
        result = detector_threshold_table(rows,predictions,[.5])[0]
        self.assertEqual(result['image_sensitivity_any_box'],1.)
        self.assertEqual(result['image_sensitivity_localized'],0.)

    def test_detector_rejects_missing_negative_and_reserved_split(self):
        rows,predictions = self.detector_fixture()
        with self.assertRaises(ValueError): detector_threshold_table(rows,{'a':predictions['a']},[.5])
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError): detector_threshold_table(rows,predictions,[.5])

    def test_notebook_has_no_test_inference_or_freeze_switch(self):
        notebook = json.loads((ROOT/'notebooks/06_fase5_seleccion_explicabilidad_evaluacion.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for forbidden in ['RUN_TEST', 'manifests/test.csv', "split='test'", 'run_suite_model', '.train(']:
            self.assertNotIn(forbidden,code)
        self.assertIn('decisions_frozen=False',code)
        self.assertIn('RUN_DETECTOR_VALIDATION = False',code)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code': compile(''.join(cell['source']),cell['id'],'exec')


if __name__ == '__main__': unittest.main()

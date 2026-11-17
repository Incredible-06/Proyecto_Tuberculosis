"""Controles del híbrido ConvNeXt-Tiny + ViT-B/16, exportación de 5 modelos y notebook."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ['torch', 'torchvision', 'matplotlib'])
if AVAILABLE:
    import torch
    from torchvision.models import vit_b_16
    from tbxtrain.hybrid_vit_experiment import (
        ConvNeXtViTHybridClassifier,
        export_hybrid_vit_report_outputs,
        run_hybrid_vit_model,
        validate_hybrid_vit_config,
    )
from tbxdata.common import sha256, write_csv
from tbxtrain.metrics import classification_metrics
from tbxtrain.phase5 import CLASSES, validation_rows
from tbxtrain.protocol import atomic_json


@unittest.skipUnless(AVAILABLE, 'Fixture neuronal opcional: torch/torchvision/matplotlib requeridos.')
class HybridVitExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.cfg = json.loads((ROOT / 'configs/hybrid_convnext_vit_controlled.json').read_text(encoding='utf-8'))

    def test_hybrid_convnext_vit_phases_gradients_and_interpolation(self):
        torch.manual_seed(53)
        model = ConvNeXtViTHybridClassifier(pretrained=False, image_size=512)
        src_224 = vit_b_16(weights=None, image_size=224)
        model.load_vit_interpolated_state(src_224.state_dict(), image_size=512)

        model.configure_phase(False)
        self.assertFalse(any(p.requires_grad for m in model.backbones() for p in m.parameters()))
        self.assertFalse(model.vit.class_token.requires_grad)
        self.assertFalse(model.vit.encoder.pos_embedding.requires_grad)
        self.assertTrue(all(p.requires_grad for m in model.head_modules() for p in m.parameters()))

        model.configure_phase(True)
        self.assertFalse(any(p.requires_grad for p in model.conv[:-2].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.vit.encoder.layers[:-2].parameters()))
        self.assertTrue(all(p.requires_grad for m in model.last_modules() for p in m.parameters()))

        logits = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 3))
        torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
        groups = model.parameter_groups(self.cfg['head_lr'], self.cfg['backbone_lr'])
        self.assertEqual(len(groups), 2)

    def test_config_parity_and_colab_guard(self):
        validate_hybrid_vit_config(self.cfg, ROOT)
        with self.assertRaises(ValueError):
            validate_hybrid_vit_config(dict(self.cfg, seed=123), ROOT)
        with self.assertRaises(ValueError):
            validate_hybrid_vit_config(dict(self.cfg, precision='fp16'), ROOT)
        with self.assertRaises(RuntimeError):
            run_hybrid_vit_model(ROOT, ROOT / 'TBX11K', ROOT / 'tmp', self.cfg)

    def test_export_report_outputs_generates_5model_figures_and_payload(self):
        rows = validation_rows(ROOT)
        y = np.array([int(r['label']) for r in rows])
        probs = np.full((len(rows), 3), 0.05, dtype=float)
        for idx, label in enumerate(y):
            probs[idx, label] = 0.90
        metrics = classification_metrics(y, probs, CLASSES)

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            drive = Path(temp)
            run_id = 'phase3_hybrid_convnext_vit_synthetic'
            exp = drive / 'experiments' / run_id
            exp.mkdir(parents=True)
            ckpt = drive / 'checkpoints' / run_id / 'best.pt'
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b'synthetic hybrid vit')

            write_csv(exp / 'validation_predictions.csv', [
                dict(
                    relative_path=r['relative_path'], group_id=r['group_id'],
                    label=int(label), prediction=int(p.argmax()),
                    **{f'p_{c}': float(p[i]) for i, c in enumerate(CLASSES)},
                )
                for r, label, p in zip(rows, y, probs)
            ])
            history = [
                dict(epoch=1, train_loss=0.8, train_accuracy=0.80, optimizer_updates=181, train_n=5792,
                     stage='head_only', validation_loss=0.6, validation_accuracy=0.90,
                     validation_macro_f1=0.85, validation_tb_recall=0.70, seconds=100.0),
                dict(epoch=2, train_loss=0.4, train_accuracy=0.91, optimizer_updates=181, train_n=5792,
                     stage='head_only', validation_loss=0.3, validation_accuracy=0.95,
                     validation_macro_f1=0.92, validation_tb_recall=0.80, seconds=100.0),
                dict(epoch=3, train_loss=0.2, train_accuracy=0.96, optimizer_updates=181, train_n=5792,
                     stage='last_stages', validation_loss=0.2, validation_accuracy=1.00,
                     validation_macro_f1=1.00, validation_tb_recall=1.00, seconds=100.0),
            ]
            write_csv(exp / 'history.csv', history)
            atomic_json(exp / 'summary.json', {
                'schema': 'hybrid_convnext_vit_controlled_v1', 'run_id': run_id,
                'architecture': 'hybrid_convnext_vit', 'status': 'completed_controlled',
                'training_executed': True, 'protocol': self.cfg, 'completed_epochs': 3,
                'optimizer_updates': 543, 'best_epoch': 3, 'stop_reason': 'epoch_budget',
                'validation': metrics, 'provenance': {'fixture': 'ok'},
                'checkpoint_sha256': sha256(ckpt), 'test_evaluated': False,
                'external_evaluated': False, 'phase3_complete': False,
            })

            out = drive / 'export'
            payload = export_hybrid_vit_report_outputs(ROOT, drive, run_id, out)
            self.assertFalse(payload['test_evaluated'])
            self.assertEqual(len(payload['comparison_5models']), 5)
            for fname in [
                'hybrid_convnext_vit_training.png', 'hybrid_convnext_vit_training.pdf',
                'hybrid_convnext_vit_training.svg', 'hybrid_convnext_vit_training.csv',
                'hybrid_convnext_vit_confusion.png', 'hybrid_convnext_vit_roc.png',
                'classifier_comparison.png', 'TB_roc_precision_recall.png',
                'hybrid_convnext_vit_thresholds.png',
                'hybrid_convnext_vit_TB_thresholds_validation.csv',
                'validation_comparison_5models.csv', 'hybrid_vit_report_payload.json',
            ]:
                self.assertTrue((out / fname).is_file(), f'Falta {fname}')

    def test_notebook_compiles_matches_package_and_has_no_test_leakage(self):
        nb_path = ROOT / 'notebooks/04c_entrenamiento_hibrido_convnext_vit.ipynb'
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        info = json.loads((ROOT / 'dist/hybrid_vit_package_info.json').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertIn(info['sha256'], code)
        for forbidden in ['RUN_TEST', 'manifests/test.csv', "split='test'"]:
            self.assertNotIn(forbidden, code)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()

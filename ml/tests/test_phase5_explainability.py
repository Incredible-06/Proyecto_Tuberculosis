"""Pruebas unitarias locales para explicabilidad verificada sobre las 5 arquitecturas."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tbxtrain.architectures import build_classifier
from tbxtrain.hybrid_vit_experiment import ConvNeXtViTHybridClassifier
from tbxtrain.phase5_explainability import (
    _vit_rollout_and_logits,
    explain_batch,
    render_explainability_panels,
    run_verified_explainability_suite,
    summarize_explainability_checks,
)
from tbxtrain.vit_experiment import ViTClassifier


def _tiny_sample(path='imgs/tb/sample.png', cls_name='TB'):
    rng = np.random.default_rng(123)
    arr = rng.normal(0.0, 1.0, size=(3, 128, 128)).astype(np.float32)
    return {
        'relative_path': path,
        'class_name': cls_name,
        'tensor': arr,
        'boxes_json': json.dumps([{'bbox': [20, 20, 48, 48]}]) if cls_name == 'TB' else '[]',
    }


class Phase5ExplainabilityTests(unittest.TestCase):
    def test_vit_rollout_matches_model_forward(self):
        model = ViTClassifier(pretrained=False, image_size=128).eval()
        # Reducir profundidad para prueba rápida en CPU.
        model.encoder.encoder.layers = model.encoder.encoder.layers[:2]
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            seq, rollout = _vit_rollout_and_logits(model.encoder, x)
            logits_roll = model.encoder.heads(seq[:, 0])
            logits_ref = model(x)
        self.assertTrue(torch.allclose(logits_roll, logits_ref, atol=1e-5, rtol=1e-5))
        self.assertEqual(rollout.shape, (1, 8, 8))
        self.assertTrue(np.isfinite(rollout).all())

    def test_explainability_and_sanity_checks_across_architectures(self):
        # Prueba rápida en CPU para ConvNeXt, VMamba, Híbrido C+VM, ViT-B/16 e Híbrido C+ViT.
        models = {
            'convnext_tiny': build_classifier('convnext_tiny', pretrained=False),
            'vmamba_tiny': build_classifier('vmamba_tiny', pretrained=False),
            'hybrid_cross_attention': build_classifier('hybrid_cross_attention', pretrained=False),
            'vit_b_16': ViTClassifier(pretrained=False, image_size=128),
            'hybrid_convnext_vit': ConvNeXtViTHybridClassifier(pretrained=False, image_size=128),
        }
        # Recortar bloques de ViT en prueba CPU para acelerar sin cambiar interfaces.
        models['vit_b_16'].encoder.encoder.layers = models['vit_b_16'].encoder.encoder.layers[:2]
        models['hybrid_convnext_vit'].vit.encoder.layers = models['hybrid_convnext_vit'].vit.encoder.layers[:2]

        sample = _tiny_sample()
        visuals_by_arch = {}
        all_records = []
        for arch, model in models.items():
            x = torch.from_numpy(sample['tensor']).unsqueeze(0)
            out = explain_batch(model, arch, x, target_class=2, image_size=128)
            self.assertEqual(out['probabilities'].shape, (1, 3))
            self.assertTrue(np.isfinite(out['probabilities']).all())
            for branch_name, st in out['branch_stats'].items():
                self.assertTrue(st[0]['activation_finite'], f'{arch}:{branch_name}')
                self.assertTrue(st[0]['gradient_finite'], f'{arch}:{branch_name}')
                self.assertGreater(st[0]['gradient_norm'], 0.0, f'{arch}:{branch_name}')

            records, visuals = run_verified_explainability_suite(
                model,
                arch,
                [sample],
                device='cpu',
                image_size=128,
                run_randomization_check=(arch in {'convnext_tiny', 'vit_b_16', 'hybrid_convnext_vit'}),
            )
            self.assertTrue(records[0]['all_finite'], arch)
            self.assertTrue(records[0]['all_nonzero_grad'], arch)
            self.assertIsNotNone(records[0]['pointing_game_hit'], arch)
            all_records.extend(records)
            visuals_by_arch[arch] = visuals

        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            tmp = Path(temp)
            img_path = tmp / 'data' / sample['relative_path']
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (128, 128), color=(90, 90, 90)).save(img_path)
            exported = render_explainability_panels(tmp / 'data', visuals_by_arch, tmp / 'panels', image_size=128)
            self.assertEqual(len(exported), 1)
            self.assertTrue((tmp / 'panels' / exported[0]).is_file())
            flat = summarize_explainability_checks(all_records, tmp / 'checks')
            self.assertEqual(len(flat), 5)
            self.assertTrue((tmp / 'checks/explainability_checks.csv').is_file())
            self.assertTrue((tmp / 'checks/explainability_checks.json').is_file())


if __name__ == '__main__':
    unittest.main()

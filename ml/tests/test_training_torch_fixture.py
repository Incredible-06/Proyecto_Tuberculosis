"""Pruebas opcionales CPU con datos sintéticos; no entrenan TBX11K.

Requieren torch/torchvision. La entrada pública sigue restringida a Colab GPU.
"""
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
AVAILABLE = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'))
if AVAILABLE:
    import torch
    from tbxtrain.train import atomic_checkpoint, build_model, evaluate, set_trainable, train_epoch, rng_state, restore_rng


@unittest.skipUnless(AVAILABLE, 'Fixture opcional: torch/torchvision no instalados en este intérprete.')
class TorchFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT/'configs/convnext_tiny_baseline.json').read_text())

    def test_exact_weighted_gradient_accumulation_including_last_partial_window(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh())
                self.classifier = torch.nn.Linear(5, 3)
            def forward(self, x):
                return self.classifier(self.features(x))
        torch.manual_seed(41)
        model = Tiny()
        reference = copy.deepcopy(model)
        x, y = torch.randn(7, 4), torch.tensor([0, 1, 2, 2, 0, 1, 2])
        weights = torch.tensor([.5, 1., 4.])
        config = {**self.config, 'amp': False, 'unfreeze_last_stage': False,
                  'batch_size': 2, 'accumulation_steps': 3, 'gradient_clip': 1e6}
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.02)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
        result = train_epoch(torch, model, loader, optimizer, torch.amp.GradScaler('cpu', enabled=False),
                             weights, config, epoch=0)
        set_trainable(reference, False)
        for begin, end in [(0, 6), (6, 7)]:
            ref_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(reference(x[begin:end]), y[begin:end], weight=weights)
            loss.backward(); ref_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(result['optimizer_updates'], 2)
        self.assertEqual(result['train_n'], 7)

    def test_real_convnext_forward_backward_and_checkpoint_roundtrip(self):
        torch.manual_seed(43)
        model = build_model(torch, pretrained=False)  # Sin descarga de pesos.
        set_trainable(model, False)
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        set_trainable(model, True)
        self.assertTrue(all(p.requires_grad for m in model.features[-2:] for p in m.parameters()))
        self.assertFalse(any(p.requires_grad for m in model.features[:-2] for p in m.parameters()))
        self.assertFalse(model.features[1].training)
        self.assertTrue(model.features[-1].training)
        x, y = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
        weights = torch.tensor([.7, .8, 3.4])
        config = {**self.config, 'amp': False, 'accumulation_steps': 2}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.classifier[2].weight.detach().clone()
        result = train_epoch(torch, model, loader, optimizer, torch.amp.GradScaler('cpu', enabled=False),
                             weights, config, epoch=1)
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertFalse(torch.equal(before, model.classifier[2].weight.detach()))
        metrics, labels, probabilities = evaluate(torch, model, loader, weights, False, ['health', 'sick', 'TB'])
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertEqual(metrics['n'], 3)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        directory = (ROOT/'tmp').resolve()
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            if directory not in Path(temp).resolve().parents:
                raise ValueError('Temporal fuera del proyecto.')
            target = Path(temp)/'last.pt'
            # El sandbox Windows no permite que PyTorch serialice en su Temp
            # virtualizado; el fixture usa un temporal explícito del proyecto.
            with mock.patch.object(tempfile, 'tempdir', str(directory)):
                atomic_checkpoint(torch, target, {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()})
            restored = torch.load(target, map_location='cpu', weights_only=True)
            reloaded = build_model(torch, pretrained=False)
            reloaded.load_state_dict(restored['model_state'])
            _, y2, p2 = evaluate(torch, reloaded, loader, weights, False, ['health', 'sick', 'TB'])
            np.testing.assert_array_equal(labels, y2)
            np.testing.assert_allclose(probabilities, p2, atol=1e-7)
            self.assertFalse(target.with_name('last.pt.tmp').exists())

    def test_optimizer_and_rng_resume_reproduces_next_update(self):
        torch.manual_seed(47); np.random.seed(47); random.seed(47)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        def update(m, o):
            x, labels = torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 2])
            o.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(m(x), labels).backward()
            o.step()
        update(model, optimizer)
        state = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                 'rng': rng_state(torch)}
        update(model, optimizer)
        expected_np, expected_py = np.random.random(), random.random()
        restored = torch.nn.Linear(4, 3)
        restored.load_state_dict(state['model'])
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        resumed_optimizer.load_state_dict(state['optimizer'])
        restore_rng(torch, state['rng'])
        update(restored, resumed_optimizer)
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(np.random.random(), expected_np)
        self.assertEqual(random.random(), expected_py)


if __name__ == '__main__':
    unittest.main()

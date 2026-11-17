"""Ejecuta pruebas de Fase 3 y registra su alcance real, sin entrenar TBX11K."""
import importlib.util
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tbxtrain.protocol import atomic_json


def main():
    tests = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_training_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    torch_available = bool(importlib.util.find_spec('torch'))
    versions = {}
    if torch_available:
        import torch
        import torchvision
        versions = {'torch': torch.__version__, 'torchvision': torchvision.__version__}
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'versions': versions, 'colab_training_executed': False, 'tbx11k_training_executed': False,
        'fixture_scope': 'CPU, datos sintéticos: métricas, acumulación ponderada, ConvNeXt 32px sin pesos descargados, checkpoint, optimizador/RNG y figuras. CUDA/AMP/512px reales pendientes del smoke Colab.'}
    atomic_json(ROOT/'reports/phase3/tests.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

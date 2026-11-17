"""Congela el contrato inmutable de Fase 5 antes de abrir test reservado."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tbxdata.common import sha256
from tbxtrain.protocol import atomic_json


def build_contract(project=ROOT):
    project = Path(project)
    code_files = sorted(
        list((project / 'src/tbxdata').glob('*.py'))
        + [
            project / 'src/tbxtrain/__init__.py',
            project / 'src/tbxtrain/architectures.py',
            project / 'src/tbxtrain/metrics.py',
            project / 'src/tbxtrain/phase5.py',
            project / 'src/tbxtrain/phase5_diagnostics.py',
            project / 'src/tbxtrain/phase5_evaluation.py',
            project / 'src/tbxtrain/protocol.py',
            project / 'src/tbxtrain/train.py',
            project / 'src/tbxtrain/vit_experiment.py',
        ]
    )
    code_hashes = {p.relative_to(project).as_posix(): sha256(p) for p in code_files}
    return {
        'schema': 'phase5_frozen_contract_v1',
        'decisions_frozen': True,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rationale': (
            'ViT-B/16 empatado en Macro-F1 (0.9644 vs 0.9645 de Híbrido C+VM; IC 95 % del delta cruza cero), '
            'mayor sensibilidad TB en argmax (91.67 %) y combinada con YOLO11n (96.67 % a p_TB>=0.30), '
            'mayor enriquecimiento Grad-CAM en cajas GT (1.42x) y portabilidad PyTorch estándar sin kernel oflex.'
        ),
        'classifier': {
            'architecture': 'vit_b_16',
            'display_name': 'ViT-B/16',
            'run_id': 'phase3_vit_b_16_20261116T014808Z_2e8a27',
            'protocol_config': 'configs/vit_controlled.json',
            'checkpoint_sha256': '2873fbcff636ad151b41451022b0fc34eeb197866cf5acd200e3f764f4f3d50d',
            'best_epoch': 11,
            'requires_oflex': False,
            'operational_tb_threshold': 0.30,
            'preserve_argmax_and_probabilities': True,
        },
        'detector': {
            'architecture': 'yolo11n',
            'run_id': 'detector_yolo11n_20261103T032033865005Z',
            'weights_sha256': '3e28e82987c8c2c84ca503f639354c7da1130472165adae8642737c7b5263dd0',
            'confidence_threshold': 0.25,
            'inference_options': {
                'imgsz': 512,
                'conf': 0.001,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'rect': False,
                'augment': False,
            },
            'nms_iou': 0.7,
            'max_det': 300,
            'match_iou_for_localization_metrics': 0.5,
        },
        'concordance_rule': {
            'name': 'additive_4state_no_suppression_v1',
            'suppress_classifier_without_boxes': False,
            'multiply_scores': False,
            'states': ['both_alert', 'classifier_only_review', 'detector_only_review', 'neither_alert'],
        },
        'preprocessing': {
            'data_config': 'configs/data.json',
            'data_config_sha256': sha256(project / 'configs/data.json'),
            'image_size': 512,
            'color_mode': 'RGB',
            'letterbox': True,
            'normalization': 'imagenet_rgb',
        },
        'validation_evidence': {
            'diagnostic_run_id': 'phase5_diag_20261116T042213Z_ac3bcb',
            'validation_manifest_sha256': sha256(project / 'manifests/validation.csv'),
            'test_manifest_sha256': sha256(project / 'manifests/test.csv'),
        },
        'code_hashes': code_hashes,
    }


def main():
    contract = build_contract(ROOT)
    target = ROOT / 'configs/phase5_frozen_contract.json'
    atomic_json(target, contract)
    print('Contrato congelado escrito en:', target)


if __name__ == '__main__':
    main()

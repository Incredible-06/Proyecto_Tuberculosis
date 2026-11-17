"""Preparación y análisis exclusivo de validation para Fase 5.

No ofrece un lector de test ni una función de evaluación final. Los umbrales
explorados no congelan decisiones y las probabilidades originales se conservan.
"""
import json
from pathlib import Path
import numpy as np
from tbxdata.common import read_csv, safe_path, sha256
from .metrics import classification_metrics

CLASSES = ['health', 'sick', 'TB']


def validation_rows(project):
    rows = read_csv(Path(project)/'manifests/validation.csv')
    if not rows or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Solo se admite validation en esta preparación.')
    paths = [r['relative_path'] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Rutas duplicadas en validation.')
    for r in rows:
        safe_path(project, r['relative_path'])
        if r['class_name'] not in CLASSES or int(r['label']) != CLASSES.index(r['class_name']):
            raise ValueError('Mapa de clases alterado.')
    return rows


def verify_provenance(project, recorded):
    """Verifica cada fuente ORIGINAL registrada; permite módulos nuevos."""
    if not recorded:
        raise ValueError('Falta procedencia del entrenamiento.')
    for name, expected in recorded.items():
        path = safe_path(project, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Procedencia distinta del entrenamiento: '+name)
    return len(recorded)


def read_validation_predictions(path, rows):
    predictions = read_csv(path)
    by_path = {r['relative_path']: r for r in predictions}
    if len(by_path) != len(predictions) or set(by_path) != {r['relative_path'] for r in rows}:
        raise ValueError('Predicciones duplicadas o cobertura distinta de validation.')
    ordered = [by_path[r['relative_path']] for r in rows]
    if any(int(p['label']) != int(r['label']) or p['group_id'] != r['group_id']
           for p, r in zip(ordered, rows)):
        raise ValueError('Etiquetas/grupos de predicción distintos del manifiesto.')
    y = np.array([int(r['label']) for r in rows])
    p = np.array([[float(r['p_'+c]) for c in CLASSES] for r in ordered])
    metrics = classification_metrics(y, p, CLASSES)
    if (p > 1).any() or any(int(r['prediction']) != int(i) for r, i in zip(ordered, p.argmax(1))):
        raise ValueError('Argmax/probabilidades registrados inconsistentes.')
    return ordered, y, p, metrics


def verify_classifier(project, base, spec, rows):
    folder = Path(base)/'experiments'/spec['run_id']
    summary = json.loads((folder/'summary.json').read_text())
    if (summary['run_id'] != spec['run_id'] or summary['architecture'] != spec['architecture']
            or summary['status'] != 'completed_controlled' or not summary['training_executed']
            or summary['test_evaluated'] or summary['external_evaluated']):
        raise ValueError('Corrida fuera del protocolo: '+spec['run_id'])
    if summary['protocol'] != json.loads((Path(project)/'configs/phase3_suite.json').read_text()):
        raise ValueError('Protocolo distinto.')
    verify_provenance(project, summary['provenance'])
    checkpoint = Path(base)/'checkpoints'/spec['run_id']/'best.pt'
    actual = sha256(checkpoint)
    if actual != spec['sha256'] or actual != summary['checkpoint_sha256']:
        raise ValueError('Pesos alterados: '+str(checkpoint))
    ordered, y, p, metrics = read_validation_predictions(folder/'validation_predictions.csv', rows)
    if (metrics['confusion_matrix'] != summary['validation']['confusion_matrix']
            or not np.isclose(metrics['macro_f1'], summary['validation']['macro_f1'], atol=1e-12, rtol=0)
            or metrics['n'] != summary['validation']['n']):
        raise ValueError('Métricas no reproducidas.')
    return summary, ordered, y, p, metrics


def restore_classifier(project, base, spec, device='cuda'):
    """Solo best.pt propio y verificado; sin descargar pesos ImageNet."""
    import torch
    from .architectures import build_classifier, scan_backend
    rows = validation_rows(project)
    summary, _, _, _, _ = verify_classifier(project, base, spec, rows)
    if spec['architecture'] != 'convnext_tiny' and scan_backend() != 'oflex':
        raise RuntimeError('Se requiere selective scan oflex; no usar torch_reference silenciosamente.')
    payload = torch.load(Path(base)/'checkpoints'/spec['run_id']/'best.pt',
                         map_location='cpu', weights_only=True)
    if (payload['architecture'] != spec['architecture'] or payload['run_id'] != spec['run_id']
            or payload['epoch'] != summary['best_epoch'] or payload['classes'] != CLASSES
            or payload['config'] != summary['protocol'] or payload['provenance'] != summary['provenance']
            or payload['data_config'] != json.loads((Path(project)/'configs/data.json').read_text())):
        raise ValueError('Contrato interno del checkpoint inconsistente.')
    model = build_classifier(spec['architecture'], pretrained=False)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval()


def tb_threshold_table(labels, probabilities, thresholds=None):
    """Alerta p_TB >= t independiente de argmax; incluye todos los cambios."""
    classification_metrics(labels, probabilities, CLASSES)
    y, score = np.asarray(labels) == 2, np.asarray(probabilities)[:, 2]
    thresholds = np.unique(np.r_[0., score, 1.]) if thresholds is None else np.asarray(thresholds)
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError('Umbral fuera de [0,1].')
    out = []
    for threshold in thresholds:
        alert = score >= threshold
        tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
        fn, tn = int((~alert & y).sum()), int((~alert & ~y).sum())
        out.append(dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                        sensitivity=tp/(tp+fn) if tp+fn else None,
                        specificity=tn/(tn+fp) if tn+fp else None,
                        precision=tp/(tp+fp) if tp+fp else None,
                        alerts=int(alert.sum())))
    return out


def concordance(p_tb, boxes, classifier_threshold, detector_threshold):
    """Regla exploratoria v1: no suprime ninguna alerta del clasificador."""
    for x in (p_tb, classifier_threshold, detector_threshold):
        if not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Score/umbral fuera de [0,1].')
    ca = bool(p_tb >= classifier_threshold)
    da = any(b['confidence'] >= detector_threshold for b in boxes)
    state = {(True, True): 'both_alert', (True, False): 'classifier_only_review',
             (False, True): 'detector_only_review', (False, False): 'neither_alert'}[ca, da]
    return dict(classifier_alert=ca, detector_alert=da, review_required=bool(ca or da), state=state)


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left)*max(0, bottom-top)
    area = lambda v: (v[2]-v[0])*(v[3]-v[1])
    return intersection/(area(a)+area(b)-intersection)


def detector_threshold_table(rows, predictions, thresholds, match_iou=0.5):
    """Matching greedy por confianza, uno a uno, en coordenadas originales.

    FP/caja por imagen se calcula sobre TODAS las imágenes; también se informa
    tasa de imágenes negativas con alguna caja. No es el mAP de Ultralytics.
    """
    if set(predictions) != {r['relative_path'] for r in rows}:
        raise ValueError('Detector debe cubrir exactamente validation, incluidos resultados sin cajas.')
    if not 0 < match_iou <= 1 or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Solo validation con IoU válido.')
    for row in rows:
        if row['class_name'] == 'TB' and not json.loads(row['boxes_json']):
            raise ValueError('TB sin cajas no puede tratarse como negativo.')
        for box in predictions[row['relative_path']]:
            a = np.asarray(box['xyxy'], float)
            if (a.shape != (4,) or not np.isfinite(a).all() or min(a) < 0
                    or a[2] <= a[0] or a[3] <= a[1]
                    or a[2] > int(row['width'])+.001 or a[3] > int(row['height'])+.001
                    or not 0 <= box['confidence'] <= 1):
                raise ValueError('Caja predicha inválida.')
    output = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Confianza inválida.')
        tp = fp = gt_count = positive = negative = image_tp = negative_alerts = localized = 0
        for row in rows:
            gt = [[b['bbox'][0], b['bbox'][1], b['bbox'][0]+b['bbox'][2],
                   b['bbox'][1]+b['bbox'][3]] for b in json.loads(row['boxes_json'])]
            boxes = sorted((b for b in predictions[row['relative_path']] if b['confidence'] >= threshold),
                           key=lambda b: -b['confidence'])
            positive += bool(gt); negative += not bool(gt); gt_count += len(gt)
            image_tp += bool(gt) and bool(boxes)
            negative_alerts += not bool(gt) and bool(boxes)
            used = set()
            for box in boxes:
                candidates = [(i, _iou(box['xyxy'], g)) for i, g in enumerate(gt) if i not in used]
                i, overlap = max(candidates, key=lambda v: v[1], default=(-1, 0.))
                if overlap >= match_iou:
                    used.add(i); tp += 1
                else:
                    fp += 1
            localized += bool(used)
        output.append(dict(threshold=float(threshold), match_iou=match_iou, matched_boxes=tp,
            gt_boxes=gt_count, false_positive_boxes=fp,
            lesion_recall=tp/gt_count if gt_count else None,
            image_sensitivity_any_box=image_tp/positive if positive else None,
            image_sensitivity_localized=localized/positive if positive else None,
            fp_boxes_per_image=fp/len(rows),
            negative_image_alert_rate=negative_alerts/negative if negative else None))
    return output

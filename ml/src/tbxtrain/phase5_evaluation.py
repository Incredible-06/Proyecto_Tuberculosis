"""Evaluación final única bajo contrato congelado para Fase 5 (Parte 2).

Verifica el contrato inmutable (código, pesos, preprocesamiento, umbrales y
concordancia), ejecuta una sola evaluación sobre manifests/test.csv, audita y
evalúa Montgomery/Shenzhen con mapeo binario explícito y exporta el paquete
congelado para api/ y web/.
"""
import csv
import functools
import gc
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from tbxdata.common import read_csv, safe_path, sha256, write_csv
from tbxdata.dataset import RadiographTransform, TBXDataset, load_config
from tbxdata.validate import validate_manifests
from .metrics import binary_curve, classification_metrics
from .phase5 import CLASSES, _iou, concordance, tb_threshold_table
from .phase5_diagnostics import (
    _fast_multiclass_summary,
    _group_strata,
    _summarize_distribution,
    restore_any_classifier,
    verify_detector,
)
from .protocol import atomic_json


def verify_frozen_contract(project, contract_path=None):
    """Carga y verifica integridad completa del contrato inmutable antes de abrir test."""
    project = Path(project)
    c_path = Path(contract_path) if contract_path else (project / 'configs/phase5_frozen_contract.json')
    contract = json.loads(c_path.read_text(encoding='utf-8'))
    if contract.get('schema') != 'phase5_frozen_contract_v1' or contract.get('decisions_frozen') is not True:
        raise ValueError('El contrato debe tener schema phase5_frozen_contract_v1 y decisions_frozen=True.')
    cls_cfg = contract['classifier']
    det_cfg = contract['detector']
    rule = contract['concordance_rule']
    if not (0.0 <= float(cls_cfg['operational_tb_threshold']) <= 1.0):
        raise ValueError('Umbral operativo del clasificador fuera de [0, 1].')
    if not (0.0 <= float(det_cfg['confidence_threshold']) <= 1.0):
        raise ValueError('Umbral del detector fuera de [0, 1].')
    if rule.get('suppress_classifier_without_boxes') is not False or rule.get('multiply_scores') is not False:
        raise ValueError('La regla congelada no puede suprimir alertas ni multiplicar scores.')
    prep = contract['preprocessing']
    data_path = safe_path(project, prep['data_config'])
    if sha256(data_path) != prep['data_config_sha256']:
        raise ValueError('configs/data.json alterado respecto al contrato congelado.')
    ev = contract['validation_evidence']
    if sha256(project / 'manifests/validation.csv') != ev['validation_manifest_sha256']:
        raise ValueError('manifests/validation.csv alterado respecto al contrato.')
    if sha256(project / 'manifests/test.csv') != ev['test_manifest_sha256']:
        raise ValueError('manifests/test.csv alterado respecto al contrato.')
    for rel_path, expected_hash in contract['code_hashes'].items():
        actual = sha256(safe_path(project, rel_path))
        if actual != expected_hash:
            raise ValueError(f'Código alterado respecto al contrato congelado: {rel_path}')
    return contract


def evaluate_detector_on_split(rows, predictions, thresholds, expected_split='test', match_iou=0.5):
    """Métricas por umbral del detector (matching greedy 1 a 1) para cualquier partición."""
    if set(predictions) != {r['relative_path'] for r in rows}:
        raise ValueError('El detector debe cubrir exactamente todas las imágenes de la partición.')
    if not 0 < match_iou <= 1 or any(r['split'] != expected_split for r in rows):
        raise ValueError(f'Todas las filas deben pertenecer a {expected_split}.')
    health_rows = [r for r in rows if r['class_name'] == 'health']
    sick_rows = [r for r in rows if r['class_name'] == 'sick']
    tb_rows = [r for r in rows if r['class_name'] == 'TB']

    for row in rows:
        if row['class_name'] == 'TB' and not json.loads(row['boxes_json']):
            raise ValueError('TB sin cajas no puede tratarse como negativo.')
        for box in predictions[row['relative_path']]:
            a = np.asarray(box['xyxy'], float)
            if (a.shape != (4,) or not np.isfinite(a).all() or min(a) < 0
                    or a[2] <= a[0] or a[3] <= a[1]
                    or a[2] > int(row['width']) + 0.001 or a[3] > int(row['height']) + 0.001
                    or not 0 <= box['confidence'] <= 1):
                raise ValueError('Caja predicha inválida.')

    output = []
    for threshold in thresholds:
        tp = fp = gt_count = positive = negative = image_tp = negative_alerts = localized = 0
        for row in rows:
            gt = [[b['bbox'][0], b['bbox'][1], b['bbox'][0] + b['bbox'][2], b['bbox'][1] + b['bbox'][3]]
                  for b in json.loads(row['boxes_json'])]
            boxes = sorted((b for b in predictions[row['relative_path']] if b['confidence'] >= threshold),
                           key=lambda b: -b['confidence'])
            positive += bool(gt)
            negative += not bool(gt)
            gt_count += len(gt)
            image_tp += bool(gt) and bool(boxes)
            negative_alerts += not bool(gt) and bool(boxes)
            used = set()
            for box in boxes:
                candidates = [(i, _iou(box['xyxy'], g)) for i, g in enumerate(gt) if i not in used]
                i, overlap = max(candidates, key=lambda v: v[1], default=(-1, 0.0))
                if overlap >= match_iou:
                    used.add(i)
                    tp += 1
                else:
                    fp += 1
            localized += bool(used)
        total_pred = tp + fp
        health_alerts = sum(1 for r in health_rows if any(b['confidence'] >= threshold for b in predictions[r['relative_path']]))
        sick_alerts = sum(1 for r in sick_rows if any(b['confidence'] >= threshold for b in predictions[r['relative_path']]))
        output.append({
            'threshold': float(threshold),
            'match_iou': float(match_iou),
            'matched_boxes': tp,
            'gt_boxes': gt_count,
            'false_positive_boxes': fp,
            'total_predicted_boxes': total_pred,
            'box_precision': (tp / total_pred) if total_pred else None,
            'lesion_recall': (tp / gt_count) if gt_count else None,
            'image_sensitivity_any_box': (image_tp / positive) if positive else None,
            'image_sensitivity_localized': (localized / positive) if positive else None,
            'fp_boxes_per_image': fp / len(rows),
            'negative_image_alert_rate': (negative_alerts / negative) if negative else None,
            'health_images': len(health_rows),
            'sick_images': len(sick_rows),
            'tb_images': len(tb_rows),
            'health_alert_count': health_alerts,
            'sick_alert_count': sick_alerts,
            'health_alert_rate': (health_alerts / len(health_rows)) if health_rows else None,
            'sick_alert_rate': (sick_alerts / len(sick_rows)) if sick_rows else None,
        })
    return output


def _bootstrap_frozen_test(rows, probs, det_predictions, cls_thresh, det_thresh, match_iou=0.5, n_resamples=2000, seed=20261019, ci_level=0.95):
    """Bootstrap estratificado por group_id sobre test congelado."""
    y_all = np.asarray([int(r['label']) for r in rows], dtype=np.int64)
    group_list, strata = _group_strata(rows)
    rng = np.random.default_rng(int(seed))
    boot_indices = []
    for _ in range(int(n_resamples)):
        sampled = [rng.choice(g_arr, size=len(g_arr), replace=True) for g_arr in strata.values()]
        boot_indices.append(np.concatenate([group_list[g] for g in np.concatenate(sampled)]))

    point_cls = _fast_multiclass_summary(y_all, probs)
    cls_series = defaultdict(list)
    y_tb = (y_all == 2)
    score_tb = probs[:, 2]

    # Precomputar alertas de concordancia y detector por imagen para el bootstrap.
    n = len(rows)
    cls_alert_arr = score_tb >= cls_thresh
    det_alert_arr = np.zeros(n, dtype=bool)
    tp_arr = np.zeros(n, dtype=np.int64)
    fp_arr = np.zeros(n, dtype=np.int64)
    gt_arr = np.zeros(n, dtype=np.int64)
    loc_arr = np.zeros(n, dtype=np.int64)
    for idx, row in enumerate(rows):
        gt = [[b['bbox'][0], b['bbox'][1], b['bbox'][0] + b['bbox'][2], b['bbox'][1] + b['bbox'][3]]
              for b in json.loads(row['boxes_json'])]
        boxes = sorted((b for b in det_predictions[row['relative_path']] if b['confidence'] >= det_thresh),
                       key=lambda b: -b['confidence'])
        det_alert_arr[idx] = bool(boxes)
        gt_arr[idx] = len(gt)
        used = set()
        tp_i = fp_i = 0
        for box in boxes:
            candidates = [(i, _iou(box['xyxy'], g)) for i, g in enumerate(gt) if i not in used]
            best_i, overlap = max(candidates, key=lambda v: v[1], default=(-1, 0.0))
            if overlap >= match_iou:
                used.add(best_i)
                tp_i += 1
            else:
                fp_i += 1
        tp_arr[idx] = tp_i
        fp_arr[idx] = fp_i
        loc_arr[idx] = int(bool(used))

    comb_alert_arr = cls_alert_arr | det_alert_arr

    def _op_rates(idx):
        yb = y_tb[idx]
        ca = cls_alert_arr[idx]
        comb = comb_alert_arr[idx]
        tp_c = int((ca & yb).sum())
        fp_c = int((ca & ~yb).sum())
        fn_c = int((~ca & yb).sum())
        tn_c = int((~ca & ~yb).sum())
        tp_comb = int((comb & yb).sum())
        fp_comb = int((comb & ~yb).sum())
        fn_comb = int((~comb & yb).sum())
        tn_comb = int((~comb & ~yb).sum())
        tp_d = int(tp_arr[idx].sum())
        fp_d = int(fp_arr[idx].sum())
        gt_d = int(gt_arr[idx].sum())
        pos_d = int(yb.sum())
        neg_d = int((~yb).sum())
        return {
            'cls_thresh_sensitivity': tp_c / (tp_c + fn_c) if (tp_c + fn_c) else None,
            'cls_thresh_specificity': tn_c / (tn_c + fp_c) if (tn_c + fp_c) else None,
            'cls_thresh_precision': tp_c / (tp_c + fp_c) if (tp_c + fp_c) else None,
            'combined_tb_sensitivity': tp_comb / (tp_comb + fn_comb) if (tp_comb + fn_comb) else None,
            'combined_non_tb_review_rate': fp_comb / (fp_comb + tn_comb) if (fp_comb + tn_comb) else None,
            'detector_lesion_recall': tp_d / gt_d if gt_d else None,
            'detector_box_precision': tp_d / (tp_d + fp_d) if (tp_d + fp_d) else None,
            'detector_image_any_box': int((det_alert_arr[idx] & yb).sum()) / pos_d if pos_d else None,
            'detector_image_localized': int(loc_arr[idx].sum()) / pos_d if pos_d else None,
            'detector_neg_alert_rate': int((det_alert_arr[idx] & ~yb).sum()) / neg_d if neg_d else None,
        }

    full_idx = np.arange(n, dtype=np.int64)
    point_op = _op_rates(full_idx)
    op_series = defaultdict(list)

    for b_idx in boot_indices:
        m = _fast_multiclass_summary(y_all[b_idx], probs[b_idx])
        for k, v in m.items():
            cls_series[k].append(v)
        op = _op_rates(b_idx)
        for k, v in op.items():
            op_series[k].append(v)

    return {
        'n_images': n,
        'n_groups': len(group_list),
        'n_resamples': int(n_resamples),
        'seed': int(seed),
        'ci_level': float(ci_level),
        'classifier_argmax': {k: _summarize_distribution(cls_series[k], point_cls[k], ci_level) for k in point_cls},
        'operational_and_concordance': {k: _summarize_distribution(op_series[k], point_op[k], ci_level) for k in point_op},
    }


def _export_test_figures(y, probs, metrics, concordance_rows, output_dir, cls_thresh, det_thresh):
    """Exporta matriz de confusión, curvas ROC/PR y estados de concordancia en test (PNG/PDF/SVG)."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11.5, 'axes.spines.top': False, 'axes.spines.right': False}):
        # 1. Matriz de confusión en test.
        cm = np.asarray(metrics['confusion_matrix'])
        norm_cm = cm / cm.sum(1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
        for ax, mat, title, mx in zip(axes, [cm, norm_cm], ['Conteos (test)', 'Normalizada por clase real'], [cm.max(), 1.0]):
            im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=mx)
            for (i, j), val in np.ndenumerate(mat):
                ax.text(j, i, str(int(val)) if mat is cm else f'{val:.1%}', ha='center', va='center',
                        color='white' if val > mx * 0.55 else '#152536')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Clase predicha (argmax)', ylabel='Clase real', title=title)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.suptitle('ViT-B/16 (contrato congelado) · Matriz de confusión en test reservado', fontsize=12.5)
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output_dir / f'test_confusion_matrix.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # 2. Curvas ROC OVR y PR TB en test.
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
        colors = ['#2864AD', '#C16B19', '#287C60']
        for i, (cls_name, col) in enumerate(zip(CLASSES, colors)):
            c = binary_curve(y == i, probs[:, i])
            axes[0].plot(c['fpr'], c['tpr'], color=col, lw=1.8, label=f"{cls_name} (AUC={c['roc_auc']:.4f})")
        axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
        axes[0].set(xlabel='Tasa de falsos positivos', ylabel='Sensibilidad', title='ROC OVR en test reservado',
                    xlim=(0, 1), ylim=(0, 1.01))
        axes[0].grid(alpha=0.2)
        axes[0].legend(loc='lower right', fontsize=8.5)

        y_tb = (y == 2)
        s_tb = probs[:, 2]
        order = np.argsort(-s_tb, kind='stable')
        t_sorted = y_tb[order]
        s_sorted = s_tb[order]
        ends = np.r_[np.flatnonzero(np.diff(s_sorted)), len(t_sorted) - 1]
        tp = np.cumsum(t_sorted)[ends]
        rec = np.r_[0.0, tp / t_sorted.sum()]
        prec = np.r_[1.0, tp / (ends + 1)]
        ap_tb = metrics['per_class']['TB']['average_precision']
        axes[1].plot(rec, prec, color='#7B4B94', lw=2.0, label=f'TB (AP={ap_tb:.4f})')
        axes[1].set(xlabel='Sensibilidad TB', ylabel='Precisión TB', title='Precisión-Recall TB en test reservado',
                    xlim=(0, 1.01), ylim=(0, 1.02))
        axes[1].grid(alpha=0.2)
        axes[1].legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output_dir / f'test_roc_pr.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # 3. Resumen de concordancia por clase real.
        states = ['both_alert', 'classifier_only_review', 'detector_only_review', 'neither_alert']
        mat_conc = np.zeros((3, 4), dtype=int)
        for r in concordance_rows:
            mat_conc[int(r['label']), states.index(r['state'])] += 1
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        im = ax.imshow(mat_conc, cmap='YlGnBu')
        for (i, j), v in np.ndenumerate(mat_conc):
            ax.text(j, i, str(int(v)), ha='center', va='center',
                    color='white' if v > mat_conc.max() * 0.55 else '#111111')
        ax.set(xticks=range(4), yticks=range(3),
               xticklabels=['Ambas alertas', 'Solo clasificador', 'Solo detector', 'Ninguna alerta'],
               yticklabels=CLASSES,
               title=f'Concordancia congelada en test (p_TB ≥ {cls_thresh:.2f}, YOLO11n ≥ {det_thresh:.2f})')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output_dir / f'test_concordance.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def evaluate_frozen_test(project, dataset_root, drive_base, contract, output_dir, device='cuda', batch_size=8, lock_dir=None):
    """Evaluación única e irrepetible sobre manifests/test.csv bajo contrato congelado."""
    import torch
    from ultralytics import YOLO

    project, dataset_root, drive_base, output_dir = map(Path, (project, dataset_root, drive_base, output_dir))
    if (output_dir / 'test_evaluation_summary.json').exists():
        raise RuntimeError('La evaluación de test ya fue ejecutada en este directorio.')
    if lock_dir is not None:
        lock_file = Path(lock_dir) / 'frozen_test_lock.json'
        if lock_file.exists():
            raise RuntimeError(f'Ya existe bloqueo de evaluación única en {lock_file}. No re-evaluar test.')

    verify_frozen_contract(project)
    integrity = validate_manifests(project, dataset_root, all_image_hashes=True)
    data_cfg = load_config(safe_path(project, contract['preprocessing']['data_config']))
    test_rows = read_csv(project / 'manifests/test.csv')
    if len(test_rows) != 1241 or any(r['split'] != 'test' for r in test_rows):
        raise ValueError('manifests/test.csv inválido.')

    cls_cfg = contract['classifier']
    det_cfg = contract['detector']
    cls_thresh = float(cls_cfg['operational_tb_threshold'])
    det_thresh = float(det_cfg['confidence_threshold'])

    # 1. Clasificador congelado (ViT-B/16).
    cls_spec = {
        'architecture': cls_cfg['architecture'],
        'run_id': cls_cfg['run_id'],
        'protocol_config': cls_cfg['protocol_config'],
        'sha256': cls_cfg['checkpoint_sha256'],
        'requires_oflex': bool(cls_cfg.get('requires_oflex', False)),
    }
    model = restore_any_classifier(project, drive_base, cls_spec, device=device, require_oflex=cls_spec['requires_oflex'])
    test_ds = TBXDataset(dataset_root, project / 'manifests/test.csv', data_cfg, training=False)
    probs_list = []
    labels_list = []
    with torch.inference_mode():
        for offset in range(0, len(test_ds), int(batch_size)):
            batch_items = [test_ds[i] for i in range(offset, min(offset + int(batch_size), len(test_ds)))]
            x = torch.stack([torch.from_numpy(item[0]) for item in batch_items]).to(device)
            p = model(x).softmax(dim=1).cpu().numpy()
            probs_list.append(p)
            labels_list.extend(item[1] for item in batch_items)
    probs = np.vstack(probs_list)
    y = np.asarray(labels_list, dtype=np.int64)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cls_metrics = classification_metrics(y, probs, CLASSES)
    cls_thresh_point = tb_threshold_table(y, probs, [cls_thresh])[0]
    cls_thresh_ref = tb_threshold_table(y, probs, [0.1, 0.3, 0.5])

    # 2. Detector congelado (YOLO11n).
    det_weights, det_sha, _ = verify_detector(drive_base, {
        'run_id': det_cfg['run_id'],
        'observed_sha256': det_cfg['weights_sha256'],
    })
    detector = YOLO(str(det_weights))
    if detector.names != {0: 'tb_lesion'}:
        raise ValueError(f'Clases inesperadas en detector: {detector.names}')
    det_opts = dict(det_cfg['inference_options'], device=0 if str(device).startswith('cuda') else 'cpu', save=False, verbose=False)
    det_predictions = {}
    for offset in range(0, len(test_rows), int(batch_size)):
        batch = test_rows[offset:offset + int(batch_size)]
        results = detector.predict(source=[str(safe_path(dataset_root, r['relative_path'])) for r in batch], **det_opts)
        if len(results) != len(batch):
            raise ValueError('Batch incompleto del detector en test.')
        for row, res in zip(batch, results):
            if Path(res.path).resolve() != safe_path(dataset_root, row['relative_path']):
                raise ValueError('Orden inconsistente de imágenes del detector en test.')
            det_predictions[row['relative_path']] = [
                dict(xyxy=xy, confidence=float(score))
                for xy, score in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist())
            ]
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    det_point = evaluate_detector_on_split(test_rows, det_predictions, [det_thresh], expected_split='test', match_iou=0.5)[0]
    det_ref = evaluate_detector_on_split(test_rows, det_predictions, [0.1, 0.25, 0.5], expected_split='test', match_iou=0.5)

    # 3. Concordancia congelada.
    concordance_rows = []
    pred_rows = []
    state_counts = defaultdict(int)
    tb_state_counts = defaultdict(int)
    non_tb_review = health_review = sick_review = tb_review = 0
    for row, label, p in zip(test_rows, y, probs):
        path = row['relative_path']
        pred_cls = int(p.argmax())
        pred_rows.append({
            'relative_path': path,
            'group_id': row['group_id'],
            'class_name': row['class_name'],
            'label': int(label),
            'prediction': pred_cls,
            'predicted_class': CLASSES[pred_cls],
            'p_health': float(p[0]),
            'p_sick': float(p[1]),
            'p_TB': float(p[2]),
        })
        boxes = det_predictions[path]
        flags = concordance(float(p[2]), boxes, cls_thresh, det_thresh)
        state = flags['state']
        state_counts[state] += 1
        if row['class_name'] == 'TB':
            tb_state_counts[state] += 1
            if flags['review_required']:
                tb_review += 1
        else:
            if flags['review_required']:
                non_tb_review += 1
                if row['class_name'] == 'health':
                    health_review += 1
                else:
                    sick_review += 1
        concordance_rows.append({
            'relative_path': path,
            'group_id': row['group_id'],
            'class_name': row['class_name'],
            'label': int(label),
            'argmax_prediction': CLASSES[pred_cls],
            'p_TB': float(p[2]),
            'detector_box_count': sum(1 for b in boxes if b['confidence'] >= det_thresh),
            'detector_max_conf': max((b['confidence'] for b in boxes), default=0.0),
            **flags,
            'classifier_threshold': cls_thresh,
            'detector_threshold': det_thresh,
            'decision_status': 'frozen_contract_evaluated_once',
        })

    n_tb = sum(1 for r in test_rows if r['class_name'] == 'TB')
    n_health = sum(1 for r in test_rows if r['class_name'] == 'health')
    n_sick = sum(1 for r in test_rows if r['class_name'] == 'sick')
    n_non_tb = n_health + n_sick
    concordance_summary = {
        'classifier_threshold': cls_thresh,
        'detector_threshold': det_thresh,
        'state_counts': dict(state_counts),
        'tb_state_counts': dict(tb_state_counts),
        'combined_tb_sensitivity': tb_review / n_tb,
        'tb_missed_by_both': n_tb - tb_review,
        'non_tb_review_count': non_tb_review,
        'non_tb_review_rate': non_tb_review / n_non_tb,
        'health_review_count': health_review,
        'health_review_rate': health_review / n_health,
        'sick_review_count': sick_review,
        'sick_review_rate': sick_review / n_sick,
        'total_review_count': tb_review + non_tb_review,
        'total_review_rate': (tb_review + non_tb_review) / len(test_rows),
    }

    # 4. Bootstrap agrupado en test.
    bootstrap_ci = _bootstrap_frozen_test(
        test_rows, probs, det_predictions, cls_thresh, det_thresh, match_iou=0.5, n_resamples=2000, seed=20261019
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'test_predictions.csv', pred_rows)
    write_csv(output_dir / 'test_concordance.csv', concordance_rows)
    atomic_json(output_dir / 'test_detector_predictions.json', det_predictions)
    atomic_json(output_dir / 'test_bootstrap_ci.json', bootstrap_ci)
    _export_test_figures(y, probs, cls_metrics, concordance_rows, output_dir, cls_thresh, det_thresh)

    summary = {
        'schema': 'phase5_frozen_test_evaluation_v1',
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'test',
        'n': len(test_rows),
        'single_evaluation_enforced': True,
        'decisions_frozen': True,
        'test_evaluated': True,
        'contract_classifier': cls_cfg,
        'contract_detector': {**det_cfg, 'verified_sha256': det_sha},
        'manifest_integrity': integrity,
        'classifier_metrics': cls_metrics,
        'classifier_operational_threshold': cls_thresh_point,
        'classifier_threshold_reference_points': cls_thresh_ref,
        'detector_operational_point': det_point,
        'detector_threshold_reference_points': det_ref,
        'concordance_summary': concordance_summary,
        'bootstrap_ci': bootstrap_ci,
    }
    atomic_json(output_dir / 'test_evaluation_summary.json', summary)
    if lock_dir is not None:
        atomic_json(Path(lock_dir) / 'frozen_test_lock.json', {
            'locked_at_utc': summary['evaluated_at_utc'],
            'output_dir': str(output_dir),
            'classifier_run_id': cls_cfg['run_id'],
            'detector_run_id': det_cfg['run_id'],
            'test_macro_f1': cls_metrics['macro_f1'],
            'test_combined_tb_sensitivity': concordance_summary['combined_tb_sensitivity'],
        })
    return summary


def audit_external_vs_train_val(project, external_root=None, tbx_root=None, dhash_max=2, corr_min=0.97, mae_max=10.0):
    """Audita qué imágenes de Montgomery/Shenzhen son duplicados reales de train o validation.

    Nunca abre manifests/test.csv ni imgs/test/.
    """
    project = Path(project)
    ext_base = Path(external_root) if external_root else (project / 'external_data')
    tbx_base = Path(tbx_root) if tbx_root else (project / 'TBX11K')
    members = {r['relative_path']: r for r in read_csv(project / 'manifests/all_labeled_members.csv')}
    ext_manifests = {
        ds: read_csv(project / 'manifests/external' / f'{ds}.csv')
        for ds in ('montgomery', 'shenzhen')
    }

    @functools.lru_cache(maxsize=16384)
    def _small_gray(path_str):
        with Image.open(path_str) as im:
            return np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BILINEAR), dtype=np.uint8)

    overlaps = defaultdict(list)
    any_extra_mc_sz = defaultdict(int)
    for row in read_csv(project / 'reports/external_sources/overlap_candidates.csv'):
        d = int(row['dhash_hamming'])
        ds = row['external_dataset']
        ext_img = row['external_image']
        p_rel = 'imgs/' + row['tbx11k_path']
        if p_rel.startswith('imgs/extra/mc+shenzhen') and d <= 1:
            any_extra_mc_sz[(ds, ext_img)] += 1
        elif p_rel in members and members[p_rel]['split'] in {'train', 'validation'} and d <= dhash_max:
            ext_path = ext_base / ds / 'CXR_png' / ext_img
            tbx_path = tbx_base / p_rel
            if ext_path.is_file() and tbx_path.is_file():
                a = _small_gray(str(ext_path)).astype(np.float32)
                b = _small_gray(str(tbx_path)).astype(np.float32)
                corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                mae = float(np.abs(a - b).mean())
                if corr >= corr_min or mae <= mae_max:
                    overlaps[(ds, ext_img)].append({
                        'tbx11k_path': p_rel,
                        'split': members[p_rel]['split'],
                        'class_name': members[p_rel]['class_name'],
                        'dhash_hamming': d,
                        'corr_128': corr,
                        'mae_128': mae,
                    })

    audit_rows = []
    summary = {}
    for ds, rows in ext_manifests.items():
        clean_count = tv_overlap_count = extra_count = 0
        for r in rows:
            img_name = Path(r['relative_path']).name
            matches = overlaps.get((ds, img_name), [])
            in_extra = any_extra_mc_sz.get((ds, img_name), 0) > 0
            if matches:
                tv_overlap_count += 1
            else:
                clean_count += 1
            if in_extra:
                extra_count += 1
            audit_rows.append({
                'dataset': ds,
                'relative_path': r['relative_path'],
                'binary_label': int(r['binary_label']),
                'train_val_pixel_overlap': bool(matches),
                'overlap_count_train_val': len(matches),
                'in_tbx11k_extra_mc_shenzhen_d1': in_extra,
                'eligible_clean_vs_train_val': not bool(matches),
                'matches_json': json.dumps(matches),
            })
        summary[ds] = {
            'total': len(rows),
            'train_val_pixel_overlaps': tv_overlap_count,
            'clean_vs_train_val': clean_count,
            'matched_extra_mc_shenzhen_d1': extra_count,
        }
    return {'summary': summary, 'rows': audit_rows}


def _binary_metrics_subset(y_bin, score, threshold):
    y_bin = np.asarray(y_bin, dtype=bool)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int((pred & y_bin).sum())
    fp = int((pred & ~y_bin).sum())
    fn = int((~pred & y_bin).sum())
    tn = int((~pred & ~y_bin).sum())
    curve = binary_curve(y_bin, score)
    return {
        'n': int(len(y_bin)),
        'positives': int(y_bin.sum()),
        'negatives': int((~y_bin).sum()),
        'threshold': float(threshold),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'accuracy': float((tp + tn) / len(y_bin)) if len(y_bin) else None,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) else None,
        'specificity': float(tn / (tn + fp)) if (tn + fp) else None,
        'precision': float(tp / (tp + fp)) if (tp + fp) else None,
        'roc_auc': curve['roc_auc'],
        'average_precision': curve['average_precision'],
    }


def evaluate_frozen_external(project, external_root, drive_base, contract, output_dir, overlap_audit=None, device='cuda', batch_size=8):
    """Evalúa el contrato congelado sobre Montgomery y Shenzhen con mapeo binario compatible."""
    import torch
    from ultralytics import YOLO

    project, external_root, drive_base, output_dir = map(Path, (project, external_root, drive_base, output_dir))
    verify_frozen_contract(project)
    data_cfg = load_config(safe_path(project, contract['preprocessing']['data_config']))
    transform = RadiographTransform(data_cfg, training=False)

    cls_cfg = contract['classifier']
    det_cfg = contract['detector']
    cls_thresh = float(cls_cfg['operational_tb_threshold'])
    det_thresh = float(det_cfg['confidence_threshold'])

    if overlap_audit is None:
        audit_map = {}
        for ds in ('montgomery', 'shenzhen'):
            for r in read_csv(project / 'manifests/external' / f'{ds}.csv'):
                audit_map[(ds, r['relative_path'])] = True
    else:
        def _as_bool(v):
            return v if isinstance(v, bool) else str(v).strip().lower() == 'true'
        audit_map = {(r['dataset'], r['relative_path']): _as_bool(r['eligible_clean_vs_train_val']) for r in overlap_audit['rows']}

    cls_spec = {
        'architecture': cls_cfg['architecture'],
        'run_id': cls_cfg['run_id'],
        'protocol_config': cls_cfg['protocol_config'],
        'sha256': cls_cfg['checkpoint_sha256'],
        'requires_oflex': bool(cls_cfg.get('requires_oflex', False)),
    }
    model = restore_any_classifier(project, drive_base, cls_spec, device=device, require_oflex=cls_spec['requires_oflex'])
    det_weights, _, _ = verify_detector(drive_base, {
        'run_id': det_cfg['run_id'],
        'observed_sha256': det_cfg['weights_sha256'],
    })
    detector = YOLO(str(det_weights))
    det_opts = dict(det_cfg['inference_options'], device=0 if str(device).startswith('cuda') else 'cpu', save=False, verbose=False)

    datasets_out = {}
    all_pred_rows = []
    try:
        for ds in ('montgomery', 'shenzhen'):
            rows = read_csv(project / 'manifests/external' / f'{ds}.csv')
            probs_list = []
            for offset in range(0, len(rows), int(batch_size)):
                batch = rows[offset:offset + int(batch_size)]
                tensors = []
                for r in batch:
                    with Image.open(safe_path(external_root, r['relative_path'])) as im:
                        tensors.append(torch.from_numpy(transform(im)))
                with torch.inference_mode():
                    x = torch.stack(tensors).to(device)
                    probs_list.append(model(x).softmax(dim=1).cpu().numpy())
            probs = np.vstack(probs_list)

            # Detector requiere archivos RGB 512px letterbox o rutas originales; aquí pasamos temporales letterbox 512 para alinear preprocesamiento.
            det_boxes_by_rel = {}
            for offset in range(0, len(rows), int(batch_size)):
                batch = rows[offset:offset + int(batch_size)]
                pil_batch = []
                for r in batch:
                    with Image.open(safe_path(external_root, r['relative_path'])) as im:
                        pil_batch.append(transform.image(im))
                results = detector.predict(source=pil_batch, **det_opts)
                for r, res in zip(batch, results):
                    det_boxes_by_rel[r['relative_path']] = [
                        dict(xyxy=xy, confidence=float(score))
                        for xy, score in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist())
                    ]

            ds_records = []
            for r, p in zip(rows, probs):
                rel = r['relative_path']
                y_bin = int(r['binary_label'])
                boxes = det_boxes_by_rel[rel]
                flags = concordance(float(p[2]), boxes, cls_thresh, det_thresh)
                eligible = audit_map[(ds, rel)]
                rec = {
                    'dataset': ds,
                    'relative_path': rel,
                    'binary_label': y_bin,
                    'eligible_clean_vs_train_val': eligible,
                    'argmax_prediction': CLASSES[int(p.argmax())],
                    'p_health': float(p[0]),
                    'p_sick': float(p[1]),
                    'p_TB': float(p[2]),
                    'p_abnormal_1_minus_health': float(1.0 - p[0]),
                    'detector_box_count': sum(1 for b in boxes if b['confidence'] >= det_thresh),
                    'detector_max_conf': max((b['confidence'] for b in boxes), default=0.0),
                    **flags,
                }
                ds_records.append(rec)
                all_pred_rows.append(rec)

            def _eval_slice(slice_rows):
                yb = [x['binary_label'] for x in slice_rows]
                ptb = [x['p_TB'] for x in slice_rows]
                pab = [x['p_abnormal_1_minus_health'] for x in slice_rows]
                comb = [int(x['review_required']) for x in slice_rows]
                yb_arr = np.asarray(yb, dtype=bool)
                comb_arr = np.asarray(comb, dtype=bool)
                tp_c = int((comb_arr & yb_arr).sum())
                fp_c = int((comb_arr & ~yb_arr).sum())
                fn_c = int((~comb_arr & yb_arr).sum())
                tn_c = int((~comb_arr & ~yb_arr).sum())
                return {
                    'p_TB_frozen_threshold': _binary_metrics_subset(yb, ptb, cls_thresh),
                    'p_abnormal_1_minus_health_ref_0_50': _binary_metrics_subset(yb, pab, 0.50),
                    'concordance_review_required': {
                        'n': len(slice_rows),
                        'tp': tp_c,
                        'fp': fp_c,
                        'fn': fn_c,
                        'tn': tn_c,
                        'sensitivity': tp_c / (tp_c + fn_c) if (tp_c + fn_c) else None,
                        'specificity': tn_c / (tn_c + fp_c) if (tn_c + fp_c) else None,
                        'precision': tp_c / (tp_c + fp_c) if (tp_c + fp_c) else None,
                    },
                }

            clean_slice = [x for x in ds_records if x['eligible_clean_vs_train_val']]
            datasets_out[ds] = {
                'all_images_with_overlap_caveat': _eval_slice(ds_records),
                'clean_vs_train_val_subset': _eval_slice(clean_slice) if clean_slice else None,
            }
    finally:
        del model, detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'external_predictions.csv', all_pred_rows)
    ext_summary = {
        'schema': 'phase5_frozen_external_evaluation_v1',
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'binary_mapping_note': 'Montgomery/Shenzhen solo tienen etiqueta binaria (0=normal, 1=abnormal_tb_consistent). Se reporta alerta congelada p_TB >= 0.30, concordancia combinada y referencia p_abnormal = 1 - p_health.',
        'overlap_caveat': 'TBX11K incluye imágenes de Montgomery/Shenzhen en extra/mc+shenzhen y test oficial; solo clean_vs_train_val_subset descarta solapamiento contra train/validation.',
        'datasets': datasets_out,
    }
    atomic_json(output_dir / 'external_evaluation_summary.json', ext_summary)
    return ext_summary


def export_deployment_bundle(project, drive_base, contract, output_dir):
    """Exporta contrato congelado y pesos verificados para api/ y web/."""
    project, drive_base, output_dir = map(Path, (project, drive_base, output_dir))
    verify_frozen_contract(project)
    output_dir.mkdir(parents=True, exist_ok=True)

    cls_cfg = contract['classifier']
    det_cfg = contract['detector']
    cls_src = drive_base / 'checkpoints' / cls_cfg['run_id'] / 'best.pt'
    det_src = drive_base / 'experiments' / det_cfg['run_id'] / 'training/weights/best.pt'
    if sha256(cls_src) != cls_cfg['checkpoint_sha256']:
        raise ValueError('SHA-256 de clasificador no coincide con el contrato congelado.')
    if sha256(det_src) != det_cfg['weights_sha256']:
        raise ValueError('SHA-256 de detector no coincide con el contrato congelado.')

    cls_dst = output_dir / f"{cls_cfg['architecture']}_best.pt"
    det_dst = output_dir / f"{det_cfg['architecture']}_best.pt"
    shutil.copy2(cls_src, cls_dst)
    shutil.copy2(det_src, det_dst)
    contract_dst = output_dir / 'phase5_frozen_contract.json'
    shutil.copy2(project / 'configs/phase5_frozen_contract.json', contract_dst)
    data_dst = output_dir / 'data_config.json'
    shutil.copy2(project / 'configs/data.json', data_dst)

    manifest = {
        'schema': 'tbx_deployment_bundle_v1',
        'exported_at_utc': datetime.now(timezone.utc).isoformat(),
        'classifier_weights': cls_dst.name,
        'classifier_sha256': sha256(cls_dst),
        'detector_weights': det_dst.name,
        'detector_sha256': sha256(det_dst),
        'contract_file': contract_dst.name,
        'contract_sha256': sha256(contract_dst),
        'data_config_file': data_dst.name,
        'data_config_sha256': sha256(data_dst),
    }
    atomic_json(output_dir / 'deployment_manifest.json', manifest)
    return manifest


def stage_verified_external(project, external_cache, datasets=('montgomery', 'shenzhen'), max_workers=6):
    """Descarga y verifica SHA-256 de Montgomery/Shenzhen desde sus URLs oficiales NLM."""
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    project, external_cache = Path(project), Path(external_cache)
    tasks = []
    for ds in datasets:
        for row in read_csv(project / 'manifests/external' / f'{ds}.csv'):
            url = row['source_url']
            if not url.startswith('https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/'):
                raise ValueError('URL externa fuera de dominio oficial NLM: ' + url)
            target = safe_path(external_cache, row['relative_path'])
            tasks.append((url, target, row['sha256']))

    def _fetch_one(item):
        url, target, expected_sha = item
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256(target) == expected_sha:
            return target
        for attempt in range(4):
            try:
                tmp = target.with_suffix(target.suffix + '.partial')
                with urllib.request.urlopen(url, timeout=90) as resp, tmp.open('wb') as dst:
                    while chunk := resp.read(1024 * 1024):
                        dst.write(chunk)
                if sha256(tmp) != expected_sha:
                    tmp.unlink(missing_ok=True)
                    raise ValueError('SHA-256 externo no coincide: ' + url)
                tmp.replace(target)
                return target
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        futures = [pool.submit(_fetch_one, item) for item in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if idx % 100 == 0 or idx == len(tasks):
                print(f'Externos verificados: {idx}/{len(tasks)}', flush=True)
    return external_cache


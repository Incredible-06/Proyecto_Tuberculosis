"""Diagnóstico completo de validation para Fase 5 (Parte 1).

Cubre: verificación y restauración unificada de los 5 clasificadores y YOLO11n,
diagnóstico extendido del detector, concordancia exploratoria, selección y
paneles visuales de errores/cajas, coste computacional comparable e
incertidumbre por bootstrap estratificado y agrupado por group_id.

No lee ni evalúa test ni conjuntos externos; no congela decisiones.
"""
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from tbxdata.common import read_csv, safe_path, sha256, write_csv
from .architectures import build_classifier, scan_backend
from .hybrid_vit_experiment import ConvNeXtViTHybridClassifier
from .metrics import binary_curve, classification_metrics
from .phase5 import (
    CLASSES,
    _iou,
    concordance,
    detector_threshold_table,
    read_validation_predictions,
    tb_threshold_table,
    validation_rows,
    verify_provenance,
)
from .protocol import atomic_json
from .vit_experiment import ViTClassifier

POINTER_ARCHS = ('hybrid_cross_attention', 'vit_b_16', 'hybrid_convnext_vit')


def load_diagnostic_config(project):
    """Carga y valida la configuración exclusiva de validation."""
    cfg = json.loads((Path(project) / 'configs/phase5_validation_diagnostic.json').read_text(encoding='utf-8'))
    if (cfg.get('scope') != 'validation_only' or cfg.get('decisions_frozen') is not False
            or cfg.get('test_evaluated') is not False or cfg.get('external_evaluated') is not False):
        raise ValueError('La configuración diagnóstica debe restringirse estrictamente a validation.')
    return cfg


def verify_any_classifier(project, base, spec, rows, check_checkpoint=True, source_files=None):
    """Verifica summary, procedencia, protocolo, predicciones y best.pt de cualquiera de los 5 modelos."""
    project, base = Path(project), Path(base)
    folder = base / 'experiments' / spec['run_id']
    paths = source_files or {}
    summary_path = Path(paths.get('summary', folder / 'summary.json'))
    preds_path = Path(paths.get('predictions', folder / 'validation_predictions.csv'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if (summary['run_id'] != spec['run_id'] or summary['architecture'] != spec['architecture']
            or summary['status'] != 'completed_controlled' or not summary['training_executed']
            or summary['test_evaluated'] or summary['external_evaluated']):
        raise ValueError('Corrida fuera del protocolo: ' + spec['run_id'])
    expected_protocol = json.loads((safe_path(project, spec['protocol_config'])).read_text(encoding='utf-8'))
    if summary['protocol'] != expected_protocol:
        raise ValueError('Protocolo distinto para ' + spec['architecture'])
    verify_provenance(project, summary['provenance'])
    if check_checkpoint:
        checkpoint = base / 'checkpoints' / spec['run_id'] / 'best.pt'
        actual = sha256(checkpoint)
        if actual != spec['sha256'] or actual != summary['checkpoint_sha256']:
            raise ValueError('Pesos alterados: ' + str(checkpoint))
    elif spec['sha256'] != summary['checkpoint_sha256']:
        raise ValueError('SHA-256 de summary no coincide con el registro: ' + spec['run_id'])
    ordered, y, p, metrics = read_validation_predictions(preds_path, rows)
    if (metrics['confusion_matrix'] != summary['validation']['confusion_matrix']
            or not np.isclose(metrics['macro_f1'], summary['validation']['macro_f1'], atol=1e-12, rtol=0)
            or metrics['n'] != summary['validation']['n']):
        raise ValueError('Métricas no reproducidas para ' + spec['run_id'])
    return summary, ordered, y, p, metrics


def _build_untrained_classifier(architecture, image_size=512):
    if architecture in {'convnext_tiny', 'vmamba_tiny', 'hybrid_cross_attention'}:
        return build_classifier(architecture, pretrained=False)
    if architecture == 'vit_b_16':
        return ViTClassifier(pretrained=False, image_size=image_size)
    if architecture == 'hybrid_convnext_vit':
        return ConvNeXtViTHybridClassifier(pretrained=False, image_size=image_size)
    raise ValueError('Arquitectura no soportada: ' + str(architecture))


def restore_any_classifier(project, base, spec, device='cuda', require_oflex=True):
    """Restaura estrictamente best.pt (pretrained=False) de cualquiera de los 5 clasificadores."""
    import torch

    project, base = Path(project), Path(base)
    rows = validation_rows(project)
    summary, _, _, _, _ = verify_any_classifier(project, base, spec, rows, check_checkpoint=True)
    if spec.get('requires_oflex', False) and require_oflex and scan_backend() != 'oflex':
        raise RuntimeError('Se requiere selective scan oflex para ' + spec['architecture'])
    data_cfg = json.loads((project / 'configs/data.json').read_text(encoding='utf-8'))
    checkpoint = base / 'checkpoints' / spec['run_id'] / 'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload['architecture'] != spec['architecture'] or payload['run_id'] != spec['run_id']
            or payload['epoch'] != summary['best_epoch'] or payload['classes'] != CLASSES
            or payload['config'] != summary['protocol'] or payload['provenance'] != summary['provenance']
            or payload['data_config'] != data_cfg):
        raise ValueError('Contrato interno del checkpoint inconsistente: ' + spec['run_id'])
    model = _build_untrained_classifier(spec['architecture'], image_size=int(data_cfg['image_size']))
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval()


def verify_detector(base, det_spec):
    """Verifica el peso best.pt y resumen del detector YOLO11n contra el SHA observado."""
    base = Path(base)
    folder = base / 'experiments' / det_spec['run_id']
    weights = folder / 'training/weights/best.pt'
    summary = json.loads((folder / 'detector_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'trained_validation_available' or summary['test_evaluated']:
        raise ValueError('Estado del detector fuera del protocolo.')
    actual = sha256(weights)
    if actual != det_spec['observed_sha256']:
        raise ValueError(f'Hash del detector distinto al observado: {actual} != {det_spec["observed_sha256"]}')
    return weights, actual, summary


def detector_detailed_metrics(rows, predictions, thresholds, match_iou=0.5):
    """Enriquece detector_threshold_table con precisión por caja y desglose health/sick."""
    base_table = detector_threshold_table(rows, predictions, thresholds, match_iou=match_iou)
    health_rows = [r for r in rows if r['class_name'] == 'health']
    sick_rows = [r for r in rows if r['class_name'] == 'sick']
    tb_rows = [r for r in rows if r['class_name'] == 'TB']
    out = []
    for entry in base_table:
        t = entry['threshold']
        tp = entry['matched_boxes']
        fp = entry['false_positive_boxes']
        total_pred = tp + fp
        health_alerts = sum(1 for r in health_rows if any(b['confidence'] >= t for b in predictions[r['relative_path']]))
        sick_alerts = sum(1 for r in sick_rows if any(b['confidence'] >= t for b in predictions[r['relative_path']]))
        enriched = dict(entry)
        enriched['total_predicted_boxes'] = total_pred
        enriched['box_precision'] = (tp / total_pred) if total_pred else None
        enriched['health_images'] = len(health_rows)
        enriched['sick_images'] = len(sick_rows)
        enriched['tb_images'] = len(tb_rows)
        enriched['health_alert_count'] = health_alerts
        enriched['sick_alert_count'] = sick_alerts
        enriched['health_alert_rate'] = (health_alerts / len(health_rows)) if health_rows else None
        enriched['sick_alert_rate'] = (sick_alerts / len(sick_rows)) if sick_rows else None
        out.append(enriched)
    return out


def export_detector_diagnostic_figures(det_table, output_dir):
    """Exporta curvas por umbral y FROC de YOLO11n en PNG 300 dpi, PDF, SVG y CSV."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'detector_thresholds_detailed.csv', det_table)
    thresholds = [r['threshold'] for r in det_table]
    any_box = [r['image_sensitivity_any_box'] for r in det_table]
    localized = [r['image_sensitivity_localized'] for r in det_table]
    lesion_rec = [r['lesion_recall'] for r in det_table]
    box_prec = [r['box_precision'] if r['box_precision'] is not None else float('nan') for r in det_table]
    neg_alert = [r['negative_image_alert_rate'] for r in det_table]
    fp_per_img = [r['fp_boxes_per_image'] for r in det_table]

    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11.5, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
        axes[0].plot(thresholds, any_box, 'o-', color='#2864AD', label='Sens. imagen (cualquier caja)')
        axes[0].plot(thresholds, localized, 's-', color='#287C60', label='Sens. imagen (IoU ≥ 0,5)')
        axes[0].plot(thresholds, lesion_rec, '^-', color='#C16B19', label='Recall por lesión')
        axes[0].plot(thresholds, box_prec, 'd--', color='#7B4B94', label='Precisión por caja')
        axes[0].plot(thresholds, neg_alert, 'x:', color='#B8334A', label='Falsa alarma en no-TB')
        axes[0].set(xlabel='Umbral de confianza YOLO11n', ylabel='Proporción', ylim=(-0.02, 1.04),
                    title='Detector YOLO11n · Métricas por umbral (validation)')
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8, loc='best')

        axes[1].plot(fp_per_img, lesion_rec, 'o-', color='#2864AD', lw=1.8)
        for row in det_table:
            axes[1].annotate(f"{row['threshold']}", (row['fp_boxes_per_image'], row['lesion_recall']),
                             textcoords='offset points', xytext=(4, 4), fontsize=8)
        axes[1].set(xlabel='Cajas FP por imagen (1.241 imágenes)', ylabel='Recall por lesión (IoU ≥ 0,5)',
                    ylim=(-0.02, 1.04), title='Curva FROC exploratoria (validation)')
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output_dir / f'detector_diagnostic.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return output_dir / 'detector_thresholds_detailed.csv'


def concordance_grid_analysis(
    rows,
    predictions_by_arch,
    det_predictions,
    classifier_thresholds,
    detector_thresholds,
    pointer_archs=POINTER_ARCHS,
    default_pair=(0.5, 0.25),
):
    """Evalúa la regla de concordancia exploratoria sobre la rejilla de umbrales para los 3 punteros."""
    by_path = {r['relative_path']: r for r in rows}
    grid_summary = []
    default_rows = []
    for arch in pointer_archs:
        ordered = predictions_by_arch[arch]
        for cls_t in classifier_thresholds:
            for det_t in detector_thresholds:
                counts = defaultdict(int)
                tb_review = non_tb_review = health_review = sick_review = 0
                tb_cls_only = tb_det_only = tb_both = 0
                n_tb = n_health = n_sick = 0
                for pred in ordered:
                    path = pred['relative_path']
                    row = by_path[path]
                    flags = concordance(float(pred['p_TB']), det_predictions[path], cls_t, det_t)
                    state = flags['state']
                    counts[state] += 1
                    cls_name = row['class_name']
                    if cls_name == 'TB':
                        n_tb += 1
                        if flags['review_required']:
                            tb_review += 1
                        if state == 'both_alert':
                            tb_both += 1
                        elif state == 'classifier_only_review':
                            tb_cls_only += 1
                        elif state == 'detector_only_review':
                            tb_det_only += 1
                    elif cls_name == 'health':
                        n_health += 1
                        if flags['review_required']:
                            health_review += 1
                            non_tb_review += 1
                    else:
                        n_sick += 1
                        if flags['review_required']:
                            sick_review += 1
                            non_tb_review += 1
                    if np.isclose(cls_t, default_pair[0]) and np.isclose(det_t, default_pair[1]):
                        default_rows.append({
                            'architecture': arch,
                            'relative_path': path,
                            'group_id': row['group_id'],
                            'class_name': cls_name,
                            'label': int(row['label']),
                            'p_TB': float(pred['p_TB']),
                            'argmax_prediction': CLASSES[int(pred['prediction'])],
                            'detector_box_count': sum(1 for b in det_predictions[path] if b['confidence'] >= det_t),
                            'detector_max_conf': max((b['confidence'] for b in det_predictions[path]), default=0.0),
                            **flags,
                            'classifier_threshold': float(cls_t),
                            'detector_threshold': float(det_t),
                            'decision_status': 'exploratory_not_frozen',
                        })
                n_total = len(ordered)
                n_non_tb = n_health + n_sick
                grid_summary.append({
                    'architecture': arch,
                    'classifier_threshold': float(cls_t),
                    'detector_threshold': float(det_t),
                    'both_alert': counts['both_alert'],
                    'classifier_only_review': counts['classifier_only_review'],
                    'detector_only_review': counts['detector_only_review'],
                    'neither_alert': counts['neither_alert'],
                    'tb_both_alert': tb_both,
                    'tb_classifier_only': tb_cls_only,
                    'tb_detector_only_added': tb_det_only,
                    'tb_missed_by_both': n_tb - tb_review,
                    'combined_tb_sensitivity': tb_review / n_tb if n_tb else None,
                    'classifier_tb_sensitivity': (tb_both + tb_cls_only) / n_tb if n_tb else None,
                    'non_tb_review_count': non_tb_review,
                    'non_tb_review_rate': non_tb_review / n_non_tb if n_non_tb else None,
                    'health_review_rate': health_review / n_health if n_health else None,
                    'sick_review_rate': sick_review / n_sick if n_sick else None,
                    'total_review_count': tb_review + non_tb_review,
                    'total_review_rate': (tb_review + non_tb_review) / n_total if n_total else None,
                })
    return grid_summary, default_rows


def _has_localized_match(gt_boxes_xywh, det_boxes, det_threshold, match_iou=0.5):
    gt_xyxy = [[b['bbox'][0], b['bbox'][1], b['bbox'][0] + b['bbox'][2], b['bbox'][1] + b['bbox'][3]]
               for b in gt_boxes_xywh]
    if not gt_xyxy:
        return False
    for box in det_boxes:
        if box['confidence'] >= det_threshold and any(_iou(box['xyxy'], g) >= match_iou for g in gt_xyxy):
            return True
    return False


def select_visual_review_cases(
    rows,
    predictions_by_arch,
    det_predictions,
    pointer_archs=POINTER_ARCHS,
    cls_threshold=0.5,
    det_threshold=0.25,
    max_det_only_non_tb=15,
    sample_tp_per_subtype=3,
):
    """Selección determinista de errores, discrepancias clasificador/detector y aciertos TB representativos."""
    pred_maps = {arch: {r['relative_path']: r for r in predictions_by_arch[arch]} for arch in pointer_archs}
    selected = {}

    def register(row, category, reason):
        path = row['relative_path']
        gt_raw = json.loads(row['boxes_json'])
        subtypes = json.loads(row['tb_subtypes_json'])
        det_all = sorted(det_predictions[path], key=lambda b: -b['confidence'])
        det_kept = [b for b in det_all if b['confidence'] >= det_threshold]
        entry = selected.get(path)
        if entry is None:
            entry = {
                'relative_path': path,
                'class_name': row['class_name'],
                'label': int(row['label']),
                'group_id': row['group_id'],
                'tb_subtypes': ';'.join(subtypes) if subtypes else 'none',
                'gt_box_count': len(gt_raw),
                'gt_boxes_json': row['boxes_json'],
                'det_box_count_at_threshold': len(det_kept),
                'det_max_conf': float(det_all[0]['confidence']) if det_all else 0.0,
                'det_boxes_json': json.dumps(det_kept),
                'categories': [],
                'reasons': [],
            }
            for arch in pointer_archs:
                p_row = pred_maps[arch][path]
                entry[f'{arch}_p_TB'] = float(p_row['p_TB'])
                entry[f'{arch}_pred'] = CLASSES[int(p_row['prediction'])]
            selected[path] = entry
        if category not in entry['categories']:
            entry['categories'].append(category)
        if reason not in entry['reasons']:
            entry['reasons'].append(reason)

    # 1. Todos los FN y FP TB en cualquiera de los 3 punteros + discrepancias clave.
    non_tb_det_only = []
    tp_candidates_by_subtype = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['relative_path']):
        path = row['relative_path']
        cls_name = row['class_name']
        gt_raw = json.loads(row['boxes_json'])
        det_all = det_predictions[path]
        det_kept = [b for b in det_all if b['confidence'] >= det_threshold]
        max_conf = max((b['confidence'] for b in det_kept), default=0.0)

        fn_archs = [a for a in pointer_archs
                    if cls_name == 'TB' and (int(pred_maps[a][path]['prediction']) != 2
                                             or float(pred_maps[a][path]['p_TB']) < cls_threshold)]
        fp_archs = [a for a in pointer_archs
                    if cls_name != 'TB' and (int(pred_maps[a][path]['prediction']) == 2
                                             or float(pred_maps[a][path]['p_TB']) >= cls_threshold)]
        if fn_archs:
            register(row, 'tb_fn', 'FN_TB:' + ','.join(fn_archs))
        if fp_archs:
            register(row, 'tb_fp', 'FP_TB:' + ','.join(fp_archs))

        if cls_name == 'TB':
            any_cls_alert = any(float(pred_maps[a][path]['p_TB']) >= cls_threshold for a in pointer_archs)
            all_cls_missed = all(float(pred_maps[a][path]['p_TB']) < cls_threshold for a in pointer_archs)
            localized = _has_localized_match(gt_raw, det_all, det_threshold, match_iou=0.5)
            if any_cls_alert and not det_kept:
                register(row, 'discrepancy_cls_only_tb', 'Clasificador alerta TB pero YOLO11n sin cajas')
            elif all_cls_missed and det_kept:
                register(row, 'discrepancy_det_only_tb', 'YOLO11n detecta caja cuando los 3 punteros omiten TB')
            elif not fn_archs and localized:
                subtypes = json.loads(row['tb_subtypes_json'])
                key = '+'.join(sorted(set(subtypes))) if subtypes else 'unspecified'
                min_area = min((b['bbox'][2] * b['bbox'][3] for b in gt_raw), default=0.0)
                tp_candidates_by_subtype[key].append((min_area, row))
        else:
            no_cls_alert = all(float(pred_maps[a][path]['p_TB']) < cls_threshold for a in pointer_archs)
            if no_cls_alert and det_kept:
                non_tb_det_only.append((max_conf, row))

    for _, row in sorted(non_tb_det_only, key=lambda item: (-item[0], item[1]['relative_path']))[:max_det_only_non_tb]:
        register(row, 'discrepancy_det_only_non_tb', f'No-TB con caja YOLO11n (max_conf>={det_threshold:.2f})')

    for key, items in sorted(tp_candidates_by_subtype.items()):
        items_sorted = sorted(items, key=lambda item: (item[0], item[1]['relative_path']))
        picks = [items_sorted[0]]
        if len(items_sorted) > 2:
            picks.append(items_sorted[len(items_sorted) // 2])
        if len(items_sorted) > 1:
            picks.append(items_sorted[-1])
        for _, row in picks[:sample_tp_per_subtype]:
            register(row, 'tb_tp_representative', f'Acierto TB localizado ({key})')

    ordered_out = []
    for entry in sorted(selected.values(), key=lambda e: (e['categories'][0], e['relative_path'])):
        out = dict(entry)
        out['categories'] = ';'.join(entry['categories'])
        out['reasons'] = ' | '.join(entry['reasons'])
        ordered_out.append(out)
    return ordered_out


def render_visual_review_panels(dataset_root, cases, output_dir, pointer_archs=POINTER_ARCHS, per_page=6):
    """Renderiza paneles visuales con cajas GT (verde) y cajas YOLO11n (rojo/naranja)."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image
    from tbxdata.dataset import letterbox

    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for page_idx in range(0, len(cases), per_page):
        chunk = cases[page_idx:page_idx + per_page]
        cols = 2
        rows_n = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(12.5, 5.2 * rows_n))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, case in zip(axes_flat, chunk):
            with Image.open(safe_path(dataset_root, case['relative_path'])) as im:
                rgb = letterbox(im, 512)
            ax.imshow(rgb)
            for gt in json.loads(case['gt_boxes_json']):
                x, y, w, h = gt['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=2.0, edgecolor='#1B9E77', facecolor='none')
                ax.add_patch(rect)
            for det in json.loads(case['det_boxes_json']):
                x1, y1, x2, y2 = det['xyxy']
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, linestyle='--',
                                         edgecolor='#D95F02', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, max(12, y1 - 3), f"{det['confidence']:.2f}", color='#D95F02',
                        fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.65, lw=0))
            short_scores = ' | '.join(f"{a[:7]}:{case[f'{a}_p_TB']:.2f}" for a in pointer_archs)
            ax.set_title(
                f"[{case['categories']}] {case['class_name']} · {case['relative_path']}\n"
                f"Subtipo: {case['tb_subtypes']} | {short_scores}",
                fontsize=8.3,
            )
            ax.axis('off')
        for ax in axes_flat[len(chunk):]:
            ax.axis('off')
        fig.suptitle('Revisión visual · Verde: GT TB | Naranja discontinuo: YOLO11n', fontsize=11.5)
        fig.tight_layout()
        stem = f'visual_review_page_{page_idx // per_page + 1:02d}'
        for ext in ('png', 'pdf'):
            fig.savefig(output_dir / f'{stem}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        generated.append(f'{stem}.png')
    return generated


def count_parameters(model, checkpoint_path=None):
    """Cuenta parámetros totales y entrenables del protocolo sin mutar el modelo."""
    total = sum(p.numel() for p in model.parameters())
    protocol_trainable = None
    if hasattr(model, 'head_modules') and hasattr(model, 'last_modules'):
        unique_ids = set()
        trainable_sum = 0
        for mod in list(model.head_modules()) + list(model.last_modules()):
            for p in mod.parameters():
                if id(p) not in unique_ids:
                    unique_ids.add(id(p))
                    trainable_sum += p.numel()
        protocol_trainable = trainable_sum
    size_mb = (Path(checkpoint_path).stat().st_size / (1024 ** 2)) if checkpoint_path else None
    return {
        'total_params': int(total),
        'protocol_trainable_params': int(protocol_trainable) if protocol_trainable is not None else None,
        'checkpoint_mb': float(size_mb) if size_mb is not None else None,
    }


def benchmark_classifier_cost(model, device='cuda', batch_sizes=(1, 8), warmup=20, iterations=100, image_size=512):
    """Mide latencia sincronizada, throughput y VRAM pico de un clasificador en FP32."""
    import torch

    use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
    model = model.to(device).eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randn(int(bs), 3, int(image_size), int(image_size), device=device, dtype=torch.float32)
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
            times_ms = []
            for _ in range(iterations):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if use_cuda:
                    torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times_ms, dtype=float)
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if use_cuda else None
        mean_ms = float(arr.mean())
        results[f'batch_{bs}'] = {
            'batch_size': int(bs),
            'mean_batch_ms': mean_ms,
            'mean_per_image_ms': mean_ms / bs,
            'p50_batch_ms': float(np.percentile(arr, 50)),
            'p95_batch_ms': float(np.percentile(arr, 95)),
            'throughput_img_s': float(1000.0 * bs / mean_ms) if mean_ms > 0 else None,
            'peak_vram_mb': float(peak_vram_mb) if peak_vram_mb is not None else None,
        }
        del x
    return results


def benchmark_detector_cost(
    weights_path,
    sample_image_paths,
    inference_options,
    batch_sizes=(1, 8),
    warmup=10,
    iterations=40,
):
    """Mide parámetros, VRAM pico y latencia de YOLO11n bajo las mismas opciones del contrato."""
    import torch
    from ultralytics import YOLO

    detector = YOLO(str(weights_path))
    total_params = int(sum(p.numel() for p in detector.model.parameters()))
    size_mb = float(Path(weights_path).stat().st_size / (1024 ** 2))
    use_cuda = torch.cuda.is_available()
    opts = dict(inference_options)
    opts.update(save=False, verbose=False)
    results = {}
    for bs in batch_sizes:
        imgs = [str(sample_image_paths[i % len(sample_image_paths)]) for i in range(int(bs))]
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            _ = detector.predict(source=imgs, **opts)
        if use_cuda:
            torch.cuda.synchronize()
        times_ms = []
        for _ in range(iterations):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = detector.predict(source=imgs, **opts)
            if use_cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times_ms, dtype=float)
        mean_ms = float(arr.mean())
        results[f'batch_{bs}'] = {
            'batch_size': int(bs),
            'mean_batch_ms': mean_ms,
            'mean_per_image_ms': mean_ms / bs,
            'p50_batch_ms': float(np.percentile(arr, 50)),
            'p95_batch_ms': float(np.percentile(arr, 95)),
            'throughput_img_s': float(1000.0 * bs / mean_ms) if mean_ms > 0 else None,
            'peak_vram_mb': float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if use_cuda else None,
        }
    return {
        'total_params': total_params,
        'checkpoint_mb': size_mb,
        'batches': results,
    }


def _group_strata(rows):
    """Agrupa índices de filas por group_id y asigna estrato por prioridad TB > sick > health."""
    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        groups[r['group_id']].append(idx)
    strata = {0: [], 1: [], 2: []}
    group_list = []
    for gid, idxs in sorted(groups.items()):
        labels = [int(rows[i]['label']) for i in idxs]
        stratum = max(labels)
        g_idx = len(group_list)
        group_list.append(np.asarray(idxs, dtype=np.int64))
        strata[stratum].append(g_idx)
    return group_list, {k: np.asarray(v, dtype=np.int64) for k, v in strata.items() if v}


def _summarize_distribution(values, point_estimate, ci_level=0.95):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    alpha = (1.0 - ci_level) / 2.0
    if not len(arr):
        return {'point': point_estimate, 'ci_low': None, 'ci_high': None, 'std': None, 'valid_resamples': 0}
    return {
        'point': float(point_estimate) if point_estimate is not None else None,
        'ci_low': float(np.quantile(arr, alpha)),
        'ci_high': float(np.quantile(arr, 1.0 - alpha)),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'valid_resamples': int(len(arr)),
    }


def _fast_binary_auc_ap(y_true, scores):
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if not positives or not negatives:
        return None, None
    curve = binary_curve(y_true, scores)
    return curve['roc_auc'], curve['average_precision']


def _fast_multiclass_summary(y, p):
    pred = p.argmax(1)
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    supports, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    recall = np.divide(tp, supports, out=np.zeros(3, float), where=supports > 0)
    precision = np.divide(tp, predicted, out=np.zeros(3, float), where=predicted > 0)
    f1 = np.divide(2 * tp, supports + predicted, out=np.zeros(3, float), where=(supports + predicted) > 0)
    tb_neg = len(y) - supports[2]
    tb_tn = len(y) - supports[2] - predicted[2] + tp[2]
    tb_auc, tb_ap = _fast_binary_auc_ap(y == 2, p[:, 2])
    return {
        'accuracy': float(tp.sum() / len(y)),
        'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall[supports > 0].mean()),
        'tb_recall': float(recall[2]),
        'tb_specificity': float(tb_tn / tb_neg) if tb_neg else None,
        'tb_precision': float(precision[2]) if predicted[2] else None,
        'tb_roc_auc': tb_auc,
        'tb_average_precision': tb_ap,
    }


def group_bootstrap_metrics(
    rows,
    predictions_by_arch,
    thresholds=(0.1, 0.3, 0.5),
    pointer_archs=POINTER_ARCHS,
    n_resamples=2000,
    seed=20261019,
    ci_level=0.95,
):
    """Bootstrap estratificado por grupo de duplicados (group_id) para modelos y deltas pareados."""
    y_all = np.asarray([int(r['label']) for r in rows], dtype=np.int64)
    probs_by_arch = {}
    for arch, ordered in predictions_by_arch.items():
        by_path = {r['relative_path']: r for r in ordered}
        probs_by_arch[arch] = np.asarray(
            [[float(by_path[r['relative_path']]['p_' + c]) for c in CLASSES] for r in rows],
            dtype=float,
        )

    group_list, strata = _group_strata(rows)
    rng = np.random.default_rng(int(seed))
    boot_indices = []
    for _ in range(int(n_resamples)):
        sampled_groups = []
        for g_arr in strata.values():
            sampled_groups.append(rng.choice(g_arr, size=len(g_arr), replace=True))
        flat_groups = np.concatenate(sampled_groups)
        boot_indices.append(np.concatenate([group_list[g] for g in flat_groups]))

    model_results = {}
    boot_series = {arch: defaultdict(list) for arch in probs_by_arch}
    for arch, p_all in probs_by_arch.items():
        point = _fast_multiclass_summary(y_all, p_all)
        for idx in boot_indices:
            m = _fast_multiclass_summary(y_all[idx], p_all[idx])
            for k, v in m.items():
                boot_series[arch][k].append(v)
        metrics_ci = {k: _summarize_distribution(boot_series[arch][k], point[k], ci_level) for k in point}

        thresh_ci = {}
        if arch in pointer_archs:
            y_tb = (y_all == 2)
            score_tb = p_all[:, 2]
            for t in thresholds:
                t_key = f'{t:.2f}'
                pt_row = tb_threshold_table(y_all, p_all, [t])[0]
                sens_list, spec_list, prec_list = [], [], []
                for idx in boot_indices:
                    yb = y_tb[idx]
                    alert = score_tb[idx] >= t
                    tp = int((alert & yb).sum())
                    fp = int((alert & ~yb).sum())
                    fn = int((~alert & yb).sum())
                    tn = int((~alert & ~yb).sum())
                    sens_list.append(tp / (tp + fn) if (tp + fn) else None)
                    spec_list.append(tn / (tn + fp) if (tn + fp) else None)
                    prec_list.append(tp / (tp + fp) if (tp + fp) else None)
                thresh_ci[t_key] = {
                    'threshold': float(t),
                    'sensitivity': _summarize_distribution(sens_list, pt_row['sensitivity'], ci_level),
                    'specificity': _summarize_distribution(spec_list, pt_row['specificity'], ci_level),
                    'precision': _summarize_distribution(prec_list, pt_row['precision'], ci_level),
                }
        model_results[arch] = {'metrics': metrics_ci, 'thresholds': thresh_ci}

    paired_deltas = {}
    available_pointers = [a for a in pointer_archs if a in probs_by_arch]
    for i in range(len(available_pointers)):
        for j in range(i + 1, len(available_pointers)):
            a, b = available_pointers[i], available_pointers[j]
            pt_a = _fast_multiclass_summary(y_all, probs_by_arch[a])
            pt_b = _fast_multiclass_summary(y_all, probs_by_arch[b])
            pair_out = {}
            for metric in ('accuracy', 'macro_f1', 'tb_recall', 'tb_roc_auc'):
                diffs = [va - vb for va, vb in zip(boot_series[a][metric], boot_series[b][metric])
                         if va is not None and vb is not None]
                pt_diff = (pt_a[metric] - pt_b[metric]) if (pt_a[metric] is not None and pt_b[metric] is not None) else None
                summary = _summarize_distribution(diffs, pt_diff, ci_level)
                arr = np.asarray(diffs, dtype=float)
                summary['prob_a_gt_b'] = float((arr > 0).mean()) if len(arr) else None
                pair_out[metric] = summary
            paired_deltas[f'{a}__minus__{b}'] = pair_out

    return {
        'n_images': len(rows),
        'n_groups': len(group_list),
        'n_resamples': int(n_resamples),
        'seed': int(seed),
        'ci_level': float(ci_level),
        'note': 'Bootstrap estratificado por group_id sobre una sola seed y validation reutilizada; no prueba superioridad estadística definitiva.',
        'models': model_results,
        'paired_deltas': paired_deltas,
    }


def detector_group_bootstrap(
    rows,
    predictions,
    thresholds=(0.1, 0.25, 0.5),
    match_iou=0.5,
    n_resamples=2000,
    seed=20261019,
    ci_level=0.95,
):
    """Bootstrap estratificado por group_id para métricas de YOLO11n en validation."""
    detector_threshold_table(rows, predictions, thresholds, match_iou=match_iou)
    group_list, strata = _group_strata(rows)
    rng = np.random.default_rng(int(seed))
    boot_indices = []
    for _ in range(int(n_resamples)):
        sampled = [rng.choice(g_arr, size=len(g_arr), replace=True) for g_arr in strata.values()]
        boot_indices.append(np.concatenate([group_list[g] for g in np.concatenate(sampled)]))

    point_table = {r['threshold']: r for r in detector_detailed_metrics(rows, predictions, thresholds, match_iou=match_iou)}
    out = {}
    for t in thresholds:
        n = len(rows)
        tp_arr = np.zeros(n, dtype=np.int64)
        fp_arr = np.zeros(n, dtype=np.int64)
        gt_arr = np.zeros(n, dtype=np.int64)
        pos_arr = np.zeros(n, dtype=np.int64)
        neg_arr = np.zeros(n, dtype=np.int64)
        any_arr = np.zeros(n, dtype=np.int64)
        loc_arr = np.zeros(n, dtype=np.int64)
        neg_alert_arr = np.zeros(n, dtype=np.int64)
        for idx, row in enumerate(rows):
            gt = [[b['bbox'][0], b['bbox'][1], b['bbox'][0] + b['bbox'][2], b['bbox'][1] + b['bbox'][3]]
                  for b in json.loads(row['boxes_json'])]
            boxes = sorted((b for b in predictions[row['relative_path']] if b['confidence'] >= t),
                           key=lambda b: -b['confidence'])
            gt_arr[idx] = len(gt)
            pos_arr[idx] = int(bool(gt))
            neg_arr[idx] = int(not bool(gt))
            any_arr[idx] = int(bool(gt) and bool(boxes))
            neg_alert_arr[idx] = int(not bool(gt) and bool(boxes))
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

        series = defaultdict(list)
        for b_idx in boot_indices:
            tp_s = int(tp_arr[b_idx].sum())
            fp_s = int(fp_arr[b_idx].sum())
            gt_s = int(gt_arr[b_idx].sum())
            pos_s = int(pos_arr[b_idx].sum())
            neg_s = int(neg_arr[b_idx].sum())
            any_s = int(any_arr[b_idx].sum())
            loc_s = int(loc_arr[b_idx].sum())
            neg_alert_s = int(neg_alert_arr[b_idx].sum())
            series['lesion_recall'].append(tp_s / gt_s if gt_s else None)
            series['box_precision'].append(tp_s / (tp_s + fp_s) if (tp_s + fp_s) else None)
            series['image_sensitivity_any_box'].append(any_s / pos_s if pos_s else None)
            series['image_sensitivity_localized'].append(loc_s / pos_s if pos_s else None)
            series['fp_boxes_per_image'].append(fp_s / len(b_idx))
            series['negative_image_alert_rate'].append(neg_alert_s / neg_s if neg_s else None)

        pt = point_table[float(t)]
        out[f'{t:.2f}'] = {
            metric: _summarize_distribution(series[metric], pt[metric], ci_level)
            for metric in ('lesion_recall', 'box_precision', 'image_sensitivity_any_box',
                           'image_sensitivity_localized', 'fp_boxes_per_image', 'negative_image_alert_rate')
        }
    return {
        'n_images': len(rows),
        'n_groups': len(group_list),
        'n_resamples': int(n_resamples),
        'seed': int(seed),
        'ci_level': float(ci_level),
        'thresholds': out,
    }

"""Extensión controlada de Fase 3 para entrenar y comparar ViT-B/16.

Mantiene intactos architectures.py, suite.py, phase5.py y los hashes de procedencia
de los tres clasificadores previos. Nunca abre test ni datos externos.
"""
import gc
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tbxdata.common import read_csv, sha256, write_csv
from tbxdata.dataset import TBXDataset, class_weights, load_config
from tbxdata.environment import in_colab
from tbxdata.validate import validate_manifests
from .architectures import Classifier
from .metrics import binary_curve
from .phase5 import CLASSES, read_validation_predictions, tb_threshold_table, validation_rows
from .protocol import atomic_json, provenance
from .suite import epoch_train
from .train import _seed, atomic_checkpoint, evaluate, make_loader, restore_rng, rng_state

ROOT = Path(__file__).resolve().parents[2]
SHARED_PROTOCOL_KEYS = (
    'seed', 'epochs', 'warmup_epochs', 'batch_size', 'accumulation_steps',
    'num_workers', 'head_lr', 'backbone_lr', 'weight_decay', 'gradient_clip',
    'patience', 'min_delta', 'precision', 'selection_metric',
)
PRIOR_MODELS = (
    ('convnext_tiny', 'ConvNeXt-Tiny', 'ConvNeXt', 'convnext'),
    ('vmamba_tiny', 'VMamba-Tiny', 'VMamba', 'vmamba'),
    ('hybrid_cross_attention', 'Híbrido de atención cruzada', 'Híbrido', 'hybrid'),
)
VIT_META = ('vit_b_16', 'ViT-B/16', 'ViT-B/16', 'vit_b_16')
ALL_COLORS = ['#2864AD', '#C16B19', '#287C60', '#7B4B94']


class ViTClassifier(Classifier):
    """ViT-B/16 oficial a 512px con interpolación bicúbica de embeddings posicionales."""

    def __init__(self, pretrained=True, image_size=512):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        self.image_size = int(image_size)
        self.encoder = vit_b_16(weights=None, image_size=self.image_size)
        self.encoder.heads.head = nn.Linear(self.encoder.heads.head.in_features, 3)
        if pretrained:
            state = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            self.load_interpolated_state(state, image_size=self.image_size)

    def load_interpolated_state(self, state, image_size=None):
        """Carga pesos preentrenados interpolando pos_embedding 224->image_size."""
        from torchvision.models.vision_transformer import interpolate_embeddings
        target_size = self.image_size if image_size is None else int(image_size)
        adapted = interpolate_embeddings(target_size, 16, dict(state), reset_heads=True)
        missing, unexpected = self.encoder.load_state_dict(adapted, strict=False)
        if set(missing) != {'heads.head.weight', 'heads.head.bias'} or unexpected:
            raise ValueError(f'Carga incompleta de ViT-B/16: missing={missing}, unexpected={unexpected}')

    def forward(self, x):
        return self.encoder(x)

    def head_modules(self):
        return [self.encoder.encoder.ln, self.encoder.heads]

    def backbones(self):
        return [self.encoder.conv_proj, self.encoder.encoder.dropout, self.encoder.encoder.layers]

    def last_modules(self):
        return list(self.encoder.encoder.layers[-2:])


def validate_vit_config(config, project=ROOT):
    """Valida el protocolo ViT y verifica paridad exacta con phase3_suite.json."""
    project = Path(project)
    template = json.loads((project / 'configs/vit_controlled.json').read_text(encoding='utf-8'))
    suite_cfg = json.loads((project / 'configs/phase3_suite.json').read_text(encoding='utf-8'))
    if set(config) != set(template):
        raise ValueError(f'Campos inválidos en config ViT: {set(config) ^ set(template)}')
    if config['architecture'] != 'vit_b_16':
        raise ValueError('Arquitectura ViT esperada: vit_b_16.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', config['protocol_id']):
        raise ValueError('protocol_id inválido.')
    for key in SHARED_PROTOCOL_KEYS:
        if config[key] != suite_cfg[key]:
            raise ValueError(f'Hiperparámetro {key} difiere del protocolo controlado: {config[key]} != {suite_cfg[key]}')
    return dict(config)


def run_vit_model(project, dataset, drive_base, config, resume_run=None, smoke=False):
    """Entrenamiento controlado y reanudable de ViT-B/16 exclusivamente en Colab GPU."""
    if not in_colab():
        raise RuntimeError('La ejecución real requiere Google Colab con GPU.')
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('Colab no tiene GPU CUDA disponible.')
    project, dataset, drive_base = map(Path, (project, dataset, drive_base))
    config = validate_vit_config(config, project)
    validate_manifests(project)
    data_cfg = load_config(project / 'configs/data.json')
    if data_cfg['classes'] != {'health': 0, 'sick': 1, 'TB': 2} or data_cfg['image_size'] != 512:
        raise ValueError('El protocolo exige tres clases y 512px congelados.')
    classes = list(CLASSES)
    fingerprints = provenance(project)
    _seed(torch, config['seed'])

    resumed = None
    if resume_run:
        if smoke or any(c in str(resume_run) for c in '/\\:') or str(resume_run) in {'.', '..'}:
            raise ValueError('resume_run no válido.')
        run_id = str(resume_run)
        resumed = torch.load(drive_base / 'checkpoints' / run_id / 'last.pt', map_location='cpu', weights_only=False)
        if (resumed['architecture'] != 'vit_b_16' or resumed['config'] != config
                or resumed['provenance'] != fingerprints):
            raise ValueError('Reanudar exige arquitectura, protocolo, código y datos idénticos.')
    else:
        prefix = 'smoke_vit_' if smoke else 'phase3_vit_b_16_'
        run_id = prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]

    out = drive_base / 'experiments' / run_id
    checkpoints = drive_base / 'checkpoints' / run_id
    out.mkdir(parents=True, exist_ok=bool(resumed))
    checkpoints.mkdir(parents=True, exist_ok=bool(resumed))

    history = resumed['history'] if resumed else []
    updates = resumed['updates'] if resumed else 0
    best_state = resumed['best_model_state'] if resumed else None
    best_score = resumed['best_score'] if resumed else -1.0
    best_epoch = resumed['best_epoch'] if resumed else None
    stale = resumed['stale'] if resumed else 0
    start_epoch = resumed['next_epoch'] if resumed else 0
    model = optimizer = None

    def status(state, **extra):
        record = {
            'run_id': run_id, 'architecture': 'vit_b_16', 'status': state,
            'completed_epochs': len(history), 'optimizer_updates': updates,
            'kind': 'smoke' if smoke else 'controlled', **extra,
        }
        atomic_json(out / 'status.json', record)
        if not smoke:
            atomic_json(drive_base / 'results/vit_suite' / config['protocol_id'] / 'vit_b_16.json', record)

    status('initializing')
    print('Corrida:', run_id, '\nSalidas:', out, flush=True)
    try:
        model = ViTClassifier(pretrained=resumed is None, image_size=data_cfg['image_size']).to('cuda')
        optimizer = torch.optim.AdamW(
            model.parameter_groups(config['head_lr'], config['backbone_lr']),
            weight_decay=config['weight_decay'],
        )
        if resumed:
            model.load_state_dict(resumed['model_state'], strict=True)
            optimizer.load_state_dict(resumed['optimizer_state'])
            restore_rng(torch, resumed['rng'])
        resumed = None
        gc.collect()

        train = TBXDataset(dataset, project / 'manifests/train.csv', data_cfg, training=True)
        val = TBXDataset(dataset, project / 'manifests/validation.csv', data_cfg, training=False)
        local_cfg = dict(config)
        if smoke:
            local_cfg.update(epochs=1, warmup_epochs=0, batch_size=1, accumulation_steps=2, num_workers=0)
            for ds in [train, val]:
                ds.rows = [next(r for r in ds.rows if r['class_name'] == c) for c in classes]

        weights = torch.tensor(class_weights(project / 'manifests/train.csv', data_cfg), device='cuda')
        val_loader = make_loader(torch, val, local_cfg, False)
        atomic_json(out / 'config.json', config)
        atomic_json(out / 'provenance.json', fingerprints)
        atomic_json(out / 'environment.json', {
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'precision': 'fp32', 'architecture': 'vit_b_16',
        })
        atomic_json(out / 'class_weights.json', dict(zip(classes, weights.cpu().tolist())))
        before = [p.detach().cpu().clone() for m in model.head_modules() for p in m.parameters()] if smoke else None

        status('running')
        stop_reason = 'epoch_budget'
        if stale >= config['patience']:
            stop_reason = 'early_stopping'
        else:
            for epoch in range(start_epoch, local_cfg['epochs']):
                tick = time.monotonic()
                result = epoch_train(torch, model, make_loader(torch, train, local_cfg, True, epoch), optimizer, weights, local_cfg, epoch)
                updates += result['optimizer_updates']
                metrics, y, probs = evaluate(torch, model, val_loader, weights, False, classes)
                improved = metrics['macro_f1'] > best_score + config['min_delta']
                if improved:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_score, best_epoch, stale = metrics['macro_f1'], epoch + 1, 0
                else:
                    stale += 1
                history.append({
                    'epoch': epoch + 1, **result,
                    'validation_loss': metrics['loss'],
                    'validation_accuracy': metrics['accuracy'],
                    'validation_macro_f1': metrics['macro_f1'],
                    'validation_tb_recall': metrics['per_class']['TB']['recall'],
                    'seconds': time.monotonic() - tick,
                })
                payload = {
                    'schema': 'vit_controlled_v1', 'architecture': 'vit_b_16', 'run_id': run_id,
                    'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'rng': rng_state(torch), 'config': config, 'provenance': fingerprints,
                    'history': history, 'updates': updates, 'best_model_state': best_state,
                    'best_score': best_score, 'best_epoch': best_epoch, 'stale': stale,
                    'next_epoch': epoch + 1,
                }
                atomic_checkpoint(torch, checkpoints / 'last.pt', payload)
                write_csv(out / 'history.csv', history)
                status('running', best_epoch=best_epoch)
                print(
                    f'Época {epoch+1}: macro-F1={metrics["macro_f1"]:.4f}, '
                    f'TB recall={metrics["per_class"]["TB"]["recall"]:.4f}; Drive guardado.',
                    flush=True,
                )
                if stale >= config['patience']:
                    stop_reason = 'early_stopping'
                    break

        if smoke:
            after = [p.detach().cpu() for m in model.head_modules() for p in m.parameters()]
            if not updates or not any(not torch.equal(a, b) for a, b in zip(before, after)):
                raise RuntimeError('El smoke no actualizó la cabeza; no se declara aprobado.')

        model.load_state_dict(best_state)
        metrics, y, probs = evaluate(torch, model, val_loader, weights, False, classes)
        atomic_checkpoint(torch, checkpoints / 'best.pt', {
            'architecture': 'vit_b_16', 'run_id': run_id, 'epoch': best_epoch,
            'model_state': best_state, 'config': config, 'provenance': fingerprints,
            'classes': classes, 'data_config': data_cfg, 'validation': metrics,
        })
        if smoke:
            reloaded = torch.load(checkpoints / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(reloaded['model_state'])
            del reloaded
            _, yy, pp = evaluate(torch, model, val_loader, weights, False, classes)
            if not np.array_equal(yy, y) or not np.allclose(pp, probs, rtol=1e-4, atol=1e-4):
                raise RuntimeError('El checkpoint smoke no reproduce predicciones tras recarga.')

        write_csv(out / 'validation_predictions.csv', [
            dict(
                relative_path=r['relative_path'], group_id=r['group_id'],
                label=int(label), prediction=int(p.argmax()),
                **{f'p_{c}': float(p[i]) for i, c in enumerate(classes)},
            )
            for r, label, p in zip(val.rows, y, probs)
        ])
        atomic_json(out / 'best_validation.json', metrics)
        if not smoke:
            from .review import plot_run
            plot_run(out)

        summary = {
            'schema': 'vit_controlled_v1', 'run_id': run_id, 'architecture': 'vit_b_16',
            'status': 'completed_smoke' if smoke else 'completed_controlled',
            'training_executed': updates > 0, 'protocol': config,
            'completed_epochs': len(history), 'optimizer_updates': updates,
            'best_epoch': best_epoch, 'stop_reason': stop_reason, 'validation': metrics,
            'provenance': fingerprints, 'checkpoint_sha256': sha256(checkpoints / 'best.pt'),
            'test_evaluated': False, 'external_evaluated': False, 'phase3_complete': False,
        }
        atomic_json(out / 'summary.json', summary)
        status(summary['status'])
        return summary
    except BaseException as exc:
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def _pr_curve(y, score):
    order = np.argsort(-score, kind='stable')
    target, sorted_score = np.asarray(y)[order], np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(target) - 1]
    tp = np.cumsum(target)[ends]
    return np.r_[0.0, tp / target.sum()], np.r_[1.0, tp / (ends + 1)]


def _load_prior_run(project, drive_base, arch, short_name, prior_runs=None):
    """Carga evidencia de un clasificador previo desde Drive o desde drive_evidence local."""
    folder = None
    if prior_runs and arch in prior_runs:
        candidate = Path(drive_base) / 'experiments' / prior_runs[arch]
        if (candidate / 'summary.json').is_file():
            folder = candidate
    if folder is not None:
        return (
            folder / 'summary.json',
            folder / 'history.csv',
            folder / 'validation_predictions.csv',
        )
    evidence = Path(project) / 'reports/phase5_readiness/drive_evidence'
    return (
        evidence / f'{short_name}_summary.json',
        evidence / f'{short_name}_history.csv',
        evidence / f'{short_name}_validation_predictions.csv',
    )


def export_vit_report_outputs(project, drive_base, vit_run_id, output_dir, prior_runs=None):
    """Genera figuras, tablas y JSON completo con el formato de Fase 5 para los 4 clasificadores."""
    import matplotlib.pyplot as plt
    project, drive_base, output_dir = map(Path, (project, drive_base, output_dir))
    rows = validation_rows(project)
    output_dir.mkdir(parents=True, exist_ok=True)

    vit_folder = drive_base / 'experiments' / vit_run_id
    run_entries = []
    for arch, display_name, short_label, short_file in PRIOR_MODELS:
        s_path, h_path, p_path = _load_prior_run(project, drive_base, arch, short_file, prior_runs)
        run_entries.append((arch, display_name, short_label, s_path, h_path, p_path))
    run_entries.append((
        VIT_META[0], VIT_META[1], VIT_META[2],
        vit_folder / 'summary.json',
        vit_folder / 'history.csv',
        vit_folder / 'validation_predictions.csv',
    ))

    sources = {str(project / 'manifests/validation.csv'): sha256(project / 'manifests/validation.csv')}
    records, data = [], []

    def track(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return path

    def save(fig, name, title, table, note, architectures):
        fig.tight_layout()
        try:
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(output_dir / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
        if table is not None:
            write_csv(output_dir / f'{name}.csv', table)
        records.append(dict(
            name=name, title=title, split='validation', architectures=architectures,
            note=note, files=[f'{name}.{ext}' for ext in (['png', 'pdf', 'svg', 'csv'] if table is not None else ['png', 'pdf', 'svg'])],
        ))

    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': False, 'svg.fonttype': 'none'}):
        for arch, display_name, short_label, s_path, h_path, p_path in run_entries:
            summary = json.loads(track(s_path).read_text(encoding='utf-8'))
            if summary['architecture'] != arch or summary.get('test_evaluated'):
                raise ValueError(f'Corrida inválida o fuera de protocolo: {arch}')
            history = read_csv(track(h_path))
            _, y, p, metrics = read_validation_predictions(track(p_path), rows)
            if metrics['confusion_matrix'] != summary['validation']['confusion_matrix']:
                raise ValueError(f'Matriz histórica distinta de las predicciones para {arch}.')
            data.append((arch, display_name, short_label, summary, history, y, p, metrics))

            if arch == 'vit_b_16':
                epochs = [int(r['epoch']) for r in history]
                if len(epochs) != summary['completed_epochs'] or summary['best_epoch'] not in epochs:
                    raise ValueError('Historial incompleto para ViT-B/16.')

                fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
                groups = [
                    [('train_loss', 'Train'), ('validation_loss', 'Validation')],
                    [('train_accuracy', 'Train'), ('validation_accuracy', 'Validation')],
                    [('validation_macro_f1', 'Macro-F1'), ('validation_tb_recall', 'Sensibilidad TB')],
                ]
                for ax, columns, title in zip(axes, groups, ['Pérdida CE ponderada', 'Accuracy', 'Métricas de validation']):
                    for color, (key, label) in zip(ALL_COLORS[:3], columns):
                        ax.plot(epochs, [float(r[key]) for r in history], marker='o', markersize=3, color=color, label=label)
                    ax.axvline(summary['best_epoch'], color='#656565', ls='--', lw=1, label='Mejor época')
                    ax.set(title=title, xlabel='Época')
                    ax.grid(alpha=0.18)
                    ax.legend(fontsize=8)
                    ax.set_xticks(epochs)
                    if ax is not axes[0]:
                        ax.set_ylim(0, 1.02)
                fig.suptitle(f'{display_name} · historial de entrenamiento', fontsize=14)
                save(
                    fig, 'vit_b_16_training', f'Historial de {display_name}', history,
                    'Epochs reales; línea en mejor época. Train se mide durante updates con aumentos; '
                    'validation sin aumentos. No se rellenan épocas tras early stopping.', [arch],
                )

                cm = np.asarray(metrics['confusion_matrix'])
                normalized = cm / cm.sum(1, keepdims=True)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
                for ax, matrix, title, maximum in zip(axes, [cm, normalized], ['Conteos', 'Normalizada por clase real'], [cm.max(), 1.0]):
                    picture = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum)
                    for (i, j), v in np.ndenumerate(matrix):
                        ax.text(j, i, str(int(v)) if matrix is cm else f'{v:.1%}', ha='center', va='center',
                                color='white' if v > maximum * 0.55 else '#152536')
                    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                           xlabel='Clase predicha (argmax)', ylabel='Clase real', title=title)
                    fig.colorbar(picture, ax=ax, fraction=0.045, pad=0.04)
                fig.suptitle(f'{display_name} · validation · mejor época {summary["best_epoch"]}', fontsize=13)
                cm_table = [
                    dict(actual=CLASSES[i], predicted=CLASSES[j], count=int(cm[i, j]),
                         proportion_within_actual=float(normalized[i, j]))
                    for i in range(3) for j in range(3)
                ]
                save(
                    fig, 'vit_b_16_confusion', f'Matriz de confusión de {display_name}', cm_table,
                    'Filas reales, columnas predichas; argmax original; normalización por fila.', [arch],
                )

                fig, ax = plt.subplots(figsize=(6.1, 5))
                roc_table = []
                for i, (cls, color) in enumerate(zip(CLASSES, ALL_COLORS[:3])):
                    curve = binary_curve(y == i, p[:, i])
                    ax.plot(curve['fpr'], curve['tpr'], color=color, label=f"{cls} · AUC={curve['roc_auc']:.4f}")
                    roc_table += [dict(class_name=cls, fpr=x, tpr=z) for x, z in zip(curve['fpr'], curve['tpr'])]
                ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1)
                ax.set(xlabel='Tasa de falsos positivos', ylabel='Sensibilidad',
                       title=f'{display_name}\nROC OVR · validation', xlim=(0, 1), ylim=(0, 1.01))
                ax.legend(loc='lower right')
                ax.grid(alpha=0.18)
                save(fig, 'vit_b_16_roc', f'ROC multiclase de {display_name}', roc_table,
                     'One-vs-rest, scores originales; no corresponde a test.', [arch])

        comparison = []
        for arch, display_name, short_label, s, _, _, _, m in data:
            cm = m['confusion_matrix']
            comparison.append(dict(
                architecture=arch, display_name=display_name, short_label=short_label,
                run_id=s['run_id'], completed_epochs=s['completed_epochs'],
                best_epoch=s['best_epoch'], stop_reason=s['stop_reason'],
                accuracy=m['accuracy'], macro_f1=m['macro_f1'],
                TB_recall=m['per_class']['TB']['recall'],
                TB_precision=m['per_class']['TB']['precision'],
                TB_specificity=m['per_class']['TB']['specificity'],
                TB_roc_auc=m['per_class']['TB']['roc_auc'],
                TB_recognized=int(cm[2][2]),
                TB_to_health=int(cm[2][0]), TB_to_sick=int(cm[2][1]),
                sick_to_TB=int(cm[1][2]), health_to_TB=int(cm[0][2]),
                checkpoint_sha256=s['checkpoint_sha256'],
            ))
        write_csv(output_dir / 'validation_comparison_4models.csv', comparison)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
        for ax, columns, title in zip(axes, [['accuracy', 'macro_f1'], ['TB_recall', 'TB_precision', 'TB_specificity']],
                                      ['Clasificación multiclase', 'Clase TB']):
            labels = {'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1', 'TB_recall': 'Sensibilidad',
                      'TB_precision': 'Precisión', 'TB_specificity': 'Especificidad'}
            x = np.arange(len(comparison))
            width = 0.8 / len(columns)
            for j, key in enumerate(columns):
                values = [r[key] for r in comparison]
                bars = ax.bar(x + (j - (len(columns) - 1) / 2) * width, values, width, label=labels[key], color=ALL_COLORS[j])
                ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=3, fontsize=8)
            ax.set(xticks=x, xticklabels=[r['short_label'] for r in comparison], ylim=(0, 1.16), title=title, ylabel='Proporción')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=len(columns), fontsize=9)
            ax.yaxis.grid(alpha=0.18)
            ax.set_axisbelow(True)
        fig.suptitle('Comparación de los cuatro clasificadores · validation · mejores checkpoints', fontsize=14)
        save(
            fig, 'classifier_comparison', 'Comparación de los cuatro clasificadores', comparison,
            'Una seed y validation reutilizada para selección; sin intervalos ni prueba de superioridad estadística.',
            [r['architecture'] for r in comparison],
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        pr_table = []
        for color, (arch, display_name, _, _, _, y, p, _) in zip(ALL_COLORS, data):
            target = y == 2
            curve = binary_curve(target, p[:, 2])
            recall, precision = _pr_curve(target, p[:, 2])
            axes[0].plot(curve['fpr'], curve['tpr'], color=color, label=f"{display_name} · AUC={curve['roc_auc']:.4f}")
            axes[1].step(recall, precision, where='pre', color=color, label=f"{display_name} · AP={curve['average_precision']:.4f}")
            pr_table += [dict(architecture=arch, curve='ROC', x=x, y=z) for x, z in zip(curve['fpr'], curve['tpr'])]
            pr_table += [dict(architecture=arch, curve='PR', x=x, y=z) for x, z in zip(recall, precision)]
        axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
        axes[1].axhline(float((data[0][5] == 2).mean()), ls='--', color='gray', lw=1, label='Prevalencia de validation')
        for ax, title, xlabel, ylabel in zip(axes, ['ROC TB vs resto', 'Precisión–recall TB'],
                                             ['Tasa de falsos positivos', 'Sensibilidad'], ['Sensibilidad', 'Precisión']):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8, loc='lower left')
            ax.grid(alpha=0.18)
        fig.suptitle('Comparación de scores TB · validation', fontsize=14)
        save(
            fig, 'TB_roc_precision_recall', 'ROC y precisión–recall TB comparativas', pr_table,
            'AP es average precision, no área trapezoidal. Scores TB originales; ningún umbral queda seleccionado.',
            [r['architecture'] for r in comparison],
        )

        vit_entry = data[-1]
        vit_y, vit_p = vit_entry[5], vit_entry[6]
        vit_thresh_full = tb_threshold_table(vit_y, vit_p)
        write_csv(output_dir / 'vit_b_16_TB_thresholds_validation.csv', vit_thresh_full)
        vit_thresh_grid = tb_threshold_table(vit_y, vit_p, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

        best_entry = max(data, key=lambda d: d[7]['macro_f1'])
        best_thresh_grid = tb_threshold_table(best_entry[5], best_entry[6], [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

        fig, ax = plt.subplots(figsize=(7, 4))
        ts = [r['threshold'] for r in vit_thresh_full]
        ax.plot(ts, [r['sensitivity'] for r in vit_thresh_full], label='Sensibilidad TB')
        ax.plot(ts, [r['specificity'] for r in vit_thresh_full], label='Especificidad TB')
        ax.set(xlabel='Umbral de alerta p_TB', ylabel='Proporción', ylim=(0, 1.02), title='ViT-B/16 · Validation · umbrales TB')
        ax.legend()
        ax.grid(alpha=0.18)
        save(fig, 'vit_b_16_thresholds', 'Curva de umbrales TB de ViT-B/16', None,
             'Exploración de p_TB >= t en validation; no congela umbral operativo.', ['vit_b_16'])

    vit_summary = vit_entry[3]
    payload = {
        'schema': 'vit_phase5_report_payload_v1',
        'scope': 'validation_only',
        'test_evaluated': False,
        'external_evaluated': False,
        'candidate_by_macro_f1': best_entry[0],
        'comparison_4models': comparison,
        'vit_details': {
            'run_id': vit_summary['run_id'],
            'completed_epochs': vit_summary['completed_epochs'],
            'best_epoch': vit_summary['best_epoch'],
            'stop_reason': vit_summary['stop_reason'],
            'checkpoint_sha256': vit_summary['checkpoint_sha256'],
            'validation': vit_entry[7],
            'history': vit_entry[4],
            'thresholds_grid': vit_thresh_grid,
        },
        'candidate_thresholds_grid': best_thresh_grid,
        'figures': records,
        'source_sha256': sources,
    }
    atomic_json(output_dir / 'vit_report_payload.json', payload)
    return payload

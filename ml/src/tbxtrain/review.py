"""Curvas, comparaciones y bitácora sin evaluar el test."""
import json
from pathlib import Path
import numpy as np
from tbxdata.common import read_csv
from .metrics import binary_curve
from .protocol import atomic_json


def plot_run(run_dir):
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    history = read_csv(run_dir/'history.csv')
    metrics = json.loads((run_dir/'best_validation.json').read_text())
    figures = run_dir/'figures'
    figures.mkdir(exist_ok=True)
    epochs = [int(r['epoch']) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [('train_loss', 'Train'), ('validation_loss', 'Validación')]:
        axes[0].plot(epochs, [float(r[key]) for r in history], 'o-', label=label)
    axes[0].set(title='Pérdida ponderada', xlabel='Época', ylabel='Cross entropy')
    axes[1].plot(epochs, [float(r['validation_macro_f1']) for r in history], 'o-', label='Macro-F1')
    axes[1].plot(epochs, [float(r['validation_tb_recall']) for r in history], 'o-', label='Sensibilidad TB')
    axes[1].set(title='Validación por época', xlabel='Época', ylim=(0, 1))
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2); ax.set_xticks(epochs)
    fig.tight_layout(); fig.savefig(figures/'learning_curves.png', dpi=160); plt.close(fig)
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for (r, c), value in np.ndenumerate(cm):
        ax.text(c, r, str(value), ha='center', va='center', color='white' if value > cm.max()/2 else 'black')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=metrics['classes'], yticklabels=metrics['classes'],
           xlabel='Predicción', ylabel='Etiqueta real', title='Mejor época · validación')
    fig.tight_layout(); fig.savefig(figures/'confusion_matrix.png', dpi=160); plt.close(fig)
    rows = read_csv(run_dir/'validation_predictions.csv')
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, name in enumerate(metrics['classes']):
        curve = binary_curve([int(r['label']) == i for r in rows], [float(r[f'p_{name}']) for r in rows])
        if curve['roc_auc'] is not None:
            ax.plot(curve['fpr'], curve['tpr'], label=f"{name} AUC={curve['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Tasa de falsos positivos', ylabel='Sensibilidad', title='ROC OVR · validación')
    ax.legend(); fig.tight_layout(); fig.savefig(figures/'validation_roc.png', dpi=160); plt.close(fig)


def experiment_table(drive_base):
    records = []
    for path in sorted((Path(drive_base)/'experiments').glob('*/summary.json')):
        s = json.loads(path.read_text())
        if s.get('status') != 'completed_preliminary' or not s.get('training_executed'):
            continue
        m, c = s['validation'], s['config']
        records.append({'run_id': s['run_id'], 'parent_run': c['parent_run'],
            'epochs': s['completed_epochs'], 'best_epoch': s['best_epoch'],
            'macro_f1': m['macro_f1'], 'accuracy': m['accuracy'],
            'tb_recall': m['per_class']['TB']['recall'], 'tb_specificity': m['per_class']['TB']['specificity'],
            'hypothesis': c['hypothesis'], 'change_reason': c['change_reason']})
    return records


def record_review(run_dir, observation, next_hypothesis, decision):
    if not all(str(v).strip() for v in [observation, next_hypothesis, decision]):
        raise ValueError('Complete observación, hipótesis siguiente y decisión.')
    run_dir = Path(run_dir)
    summary = json.loads((run_dir/'summary.json').read_text())
    atomic_json(run_dir/'review.json', {'run_id': summary['run_id'],
        'validation_macro_f1': summary['validation']['macro_f1'],
        'observation': observation, 'next_hypothesis': next_hypothesis, 'decision': decision,
        'basis': 'validation; sin abrir test ni externos'})

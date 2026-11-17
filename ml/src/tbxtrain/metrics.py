"""Métricas de validación sin dependencia de torch ni scikit-learn.

Las curvas agrupan scores empatados. AP es average precision, no área trapezoidal.
Las clases sin positivos/negativos se informan como None en las áreas.
"""
import numpy as np


def binary_curve(target, score):
    target, score = np.asarray(target, dtype=bool), np.asarray(score, dtype=float)
    if target.ndim != 1 or target.shape != score.shape or not len(target):
        raise ValueError('Scores y etiquetas incompatibles o vacíos.')
    if not np.isfinite(score).all():
        raise ValueError('Scores no finitos.')
    positives, negatives = int(target.sum()), int((~target).sum())
    if not positives or not negatives:
        return {'roc_auc': None, 'average_precision': None, 'fpr': [], 'tpr': []}
    order = np.argsort(-score, kind='stable')
    y, s = target[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, 1+ends-np.cumsum(y)[ends]]
    tpr, fpr = tp/positives, fp/negatives
    precision = np.divide(tp[1:], tp[1:]+fp[1:])
    # Trapezoides explícitos: funciona también con NumPy 1.26.
    auc = np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
    ap = np.sum(np.diff(tpr)*precision)
    return {'roc_auc': float(auc), 'average_precision': float(ap),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist()}


def classification_metrics(labels, probabilities, classes):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    k = len(classes)
    if y.ndim != 1 or p.shape != (len(y), k) or not len(y):
        raise ValueError('Dimensiones incorrectas o evaluación vacía.')
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= k:
        raise ValueError('Etiquetas fuera del mapa de clases.')
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Probabilidades inválidas.')
    pred = p.argmax(1)
    cm = np.bincount(y*k+pred, minlength=k*k).reshape(k, k)
    supports, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    recall = np.divide(tp, supports, out=np.zeros(k, float), where=supports > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k, float), where=predicted > 0)
    f1 = np.divide(2*tp, supports+predicted, out=np.zeros(k, float), where=(supports+predicted) > 0)
    per_class = {}
    for i, name in enumerate(classes):
        negatives = len(y)-supports[i]
        tn = len(y)-supports[i]-predicted[i]+tp[i]
        curve = binary_curve(y == i, p[:, i])
        per_class[name] = {'support': int(supports[i]), 'precision': float(precision[i]),
                           'recall': float(recall[i]), 'f1': float(f1[i]),
                           'specificity': float(tn/negatives) if negatives else None,
                           'roc_auc': curve['roc_auc'], 'average_precision': curve['average_precision']}
    aucs = [v['roc_auc'] for v in per_class.values()]
    return {'n': len(y), 'accuracy': float(tp.sum()/len(y)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall[supports > 0].mean()),
            'macro_roc_auc_ovr': float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
            'confusion_matrix': cm.tolist(), 'classes': list(classes), 'per_class': per_class,
            'decision_rule': 'argmax softmax; sin ajuste de umbral',
            'evaluation_unit': 'imagen; sin intervalos de confianza ni garantía de independencia por paciente'}

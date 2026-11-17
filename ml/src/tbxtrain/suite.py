"""Comparación controlada reanudable; el cierre requiere evidencia de tres corridas.

Mantiene el entrenador histórico intacto. Nunca crea tensores de test/externos.
"""
import gc
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tbxdata.common import read_csv, sha256, write_csv
from tbxdata.dataset import TBXDataset, class_weights, load_config
from tbxdata.environment import in_colab
from tbxdata.validate import validate_manifests
from .protocol import atomic_json, provenance
from .train import evaluate, make_loader, atomic_checkpoint, rng_state, restore_rng, _seed

ARCHITECTURES = ('convnext_tiny', 'vmamba_tiny', 'hybrid_cross_attention')


def validate_suite(config):
    expected = set(json.loads((Path(__file__).resolve().parents[2]/'configs/phase3_suite.json').read_text()))
    if set(config) != expected or config['architectures'] != list(ARCHITECTURES):
        raise ValueError('Protocolo incompleto o arquitecturas distintas.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', config['protocol_id']): raise ValueError('protocol_id')
    for field in ['epochs', 'batch_size', 'accumulation_steps', 'patience']:
        if type(config[field]) is not int or config[field] < 1: raise ValueError(field)
    if not 0 <= config['warmup_epochs'] < config['epochs']: raise ValueError('warmup_epochs')
    if not 0 <= config['seed'] <= 2**32-1 or not 0 <= config['num_workers'] <= 16: raise ValueError('seed/workers')
    if config['precision'] != 'fp32' or config['selection_metric'] != 'validation_macro_f1':
        raise ValueError('Se fija FP32 y macro-F1 de validation para comparar.')
    for key in ['head_lr', 'backbone_lr', 'gradient_clip']:
        if not 0 < config[key] < float('inf'): raise ValueError(key)
    for key in ['weight_decay', 'min_delta']:
        if not 0 <= config[key] < float('inf'): raise ValueError(key)
    return dict(config)


def suite_fingerprint(project):
    project = Path(project)
    p = provenance(project)
    for file in sorted((project/'vendor/vmamba').glob('*')):
        if file.is_file(): p[file.relative_to(project).as_posix()] = sha256(file)
    return p


def epoch_train(torch, model, loader, optimizer, weights, config, epoch):
    model.configure_phase(epoch >= config['warmup_epochs'])
    optimizer.zero_grad(set_to_none=True)
    total_loss = denominator = window_weight = 0.0
    correct = count = updates = 0
    device = next(model.parameters()).device
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, dtype=torch.long, non_blocking=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights, reduction='sum')
        if not torch.isfinite(loss): raise FloatingPointError('Pérdida no finita.')
        loss.backward()
        w = weights[y].sum().item()
        total_loss += loss.item(); denominator += w; window_weight += w
        correct += (logits.detach().argmax(1) == y).sum().item(); count += len(y)
        if (i+1) % config['accumulation_steps'] == 0 or i+1 == len(loader):
            params = [p for p in model.parameters() if p.grad is not None]
            for p in params: p.grad.div_(window_weight)
            torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            window_weight = 0.; updates += 1
        if (i+1) % 200 == 0 or i+1 == len(loader):
            print(f'  época {epoch+1} lote {i+1}/{len(loader)} loss={total_loss/denominator:.4f}', flush=True)
    return {'train_loss': total_loss/denominator, 'train_accuracy': correct/count,
            'optimizer_updates': updates, 'train_n': count,
            'stage': 'head_only' if epoch < config['warmup_epochs'] else 'last_stages'}


def run_suite_model(project, dataset, drive_base, architecture, config, weights_path=None,
                    resume_run=None, smoke=False):
    if not in_colab(): raise RuntimeError('La ejecución real requiere Google Colab con GPU.')
    import torch
    import torchvision
    from .architectures import build_classifier, scan_backend
    if not torch.cuda.is_available(): raise RuntimeError('Colab no tiene GPU CUDA disponible.')
    config = validate_suite(config)
    if architecture not in ARCHITECTURES: raise ValueError('Arquitectura no admitida.')
    if architecture != 'convnext_tiny' and scan_backend() == 'torch_reference':
        raise RuntimeError('Falta el kernel CUDA de selective scan. Ejecute su instalación y reinicie el runtime si ya importó VMamba; el entrenamiento no usará silenciosamente el bucle de referencia.')
    project, dataset, drive_base = map(Path, (project, dataset, drive_base))
    validate_manifests(project)
    data_cfg = load_config(project/'configs/data.json')
    if data_cfg['classes'] != {'health': 0, 'sick': 1, 'TB': 2} or data_cfg['image_size'] != 512:
        raise ValueError('El protocolo exige tres clases y 512px congelados.')
    classes = ['health', 'sick', 'TB']
    fingerprints = suite_fingerprint(project)
    _seed(torch, config['seed'])
    resumed = None
    if resume_run:
        if smoke or any(c in str(resume_run) for c in '/\\:') or str(resume_run) in {'.', '..'}:
            raise ValueError('resume_run no válido.')
        run_id = str(resume_run)
        resumed = torch.load(drive_base/'checkpoints'/run_id/'last.pt', map_location='cpu', weights_only=False)
        if (resumed['architecture'] != architecture or resumed['config'] != config or resumed['provenance'] != fingerprints):
            raise ValueError('Reanudar exige arquitectura, protocolo, código y datos idénticos.')
    else:
        run_id = ('smoke_suite_' if smoke else 'phase3_')+architecture+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    out, checkpoints = drive_base/'experiments'/run_id, drive_base/'checkpoints'/run_id
    out.mkdir(parents=True, exist_ok=bool(resumed)); checkpoints.mkdir(parents=True, exist_ok=bool(resumed))
    history = resumed['history'] if resumed else []
    updates = resumed['updates'] if resumed else 0
    best_state = resumed['best_model_state'] if resumed else None
    best_score = resumed['best_score'] if resumed else -1.
    best_epoch = resumed['best_epoch'] if resumed else None
    stale = resumed['stale'] if resumed else 0
    start_epoch = resumed['next_epoch'] if resumed else 0
    model = optimizer = None
    def status(state, **extra):
        record = {'run_id': run_id, 'architecture': architecture, 'status': state,
            'completed_epochs': len(history), 'optimizer_updates': updates, 'kind': 'smoke' if smoke else 'controlled', **extra}
        atomic_json(out/'status.json', record)
        if not smoke:
            atomic_json(drive_base/'results/phase3_suite'/config['protocol_id']/(architecture+'.json'), record)
    status('initializing')
    print('Corrida:', run_id, '\nSalidas:', out, flush=True)
    try:
        model = build_classifier(architecture, pretrained=resumed is None, weights_path=weights_path).to('cuda')
        optimizer = torch.optim.AdamW(model.parameter_groups(config['head_lr'], config['backbone_lr']), weight_decay=config['weight_decay'])
        if resumed:
            model.load_state_dict(resumed['model_state'], strict=True)
            optimizer.load_state_dict(resumed['optimizer_state']); restore_rng(torch, resumed['rng'])
        resumed = None; gc.collect()
        train = TBXDataset(dataset, project/'manifests/train.csv', data_cfg, training=True)
        val = TBXDataset(dataset, project/'manifests/validation.csv', data_cfg, training=False)
        local_cfg = dict(config)
        if smoke:
            local_cfg.update(epochs=1, warmup_epochs=0, batch_size=1, accumulation_steps=2, num_workers=0)
            # Subconjunto técnico por clase; ninguna métrica smoke se usa en comparación.
            for ds in [train, val]:
                ds.rows = [next(r for r in ds.rows if r['class_name'] == c) for c in classes]
        weights = torch.tensor(class_weights(project/'manifests/train.csv', data_cfg), device='cuda')
        val_loader = make_loader(torch, val, local_cfg, False)
        atomic_json(out/'config.json', config); atomic_json(out/'provenance.json', fingerprints)
        atomic_json(out/'environment.json', {'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'precision': 'fp32',
            'scan_backend': scan_backend() if architecture != 'convnext_tiny' else None,
            'vmamba_weights_sha256': sha256(weights_path) if architecture != 'convnext_tiny' else None})
        atomic_json(out/'class_weights.json', dict(zip(classes, weights.cpu().tolist())))
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
                improved = metrics['macro_f1'] > best_score+config['min_delta']
                if improved:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_score, best_epoch, stale = metrics['macro_f1'], epoch+1, 0
                else: stale += 1
                history.append({'epoch': epoch+1, **result, 'validation_loss': metrics['loss'],
                    'validation_accuracy': metrics['accuracy'], 'validation_macro_f1': metrics['macro_f1'],
                    'validation_tb_recall': metrics['per_class']['TB']['recall'], 'seconds': time.monotonic()-tick})
                payload = {'schema': 'suite_v1', 'architecture': architecture, 'run_id': run_id,
                    'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'rng': rng_state(torch),
                    'config': config, 'provenance': fingerprints, 'history': history, 'updates': updates,
                    'best_model_state': best_state, 'best_score': best_score, 'best_epoch': best_epoch,
                    'stale': stale, 'next_epoch': epoch+1}
                atomic_checkpoint(torch, checkpoints/'last.pt', payload)
                write_csv(out/'history.csv', history)
                status('running', best_epoch=best_epoch)
                print(f'Época {epoch+1}: macro-F1={metrics["macro_f1"]:.4f}, TB recall={metrics["per_class"]["TB"]["recall"]:.4f}; Drive guardado.', flush=True)
                if stale >= config['patience']:
                    stop_reason = 'early_stopping'; break
        if smoke:
            after = [p.detach().cpu() for m in model.head_modules() for p in m.parameters()]
            if not updates or not any(not torch.equal(a, b) for a, b in zip(before, after)):
                raise RuntimeError('El smoke no actualizó la cabeza; no se declara aprobado.')
        model.load_state_dict(best_state)
        metrics, y, probs = evaluate(torch, model, val_loader, weights, False, classes)
        atomic_checkpoint(torch, checkpoints/'best.pt', {'architecture': architecture, 'run_id': run_id,
            'epoch': best_epoch, 'model_state': best_state, 'config': config, 'provenance': fingerprints,
            'classes': classes, 'data_config': data_cfg, 'validation': metrics})
        if smoke:
            reloaded = torch.load(checkpoints/'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(reloaded['model_state']); del reloaded
            _, yy, pp = evaluate(torch, model, val_loader, weights, False, classes)
            if not np.array_equal(yy, y) or not np.allclose(pp, probs, rtol=1e-4, atol=1e-4):
                raise RuntimeError('El checkpoint smoke no reproduce predicciones tras recarga.')
        write_csv(out/'validation_predictions.csv', [dict(relative_path=r['relative_path'], group_id=r['group_id'],
            label=int(label), prediction=int(p.argmax()), **{f'p_{c}': float(p[i]) for i, c in enumerate(classes)})
            for r, label, p in zip(val.rows, y, probs)])
        atomic_json(out/'best_validation.json', metrics)
        if not smoke:
            from .review import plot_run
            plot_run(out)
        summary = {'schema': 'suite_v1', 'run_id': run_id, 'architecture': architecture,
            'status': 'completed_smoke' if smoke else 'completed_controlled', 'training_executed': updates > 0,
            'protocol': config, 'completed_epochs': len(history), 'optimizer_updates': updates,
            'best_epoch': best_epoch, 'stop_reason': stop_reason, 'validation': metrics,
            'provenance': fingerprints, 'checkpoint_sha256': sha256(checkpoints/'best.pt'),
            'test_evaluated': False, 'external_evaluated': False, 'phase3_complete': False}
        atomic_json(out/'summary.json', summary); status(summary['status'])
        return summary
    except BaseException as exc:
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc)); raise
    finally:
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()


def close_phase3(drive_base, run_ids):
    """Cierre verificable. La elección final/umbrales/test quedan para Fase 5."""
    base = Path(drive_base)
    summaries = []
    if len(run_ids) != 3 or len(set(run_ids)) != 3: raise ValueError('Se requieren tres corridas distintas.')
    for run_id in run_ids:
        if any(c in str(run_id) for c in '/\\:') or run_id in {'.', '..'}: raise ValueError('ID inválido.')
        s = json.loads((base/'experiments'/run_id/'summary.json').read_text())
        if s['status'] != 'completed_controlled' or not s['training_executed'] or not s['optimizer_updates']:
            raise ValueError('Corrida sin entrenamiento controlado completo: '+run_id)
        if s['test_evaluated'] or s['external_evaluated'] or s['validation']['n'] != 1241:
            raise ValueError('Evaluación fuera del protocolo: '+run_id)
        if sha256(base/'checkpoints'/run_id/'best.pt') != s['checkpoint_sha256']:
            raise ValueError('Checkpoint alterado o incompleto: '+run_id)
        summaries.append(s)
    if {s['architecture'] for s in summaries} != set(ARCHITECTURES):
        raise ValueError('Falta alguna arquitectura requerida.')
    if any(s['protocol'] != summaries[0]['protocol'] or s['provenance'] != summaries[0]['provenance'] for s in summaries):
        raise ValueError('Las corridas no comparten protocolo, datos y código.')
    rows = []
    for s in summaries:
        m, tb = s['validation'], s['validation']['per_class']['TB']
        rows.append({'architecture': s['architecture'], 'run_id': s['run_id'], 'epochs': s['completed_epochs'],
            'best_epoch': s['best_epoch'], 'macro_f1': m['macro_f1'], 'accuracy': m['accuracy'],
            'TB_recall': tb['recall'], 'TB_precision': tb['precision'], 'TB_specificity': tb['specificity'],
            'TB_to_sick': m['confusion_matrix'][2][1], 'TB_to_health': m['confusion_matrix'][2][0]})
    output = base/'results'/('phase3_closure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'comparison.csv', rows)
    report = {'phase3_complete': True, 'scope': 'Tres clasificadores entrenados, comparados en validation y versionados.',
        'runs': list(run_ids), 'comparison': rows, 'single_seed': True,
        'winner_finalized': False, 'test_evaluated': False, 'phase4_complete': False, 'phase5_complete': False}
    atomic_json(output/'closure.json', report)
    lines = ['# Cierre de Fase 3: entrenamiento y comparación de clasificadores',
        '', 'Se completaron tres entrenamientos controlados con una seed, los manifiestos congelados y selección por macro-F1 de validation.',
        '', '| Arquitectura | Épocas | Mejor época | Macro-F1 | Accuracy | Sensibilidad TB | TB→sick | TB→health |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['architecture']} | {r['epochs']} | {r['best_epoch']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['TB_recall']:.4f} | {r['TB_to_sick']} | {r['TB_to_health']} |")
    lines += ['', '## Trazabilidad', '']+[f"- {s['architecture']}: `{s['run_id']}`, SHA-256 best.pt `{s['checkpoint_sha256']}`." for s in summaries]
    lines += ['', '## Alcance y continuidad', '',
        'Fase 3 completa para los tres clasificadores entrenados, comparados en validation y versionados. Los pesos se inicializaron en ImageNet por rama; el híbrido usa atención cruzada bidireccional.',
        'Se usó una seed y presupuesto limitado: no se afirma superioridad estadística. No se evaluaron test ni externos.',
        'La fase 4 corresponde al detector. La selección final, umbrales, concordancia, explicabilidad y evaluación reservada quedan para Fase 5.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report, output

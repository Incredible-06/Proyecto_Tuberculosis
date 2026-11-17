"""ConvNeXt-Tiny preliminar. La entrada pública exige Colab con GPU.

Solo train/validation generan tensores; test se lee como metadatos de integridad.
No se reinstala torch/CUDA ni se modifican imágenes o manifiestos.
"""
import gc
import json
import platform
import random
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tbxdata.common import sha256, write_csv
from tbxdata.dataset import TBXDataset, class_weights, load_config
from tbxdata.environment import in_colab
from tbxdata.validate import validate_manifests
from .metrics import classification_metrics
from .protocol import atomic_json, check_warm_start, provenance, validate_config


def build_model(torch, pretrained=True):
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 3)
    return model


def set_trainable(model, last_stage):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    if last_stage:
        for module in model.features[-2:]:  # downsample + última etapa de bloques
            for p in module.parameters():
                p.requires_grad_(True)
    model.train()
    # Desactiva stochastic depth de las etapas congeladas, también en warmup.
    model.features.eval()
    if last_stage:
        for module in model.features[-2:]:
            module.train()


def make_loader(torch, dataset, config, training, epoch=0):
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(config['seed']+epoch)
    return torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=training,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=False,
        generator=generator, drop_last=False)


def train_epoch(torch, model, loader, optimizer, scaler, weights, config, epoch):
    device = next(model.parameters()).device
    last_stage = config['unfreeze_last_stage'] and epoch >= config['warmup_epochs']
    set_trainable(model, last_stage)
    optimizer.zero_grad(set_to_none=True)
    numerator, denominator, window_denominator = 0.0, 0.0, 0.0
    correct, count, updates, skipped = 0, 0, 0, 0
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=config['amp']):
            logits = model(images)
            loss_sum = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction='sum')
        if not torch.isfinite(loss_sum).item():
            raise FloatingPointError('Pérdida no finita; revise runtime/configuración antes de continuar.')
        scaler.scale(loss_sum).backward()
        batch_denominator = weights[labels].sum().item()
        window_denominator += batch_denominator
        numerator += loss_sum.detach().item()
        denominator += batch_denominator
        correct += (logits.detach().argmax(1) == labels).sum().item()
        count += len(labels)
        end_window = (step+1) % config['accumulation_steps'] == 0 or step+1 == len(loader)
        if end_window:
            scaler.unscale_(optimizer)
            # Media ponderada exacta de toda la ventana, incluido el último lote.
            # Evita sesgo al promediar medias de microlotes con pesos distintos.
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(window_denominator)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config['gradient_clip'])
            if not config['amp'] and not torch.isfinite(norm).item():
                raise FloatingPointError('Gradientes no finitos.')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            was_skipped = scaler.get_scale() < old_scale
            updates += int(not was_skipped)
            skipped += int(was_skipped)
            optimizer.zero_grad(set_to_none=True)
            window_denominator = 0.0
        if (step+1) % 100 == 0 or step+1 == len(loader):
            print(f'  época {epoch+1} | lote {step+1}/{len(loader)} | loss {numerator/denominator:.4f}', flush=True)
    return {'train_loss': numerator/denominator, 'train_accuracy': correct/count,
            'train_n': count, 'optimizer_updates': updates, 'amp_skipped_updates': skipped,
            'stage': 'last_stage' if last_stage else 'head_only'}


def evaluate(torch, model, loader, weights, amp, classes):
    device = next(model.parameters()).device
    model.eval()
    probabilities, labels_all = [], []
    weighted_sum, weight_sum, plain_sum = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            logits = logits.float()
            nll = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
            if not torch.isfinite(nll).all().item():
                raise FloatingPointError('Predicciones no finitas en validación.')
            weighted_sum += (nll*weights[labels]).sum().item()
            weight_sum += weights[labels].sum().item()
            plain_sum += nll.sum().item()
            probabilities.append(torch.softmax(logits, 1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    y, p = np.concatenate(labels_all), np.concatenate(probabilities)
    metrics = classification_metrics(y, p, classes)
    metrics.update(loss=weighted_sum/weight_sum, unweighted_loss=plain_sum/len(y), split='validation')
    return metrics, y, p


def rng_state(torch):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(torch, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_checkpoint(torch, destination, payload):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Serializa en disco temporal de Colab y publica mediante un rename en Drive.
    with tempfile.TemporaryDirectory(prefix='tbx_checkpoint_') as temp:
        source = Path(temp)/'checkpoint.pt'
        torch.save(payload, source)
        target = destination.with_name(destination.name+'.tmp')
        shutil.copyfile(source, target)
        if target.stat().st_size != source.stat().st_size:
            raise OSError('Checkpoint incompleto al copiar a Drive.')
        target.replace(destination)


def _seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_experiment(project_root, dataset_root, drive_base, config, resume_run=None,
                   init_checkpoint=None):
    """Nueva corrida, reanudación exacta por época o ajuste desde best.pt propio.

    resume_run mantiene configuración y código idénticos. init_checkpoint inicia
    otro experimento con optimizador nuevo. No se aceptan ambos a la vez.
    """
    if not in_colab():
        raise RuntimeError('Entrenamiento habilitado exclusivamente en Google Colab con GPU.')
    import torch
    import torchvision
    import PIL
    if not torch.cuda.is_available():
        raise RuntimeError('Seleccione GPU L4/T4/A100 en Colab y vuelva a conectar. No hay CUDA disponible.')
    config = validate_config(config)
    if resume_run and init_checkpoint:
        raise ValueError('Elija reanudar o iniciar un ajuste, no ambos.')
    project, dataset_root, drive_base = map(Path, (project_root, dataset_root, drive_base))
    data_config = load_config(project/'configs/data.json')
    if data_config['classes'] != {'health': 0, 'sick': 1, 'TB': 2}:
        raise ValueError('Mapa de clases distinto del baseline revisado.')
    if data_config['image_size'] != 512:
        raise ValueError('Este baseline conserva la resolución congelada de 512 px.')
    classes = sorted(data_config['classes'], key=data_config['classes'].get)
    validate_manifests(project)  # metadatos; no tensores ni métricas de test
    fingerprints = provenance(project)
    _seed(torch, config['seed'])
    resumed, parent = None, None
    if resume_run:
        # Restringe el destino a una corrida dentro del proyecto de Drive.
        run_id = str(resume_run)
        if Path(run_id).name != run_id or '/' in run_id or '\\' in run_id or ':' in run_id or run_id in {'.', '..'}:
            raise ValueError('resume_run debe ser solo el nombre de una corrida existente.')
        checkpoint_path = drive_base/'checkpoints'/run_id/'last.pt'
        # Solo checkpoints propios: torch.load con weights_only=False contiene RNG.
        resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if resumed['config'] != config or resumed['provenance'] != fingerprints:
            raise ValueError('Reanudar exige configuración, datos y código idénticos; cree otro experimento para ajustes.')
        if resumed['run_id'] != run_id:
            raise ValueError('Identidad de checkpoint incorrecta.')
    else:
        run_id = config['experiment_label']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        if init_checkpoint:
            if not config['parent_run']:
                raise ValueError('Registre parent_run para iniciar desde otro modelo.')
            parent = torch.load(Path(init_checkpoint), map_location='cpu', weights_only=False)
            check_warm_start(parent['provenance'], fingerprints)
            if parent['classes'] != classes or parent['run_id'] != config['parent_run']:
                raise ValueError('Checkpoint y parent_run no coinciden.')
    run_dir, checkpoint_dir = drive_base/'experiments'/run_id, drive_base/'checkpoints'/run_id
    if not resumed:
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
                   'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'gpu': props.name, 'gpu_memory_bytes': props.total_memory,
                   'seed': config['seed'], 'deterministic_algorithms': 'warn_only',
                   'reproducibility': 'RNG restaurable por época; no se promete igualdad entre GPUs o versiones.'}
    atomic_json(run_dir/'environment.json', environment)
    atomic_json(run_dir/'config.json', config)
    atomic_json(run_dir/'data_config.json', data_config)
    atomic_json(run_dir/'provenance.json', fingerprints)
    shutil.copy2(project/'notebooks/02_convnext_tiny_baseline.ipynb', run_dir/'notebook_source_unexecuted.ipynb')
    for split in ['train', 'validation']:
        shutil.copy2(project/f'manifests/{split}.csv', run_dir/f'{split}.csv')
    history = resumed['history'] if resumed else []
    best_score = resumed['best_score'] if resumed else -1.0
    best_epoch = resumed['best_epoch'] if resumed else None
    best_state = resumed['best_model_state'] if resumed else None
    best_metrics = resumed['best_metrics'] if resumed else None
    stale = resumed['stale'] if resumed else 0
    start_epoch = resumed['next_epoch'] if resumed else 0
    updates = resumed['optimizer_updates_total'] if resumed else 0
    started = time.monotonic()
    def status(state, **extra):
        atomic_json(run_dir/'status.json', {'run_id': run_id, 'status': state,
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'completed_epochs': len(history), 'training_executed': updates > 0,
            'optimizer_updates': updates, 'phase3_complete': False, **extra})
    status('initializing')
    print('Experimento:', run_id, '\nResultados:', run_dir, '\nCheckpoints:', checkpoint_dir, flush=True)
    try:
        model = build_model(torch, pretrained=not (resumed or parent)).to('cuda')
        if resumed:
            model.load_state_dict(resumed['model_state'])
        elif parent:
            model.load_state_dict(parent['model_state'])
            atomic_json(run_dir/'initialization.json', {'parent_run': parent['run_id'],
                'parent_epoch': parent['epoch'], 'checkpoint_sha256': sha256(init_checkpoint),
                'optimizer': 'nuevo; no es reanudación'})
        optimizer = torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': config['head_lr']},
            {'params': [p for m in model.features[-2:] for p in m.parameters()], 'lr': config['backbone_lr']}
        ], weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        if resumed:
            optimizer.load_state_dict(resumed['optimizer_state'])
            scaler.load_state_dict(resumed['scaler_state'])
            restore_rng(torch, resumed['rng_state'])
        train = TBXDataset(dataset_root, project/'manifests/train.csv', data_config, training=True)
        validation = TBXDataset(dataset_root, project/'manifests/validation.csv', data_config, training=False)
        weights = torch.tensor(class_weights(project/'manifests/train.csv', data_config), device='cuda')
        atomic_json(run_dir/'class_weights.json', dict(zip(classes, weights.cpu().tolist())))
        val_loader = make_loader(torch, validation, config, False)
        if not resumed:
            initial, _, _ = evaluate(torch, model, val_loader, weights, config['amp'], classes)
            atomic_json(run_dir/'initial_validation.json', initial)
            print(f"Antes de actualizar pesos: validation macro-F1={initial['macro_f1']:.4f}")
        # Libera copias del checkpoint antes de entrenar; conserva best_state.
        resumed = parent = None
        gc.collect()
        status('running')
        stop_reason = 'epoch_budget'
        if stale >= config['patience']:
            stop_reason = 'early_stopping'
        else:
            for epoch in range(start_epoch, config['epochs']):
                epoch_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                loader = make_loader(torch, train, config, True, epoch)
                result = train_epoch(torch, model, loader, optimizer, scaler, weights, config, epoch)
                updates += result['optimizer_updates']
                metrics, _, _ = evaluate(torch, model, val_loader, weights, config['amp'], classes)
                improved = metrics['macro_f1'] > best_score+config['min_delta']
                if improved:
                    best_score, best_epoch, best_metrics, stale = metrics['macro_f1'], epoch+1, metrics, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                history.append({'epoch': epoch+1, **result, 'validation_loss': metrics['loss'],
                    'validation_accuracy': metrics['accuracy'], 'validation_macro_f1': metrics['macro_f1'],
                    'validation_tb_recall': metrics['per_class']['TB']['recall'],
                    'validation_tb_specificity': metrics['per_class']['TB']['specificity'],
                    'seconds': time.monotonic()-epoch_started,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'best_so_far': improved})
                payload = {'schema_version': 1, 'run_id': run_id, 'epoch': epoch+1, 'next_epoch': epoch+1,
                    'config': config, 'classes': classes, 'provenance': fingerprints,
                    'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(), 'rng_state': rng_state(torch),
                    'history': history, 'best_score': best_score, 'best_epoch': best_epoch,
                    'best_model_state': best_state, 'best_metrics': best_metrics, 'stale': stale,
                    'optimizer_updates_total': updates}
                # last.pt es autosuficiente: contiene también el mejor estado.
                atomic_checkpoint(torch, checkpoint_dir/'last.pt', payload)
                if improved:
                    atomic_checkpoint(torch, checkpoint_dir/'best.pt', {
                        'schema_version': 1, 'run_id': run_id, 'epoch': best_epoch,
                        'model_state': best_state, 'classes': classes, 'config': config,
                        'data_config': data_config, 'provenance': fingerprints, 'validation': best_metrics})
                write_csv(run_dir/'history.csv', history)
                atomic_json(run_dir/'best_validation.json', best_metrics)
                status('running', best_epoch=best_epoch, best_macro_f1=best_score)
                print(f"Época {epoch+1}: val macro-F1={metrics['macro_f1']:.4f}, "
                      f"sensibilidad TB={metrics['per_class']['TB']['recall']:.4f}; guardado en Drive.", flush=True)
                if stale >= config['patience']:
                    stop_reason = 'early_stopping'
                    break
        if best_state is None or updates == 0:
            raise RuntimeError('No se completaron actualizaciones; no hay modelo entrenado que presentar.')
        model.load_state_dict(best_state)
        metrics, y, p = evaluate(torch, model, val_loader, weights, config['amp'], classes)
        # Reconstruye best.pt también tras una publicación interrumpida en Drive.
        atomic_checkpoint(torch, checkpoint_dir/'best.pt', {'schema_version': 1, 'run_id': run_id,
            'epoch': best_epoch, 'model_state': best_state, 'classes': classes, 'config': config,
            'data_config': data_config, 'provenance': fingerprints, 'validation': metrics})
        predictions = []
        for row, label, scores in zip(validation.rows, y, p):
            predictions.append({'relative_path': row['relative_path'], 'group_id': row['group_id'],
                'label': int(label), 'prediction': int(scores.argmax()),
                **{f'p_{name}': float(scores[i]) for i, name in enumerate(classes)}})
        write_csv(run_dir/'validation_predictions.csv', predictions)
        write_csv(run_dir/'history.csv', history)
        atomic_json(run_dir/'best_validation.json', metrics)
        from .review import plot_run
        plot_run(run_dir)
        summary = {'run_id': run_id, 'status': 'completed_preliminary', 'training_executed': True,
            'phase3_complete': False, 'stop_reason': stop_reason, 'completed_epochs': len(history),
            'optimizer_updates': updates, 'best_epoch': best_epoch, 'validation': metrics,
            'session_seconds': time.monotonic()-started, 'config': config,
            'checkpoint_sha256': {name: sha256(checkpoint_dir/name) for name in ['best.pt', 'last.pt']},
            'test_evaluated': False, 'external_evaluated': False,
            'scope': 'Un clasificador preliminar; comparación de arquitecturas y evaluación final pendientes.'}
        atomic_json(run_dir/'summary.json', summary)
        status('completed_preliminary', best_epoch=best_epoch, stop_reason=stop_reason)
        print('Corrida preliminar terminada. Fase 3 todavía abierta. Revise curvas y errores antes del siguiente ajuste.')
        return summary
    except BaseException as exc:
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc),
               resume_from='Última época completa en last.pt; la época parcial se repite.')
        raise
    finally:
        # No conserva referencias al modelo entre la primera corrida y el ajuste.
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()


def run_smoke(project_root, dataset_root, drive_base, config):
    """Prueba técnica con 2 imágenes por clase/split; no produce un baseline.

    Verifica el camino real de GPU/AMP/gradientes/checkpoint con datos de train y
    validation. No usa test, no inicia la corrida principal desde estos pesos.
    """
    if not in_colab():
        raise RuntimeError('La prueba smoke GPU se ejecuta exclusivamente en Google Colab.')
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('La prueba smoke requiere un runtime GPU de Colab.')
    config = validate_config(config)
    project, dataset_root, drive_base = map(Path, (project_root, dataset_root, drive_base))
    validate_manifests(project)
    data_config = load_config(project/'configs/data.json')
    classes = sorted(data_config['classes'], key=data_config['classes'].get)
    run_id = 'smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = drive_base/'experiments'/run_id
    checkpoint = drive_base/'checkpoints'/run_id/'smoke.pt'
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'status.json', {'status': 'running_smoke', 'kind': 'smoke', 'run_id': run_id})
    _seed(torch, config['seed'])
    model = None
    try:
        datasets, loaders, selected = {}, {}, {}
        for split in ['train', 'validation']:
            ds = TBXDataset(dataset_root, project/f'manifests/{split}.csv', data_config)
            indices = []
            for name in classes:
                members = [i for i, row in enumerate(ds.rows) if row['class_name'] == name][:2]
                if len(members) != 2:
                    raise ValueError('Smoke requiere dos imágenes por clase.')
                indices.extend(members)
            datasets[split] = ds
            selected[split] = [ds.rows[i]['relative_path'] for i in indices]
            loaders[split] = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, indices),
                batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        model = build_model(torch, pretrained=True).to('cuda')
        optimizer = torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': config['head_lr']},
            {'params': [p for m in model.features[-2:] for p in m.parameters()], 'lr': config['backbone_lr']}
        ], weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        weights = torch.tensor(class_weights(project/'manifests/train.csv', data_config), device='cuda')
        smoke_config = dict(config, batch_size=2, accumulation_steps=2, num_workers=0)
        before = model.classifier[2].weight.detach().cpu().clone()
        result = train_epoch(torch, model, loaders['train'], optimizer, scaler, weights,
                             smoke_config, epoch=config['warmup_epochs'])
        if result['optimizer_updates'] == 0 or torch.equal(before, model.classifier[2].weight.detach().cpu()):
            raise RuntimeError('Smoke sin actualización efectiva de la cabeza.')
        metrics, y, p = evaluate(torch, model, loaders['validation'], weights, config['amp'], classes)
        atomic_checkpoint(torch, checkpoint, {'kind': 'smoke', 'run_id': run_id,
            'model_state': model.state_dict(), 'classes': classes,
            'warning': 'Pesos de comprobación técnica; no usar como modelo base o inicialización de la corrida principal.'})
        # Comprueba que se leen los bytes publicados en Drive y que recuperan
        # las mismas predicciones. Se reutiliza el objeto para limitar memoria.
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(loaded['model_state'])
        del loaded
        _, reloaded_y, reloaded_p = evaluate(torch, model, loaders['validation'], weights, config['amp'], classes)
        if not np.array_equal(y, reloaded_y) or not np.allclose(p, reloaded_p, atol=1e-4, rtol=1e-4):
            raise RuntimeError('La recarga del checkpoint no reproduce las predicciones smoke.')
        summary = {'status': 'completed_smoke', 'kind': 'smoke', 'run_id': run_id,
            'technical_check_passed': True, 'is_baseline': False, 'phase3_complete': False,
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'input_shape': [2, 3, data_config['image_size'], data_config['image_size']],
            'optimizer_updates': result['optimizer_updates'], 'checkpoint_reload_passed': True,
            'checkpoint_sha256': sha256(checkpoint), 'selected_paths': selected,
            'validation_n': len(y),
            'metrics_scope': 'Solo comprobación numérica con 6 imágenes por split; sin estimación de calidad del modelo.',
            'provenance': provenance(project), 'test_evaluated': False}
        atomic_json(output/'summary.json', summary)
        atomic_json(output/'status.json', {'status': 'completed_smoke', 'kind': 'smoke', 'run_id': run_id})
        print('Smoke OK: datos → GPU → gradientes → actualización → Drive → recarga. NO cuenta como baseline.')
        return summary
    except BaseException as exc:
        atomic_json(output/'status.json', {'status': 'failed_smoke', 'kind': 'smoke', 'run_id': run_id, 'error': repr(exc)})
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

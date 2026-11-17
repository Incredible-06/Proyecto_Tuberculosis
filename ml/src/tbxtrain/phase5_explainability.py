"""Explicabilidad verificada para los 5 clasificadores de Fase 5.

Implementa Grad-CAM por rama, Attention Rollout para ViT-B/16 y mapas de
atención cruzada para CrossFusion, junto con chequeos numéricos obligatorios:
1) finitud y no nulidad de gradientes/activaciones,
2) sensibilidad a clase objetivo (TB vs health),
3) prueba de aleatorización de cabeza/fusión sobre copia en memoria,
4) alineación espacial auxiliar (pointing game y energía en cajas GT).

No modifica módulos protegidos ni modelos entrenados en disco.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tbxdata.common import safe_path, write_csv
from .protocol import atomic_json

HYBRID_ARCHS = {'hybrid_cross_attention', 'hybrid_convnext_vit'}


def _normalize_cam_2d(tensor_2d):
    arr = tensor_2d.detach().cpu().numpy().astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if not np.isfinite(arr).all() or mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32), mn, mx, False
    return ((arr - mn) / (mx - mn)).astype(np.float32), mn, mx, True


def _gradcam_4d(act, grad):
    """Grad-CAM estándar sobre tensor espacial (B, C, H, W)."""
    if act.ndim != 4 or grad.ndim != 4 or act.shape != grad.shape:
        raise ValueError(f'Tensores incompatibles para Grad-CAM: {act.shape} vs {grad.shape}')
    weights = grad.mean(dim=(2, 3), keepdim=True)
    raw = F.relu((weights * act).sum(dim=1, keepdim=False))
    cams = []
    stats = []
    for b in range(act.shape[0]):
        cam_norm, mn, mx, non_trivial = _normalize_cam_2d(raw[b])
        cams.append(cam_norm)
        stats.append({
            'activation_norm': float(act[b].detach().norm().item()),
            'gradient_norm': float(grad[b].detach().norm().item()),
            'activation_finite': bool(torch.isfinite(act[b]).all().item()),
            'gradient_finite': bool(torch.isfinite(grad[b]).all().item()),
            'raw_min': mn,
            'raw_max': mx,
            'non_trivial': non_trivial,
        })
    return np.stack(cams, axis=0), stats


def _tokens_to_spatial(tokens, grid_size):
    """Convierte tokens (B, 1 + G*G, D) en mapa espacial (B, D, G, G)."""
    spatial = tokens[:, 1:, :]
    b, n, d = spatial.shape
    if n != grid_size * grid_size:
        raise ValueError(f'Número de tokens espaciales {n} != {grid_size}*{grid_size}')
    return spatial.transpose(1, 2).reshape(b, d, grid_size, grid_size)


def _vit_rollout_and_logits(vit_model, x):
    """Ejecuta forward explícito de ViT capturando atenciones por bloque (sin mutar modelo)."""
    tokens = vit_model._process_input(x)
    b = x.shape[0]
    cls_tok = vit_model.class_token.expand(b, -1, -1)
    seq = torch.cat([cls_tok, tokens], dim=1)
    seq = seq + vit_model.encoder.pos_embedding
    seq = vit_model.encoder.dropout(seq)

    n_tokens = seq.shape[1]
    eye = torch.eye(n_tokens, device=seq.device, dtype=seq.dtype).unsqueeze(0)
    rollout = eye.expand(b, -1, -1).clone()

    for block in vit_model.encoder.layers:
        h = block.ln_1(seq)
        attn_out, attn_weights = block.self_attention(
            h, h, h, need_weights=True, average_attn_weights=False
        )
        seq = seq + block.dropout(attn_out)
        seq = seq + block.mlp(block.ln_2(seq))
        # Promedio sobre cabezas + conexión residual + renormalización por fila.
        attn_mean = attn_weights.mean(dim=1)
        aug = 0.5 * attn_mean + 0.5 * eye
        aug = aug / aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.bmm(aug, rollout)

    seq = vit_model.encoder.ln(seq)
    grid = int(round((n_tokens - 1) ** 0.5))
    cls_to_patches = rollout[:, 0, 1:].reshape(b, grid, grid)
    rollout_maps = []
    for i in range(b):
        norm_map, _, _, _ = _normalize_cam_2d(cls_to_patches[i])
        rollout_maps.append(norm_map)
    return seq, np.stack(rollout_maps, axis=0)


def _cross_fusion_attention_maps(fusion_mod, local_map, global_map):
    """Extrae mapas de atención cruzada 8x8 de CrossFusion sin modificar architectures.py."""
    with torch.no_grad():
        local = fusion_mod.local_proj(
            F.adaptive_avg_pool2d(local_map.detach(), fusion_mod.grid).flatten(2).transpose(1, 2)
        )
        glob = fusion_mod.global_proj(
            F.adaptive_avg_pool2d(global_map.detach(), fusion_mod.grid).flatten(2).transpose(1, 2)
        )
        ln = fusion_mod.local_norm(local)
        gn = fusion_mod.global_norm(glob)
        _, attn_l2g = fusion_mod.local_from_global(ln, gn, gn, need_weights=True, average_attn_weights=False)
        _, attn_g2l = fusion_mod.global_from_local(gn, ln, ln, need_weights=True, average_attn_weights=False)
        # Importancia promedio recibida por cada token clave (grid x grid).
        g = fusion_mod.grid
        global_attn_map = attn_l2g.mean(dim=(1, 2)).reshape(-1, g, g)
        local_attn_map = attn_g2l.mean(dim=(1, 2)).reshape(-1, g, g)
        out_local = [_normalize_cam_2d(local_attn_map[i])[0] for i in range(local_map.shape[0])]
        out_global = [_normalize_cam_2d(global_attn_map[i])[0] for i in range(global_map.shape[0])]
    return {
        'cross_attn_local_tokens': np.stack(out_local, axis=0),
        'cross_attn_global_tokens': np.stack(out_global, axis=0),
    }


def upsample_map(cam_2d, size=512):
    """Interpolación bilineal de un mapa 2D normalizado a resolución objetivo."""
    t = torch.from_numpy(np.asarray(cam_2d, dtype=np.float32))[None, None]
    up = F.interpolate(t, size=(int(size), int(size)), mode='bilinear', align_corners=False)[0, 0].numpy()
    mn, mx = float(up.min()), float(up.max())
    if mx > mn + 1e-12:
        up = (up - mn) / (mx - mn)
    return up.astype(np.float32)


def explain_batch(model, architecture, x, target_class=2, image_size=512):
    """Calcula mapas de relevancia verificados y probabilidades para un batch (sin mutar pesos)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    x_in = x.detach().clone().requires_grad_(True)
    captured = {}
    handles = []

    def save_hook(name):
        def _hook(_mod, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor.retain_grad()
            captured[name] = tensor
        return _hook

    try:
        if architecture == 'convnext_tiny':
            handles.append(model.encoder.features[-1].register_forward_hook(save_hook('conv')))
        elif architecture == 'vmamba_tiny':
            handles.append(model.encoder.layers[-1].register_forward_hook(save_hook('mamba')))
        elif architecture == 'hybrid_cross_attention':
            handles.append(model.conv[-1].register_forward_hook(save_hook('conv')))
            handles.append(model.mamba.layers[-1].register_forward_hook(save_hook('mamba')))
        elif architecture == 'vit_b_16':
            layers = model.encoder.encoder.layers
            if len(layers) >= 2:
                handles.append(layers[-2].register_forward_hook(save_hook('vit_tokens')))
            else:
                handles.append(model.encoder.conv_proj.register_forward_hook(save_hook('vit_conv_proj')))
        elif architecture == 'hybrid_convnext_vit':
            handles.append(model.conv[-1].register_forward_hook(save_hook('conv')))
            handles.append(model.vit.encoder.ln.register_forward_hook(save_hook('vit_tokens')))
        else:
            raise ValueError('Arquitectura no soportada en explicabilidad: ' + str(architecture))

        logits = model(x_in)
        probs = logits.softmax(dim=1).detach().cpu().numpy()
        score = logits[:, int(target_class)].sum()
        score.backward()
        for h in handles:
            h.remove()
        handles.clear()

        branches = {}
        stats = {}
        grid = int(image_size) // 16

        if 'conv' in captured:
            act = captured['conv']
            grad = act.grad
            if grad is None:
                raise RuntimeError('Gradiente nulo en rama ConvNeXt.')
            cams, st = _gradcam_4d(act, grad)
            branches['conv_gradcam'] = cams
            stats['conv_gradcam'] = st

        if 'mamba' in captured:
            act = captured['mamba']
            grad = act.grad
            if grad is None:
                raise RuntimeError('Gradiente nulo en rama VMamba.')
            cams, st = _gradcam_4d(act, grad)
            branches['mamba_gradcam'] = cams
            stats['mamba_gradcam'] = st

        if 'vit_tokens' in captured:
            tok = captured['vit_tokens']
            tok_grad = tok.grad
            if tok_grad is None:
                raise RuntimeError('Gradiente nulo en tokens espaciales ViT.')
            act_4d = _tokens_to_spatial(tok, grid)
            grad_4d = _tokens_to_spatial(tok_grad, grid)
            cams, st = _gradcam_4d(act_4d, grad_4d)
            branches['vit_token_gradcam'] = cams
            stats['vit_token_gradcam'] = st

        if 'vit_conv_proj' in captured:
            act = captured['vit_conv_proj']
            grad = act.grad
            if grad is None:
                raise RuntimeError('Gradiente nulo en conv_proj de ViT.')
            cams, st = _gradcam_4d(act, grad)
            branches['vit_token_gradcam'] = cams
            stats['vit_token_gradcam'] = st

        # Attention rollout y CrossFusion (en torch.no_grad usando los mapas ya capturados).
        if architecture == 'vit_b_16':
            with torch.no_grad():
                _, rollout_maps = _vit_rollout_and_logits(model.encoder, x_in.detach())
            branches['vit_rollout'] = rollout_maps
        elif architecture == 'hybrid_convnext_vit':
            with torch.no_grad():
                _, rollout_maps = _vit_rollout_and_logits(model.vit, x_in.detach())
                vit_spatial = _tokens_to_spatial(captured['vit_tokens'].detach(), grid)
                cf_maps = _cross_fusion_attention_maps(model.fusion, captured['conv'].detach(), vit_spatial)
            branches['vit_rollout'] = rollout_maps
            branches.update(cf_maps)
        elif architecture == 'hybrid_cross_attention':
            cf_maps = _cross_fusion_attention_maps(
                model.fusion, captured['conv'].detach(), captured['mamba'].detach()
            )
            branches.update(cf_maps)

        # Mapa combinado a resolución 512x512 por muestra.
        upsampled = {k: np.stack([upsample_map(v[i], image_size) for i in range(x.shape[0])], axis=0)
                     for k, v in branches.items()}
        if architecture == 'convnext_tiny':
            primary_key = 'conv_gradcam'
        elif architecture == 'vmamba_tiny':
            primary_key = 'mamba_gradcam'
        elif architecture == 'hybrid_cross_attention':
            upsampled['hybrid_combined'] = np.clip(
                0.5 * upsampled['conv_gradcam'] + 0.5 * upsampled['mamba_gradcam'], 0.0, 1.0
            )
            primary_key = 'hybrid_combined'
        elif architecture == 'vit_b_16':
            primary_key = 'vit_token_gradcam'
        else:
            upsampled['hybrid_combined'] = np.clip(
                0.5 * upsampled['conv_gradcam'] + 0.5 * upsampled['vit_token_gradcam'], 0.0, 1.0
            )
            primary_key = 'hybrid_combined'

        return {
            'architecture': architecture,
            'target_class': int(target_class),
            'probabilities': probs,
            'logits': logits.detach().cpu().numpy(),
            'raw_maps': branches,
            'upsampled_maps': upsampled,
            'primary_key': primary_key,
            'branch_stats': stats,
        }
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


def _pearson_corr(a, b):
    fa = np.asarray(a, dtype=float).ravel()
    fb = np.asarray(b, dtype=float).ravel()
    sa, sb = fa.std(), fb.std()
    if sa < 1e-12 or sb < 1e-12:
        return None
    return float(np.corrcoef(fa, fb)[0, 1])


def _gt_box_alignment(cam_512, gt_boxes_xywh):
    if not gt_boxes_xywh:
        return {'pointing_game_hit': None, 'gt_energy_ratio': None, 'gt_area_fraction': None, 'gt_enrichment': None}
    h, w = cam_512.shape
    mask = np.zeros((h, w), dtype=bool)
    for box in gt_boxes_xywh:
        x0, y0, bw, bh = box['bbox']
        x1 = max(0, int(np.floor(x0)))
        y1 = max(0, int(np.floor(y0)))
        x2 = min(w, int(np.ceil(x0 + bw)))
        y2 = min(h, int(np.ceil(y0 + bh)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    max_y, max_x = np.unravel_index(int(np.argmax(cam_512)), cam_512.shape)
    hit = bool(mask[max_y, max_x])
    total_energy = float(cam_512.sum())
    energy_ratio = float(cam_512[mask].sum() / total_energy) if total_energy > 1e-12 else 0.0
    area_frac = float(mask.mean())
    enrichment = (energy_ratio / area_frac) if area_frac > 1e-12 else None
    return {
        'pointing_game_hit': hit,
        'gt_energy_ratio': energy_ratio,
        'gt_area_fraction': area_frac,
        'gt_enrichment': enrichment,
    }


def _randomize_head(model):
    """Aleatoriza únicamente la cabeza/fusión de una copia en memoria para sanity check."""
    randomized = copy.deepcopy(model)
    modules = randomized.head_modules() if hasattr(randomized, 'head_modules') else [randomized]
    with torch.no_grad():
        for root_mod in modules:
            for mod in root_mod.modules():
                if hasattr(mod, 'reset_parameters'):
                    mod.reset_parameters()
                elif isinstance(mod, nn.MultiheadAttention):
                    mod._reset_parameters()
    return randomized.eval()


def run_verified_explainability_suite(
    model,
    architecture,
    samples,
    device='cuda',
    image_size=512,
    target_class=2,
    contrast_class=0,
    run_randomization_check=True,
):
    """Ejecuta explicabilidad + chequeos numéricos sobre una lista acotada de muestras.

    Cada elemento de `samples` debe ser un dict con:
      - `relative_path`, `class_name`, `tensor` (numpy CHW float32), `boxes_json` (str)
    """
    randomized_model = _randomize_head(model).to(device) if run_randomization_check else None
    records = []
    visuals = []
    try:
        for sample in samples:
            x = torch.from_numpy(sample['tensor']).unsqueeze(0).to(device)
            res_tb = explain_batch(model, architecture, x, target_class=target_class, image_size=image_size)
            res_contrast = explain_batch(model, architecture, x, target_class=contrast_class, image_size=image_size)

            primary_tb = res_tb['upsampled_maps'][res_tb['primary_key']][0]
            primary_contrast = res_contrast['upsampled_maps'][res_contrast['primary_key']][0]

            all_finite = all(
                st[0]['activation_finite'] and st[0]['gradient_finite']
                for st in res_tb['branch_stats'].values()
            )
            all_nonzero_grad = all(st[0]['gradient_norm'] > 0.0 for st in res_tb['branch_stats'].values())
            all_nontrivial = all(st[0]['non_trivial'] for st in res_tb['branch_stats'].values())

            contrast_corr = _pearson_corr(primary_tb, primary_contrast)
            contrast_l1 = float(np.abs(primary_tb - primary_contrast).mean())

            rand_corr = None
            rand_l1 = None
            if randomized_model is not None:
                res_rand = explain_batch(randomized_model, architecture, x, target_class=target_class, image_size=image_size)
                primary_rand = res_rand['upsampled_maps'][res_rand['primary_key']][0]
                rand_corr = _pearson_corr(primary_tb, primary_rand)
                rand_l1 = float(np.abs(primary_tb - primary_rand).mean())

            gt_boxes = json.loads(sample['boxes_json'])
            gt_align = _gt_box_alignment(primary_tb, gt_boxes)

            rec = {
                'architecture': architecture,
                'relative_path': sample['relative_path'],
                'class_name': sample['class_name'],
                'p_health': float(res_tb['probabilities'][0, 0]),
                'p_sick': float(res_tb['probabilities'][0, 1]),
                'p_TB': float(res_tb['probabilities'][0, 2]),
                'all_finite': all_finite,
                'all_nonzero_grad': all_nonzero_grad,
                'all_nontrivial_maps': all_nontrivial,
                'class_contrast_pearson': contrast_corr,
                'class_contrast_l1': contrast_l1,
                'head_randomization_pearson': rand_corr,
                'head_randomization_l1': rand_l1,
                **gt_align,
                'branch_stats': {k: v[0] for k, v in res_tb['branch_stats'].items()},
            }
            records.append(rec)
            visuals.append({
                'sample': sample,
                'primary_key': res_tb['primary_key'],
                'upsampled_maps': {k: v[0] for k, v in res_tb['upsampled_maps'].items()},
                'probabilities': res_tb['probabilities'][0].tolist(),
            })
    finally:
        if randomized_model is not None:
            del randomized_model
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return records, visuals


def render_explainability_panels(dataset_root, visuals_by_arch, output_dir, image_size=512):
    """Exporta figuras comparativas de explicabilidad por imagen (ramas + GT) en PNG y PDF."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image
    from tbxdata.dataset import letterbox

    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archs = list(visuals_by_arch.keys())
    n_samples = len(visuals_by_arch[archs[0]])
    exported = []

    for idx in range(n_samples):
        sample = visuals_by_arch[archs[0]][idx]['sample']
        with Image.open(safe_path(dataset_root, sample['relative_path'])) as im:
            rgb = np.asarray(letterbox(im, image_size))
        gt_boxes = json.loads(sample['boxes_json'])

        # Construir columnas: Original + cada mapa principal (y ramas para híbridos/ViT).
        panels = [('Original + GT', rgb, None, None)]
        for arch in archs:
            vis = visuals_by_arch[arch][idx]
            p_tb = vis['probabilities'][2]
            if arch in {'convnext_tiny', 'vmamba_tiny'}:
                key = vis['primary_key']
                panels.append((f"{arch}\n(p_TB={p_tb:.2f})", rgb, vis['upsampled_maps'][key], None))
            elif arch == 'vit_b_16':
                panels.append((f"ViT-B/16 Grad-CAM\n(p_TB={p_tb:.2f})", rgb, vis['upsampled_maps']['vit_token_gradcam'], None))
                panels.append(("ViT-B/16 Rollout", rgb, vis['upsampled_maps']['vit_rollout'], None))
            elif arch == 'hybrid_cross_attention':
                panels.append((f"C+VM Conv\n(p_TB={p_tb:.2f})", rgb, vis['upsampled_maps']['conv_gradcam'], None))
                panels.append(("C+VM Mamba", rgb, vis['upsampled_maps']['mamba_gradcam'], None))
                panels.append(("C+VM Fusión", rgb, vis['upsampled_maps']['hybrid_combined'], None))
            elif arch == 'hybrid_convnext_vit':
                panels.append((f"C+ViT Conv\n(p_TB={p_tb:.2f})", rgb, vis['upsampled_maps']['conv_gradcam'], None))
                panels.append(("C+ViT ViT-Tok", rgb, vis['upsampled_maps']['vit_token_gradcam'], None))
                panels.append(("C+ViT Fusión", rgb, vis['upsampled_maps']['hybrid_combined'], None))

        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(2.9 * n_cols, 3.6))
        for ax, (title, base_rgb, heatmap, _) in zip(np.atleast_1d(axes), panels):
            ax.imshow(base_rgb)
            if heatmap is not None:
                ax.imshow(heatmap, cmap='jet', alpha=0.45, vmin=0.0, vmax=1.0)
            for gt in gt_boxes:
                gx, gy, gw, gh = gt['bbox']
                ax.add_patch(patches.Rectangle((gx, gy), gw, gh, lw=1.6, edgecolor='#00FF7F', facecolor='none'))
            ax.set_title(title, fontsize=8.5)
            ax.axis('off')
        fig.suptitle(f"{sample['class_name']} · {sample['relative_path']} (clase objetivo: TB)", fontsize=10.5)
        fig.tight_layout()
        stem = f"explain_{idx + 1:02d}_{Path(sample['relative_path']).stem}"
        for ext in ('png', 'pdf'):
            fig.savefig(output_dir / f'{stem}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        exported.append(f'{stem}.png')
    return exported


def summarize_explainability_checks(records, output_dir):
    """Guarda resumen CSV/JSON de los chequeos numéricos de explicabilidad."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in records:
        flat = {k: v for k, v in r.items() if k != 'branch_stats'}
        flat['branches_checked'] = ';'.join(sorted(r['branch_stats'].keys()))
        flat['min_grad_norm'] = min(b['gradient_norm'] for b in r['branch_stats'].values())
        flat_rows.append(flat)
    write_csv(output_dir / 'explainability_checks.csv', flat_rows)
    atomic_json(output_dir / 'explainability_checks.json', records)
    return flat_rows

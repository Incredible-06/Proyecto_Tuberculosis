"""Tres arquitecturas reales para la comparación controlada de Fase 3.

VMamba-T s1l8 oficial; el híbrido fusiona mapas espaciales mediante atención
cruzada bidireccional, no promedia probabilidades de clasificadores.
"""
import hashlib
import ast
import importlib.util
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tbxdata.common import sha256

ROOT = Path(__file__).resolve().parents[2]


def vmamba_module():
    name = '_tbx_official_vmamba'
    if name in sys.modules:
        return sys.modules[name]
    folder = ROOT/'vendor/vmamba'
    info = json.loads((folder/'source.json').read_text())
    for file, expected in info['files'].items():
        if hashlib.sha256((folder/file).read_bytes()).hexdigest() != expected:
            raise ValueError('Fuente oficial alterada: '+file)
    spec = importlib.util.spec_from_file_location(name, folder/'vmamba.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # El fichero oficial declara kernels @triton.jit incluso cuando falla su
    # importación. Guarda esas declaraciones para permitir fixtures CPU; en
    # Colab con Triton se ejecutan normalmente. No cambia la ecuación SS2D.
    tree = ast.parse((folder/'vmamba.py').read_text(encoding='utf-8'))
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and any(
            'triton.' in ast.unparse(d) for d in node.decorator_list):
            tree.body[i] = ast.If(test=ast.Name(id='WITH_TRITON', ctx=ast.Load()), body=[node], orelse=[])
    try:
        exec(compile(ast.fix_missing_locations(tree), str(folder/'vmamba.py'), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    # El archivo oficial simplificado soporta mamba_ssm; este adaptador usa
    # csms6s oficial, que también permite el kernel oflex del mismo repositorio.
    spec2 = importlib.util.spec_from_file_location('_tbx_official_scan', folder/'csms6s.py')
    scan = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(scan)
    module.selective_scan_fn = scan.selective_scan_fn
    module.tbx_scan = scan
    return module


def scan_backend():
    scan = vmamba_module().tbx_scan
    for name, flag in [('oflex', 'WITH_SELECTIVESCAN_OFLEX'), ('core', 'WITH_SELECTIVESCAN_CORE'), ('mamba', 'WITH_SELECTIVESCAN_MAMBA')]:
        if getattr(scan, flag, False):
            return name
    return 'torch_reference'


def vmamba_tiny(pretrained=True, weights_path=None):
    module = vmamba_module()
    model = module.vmamba_tiny_s1l8(pretrained=False, channel_first=True)
    if pretrained:
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError('Faltan pesos VMamba ImageNet. Ejecute la preparación de dependencias/pesos.')
        expected = json.loads((ROOT/'vendor/vmamba/source.json').read_text())['checkpoint_sha256']
        if sha256(weights_path) != expected:
            raise ValueError('Los pesos no coinciden con el checkpoint ImageNet oficial verificado.')
        # Checkpoint oficial: se usa exclusivamente el state_dict, con validación
        # estricta de todas las capas antes de sustituir la cabeza de 1000 clases.
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
    model.classifier.head = nn.Identity()
    return model


class Classifier(nn.Module):
    def configure_phase(self, fine_tune):
        self.train()
        for p in self.parameters():
            p.requires_grad_(False)
        for module in self.head_modules():
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)
        for module in self.backbones():
            module.eval()
        if fine_tune:
            for module in self.last_modules():
                module.train()
                for p in module.parameters():
                    p.requires_grad_(True)

    def parameter_groups(self, head_lr, backbone_lr):
        head = [p for m in self.head_modules() for p in m.parameters()]
        backbone = [p for m in self.last_modules() for p in m.parameters()]
        if len({id(p) for p in head+backbone}) != len(head+backbone):
            raise ValueError('Parámetros duplicados en optimizador.')
        return [{'params': head, 'lr': head_lr}, {'params': backbone, 'lr': backbone_lr}]


class ConvNeXtClassifier(Classifier):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.classifier[2] = nn.Linear(768, 3)
    def forward(self, x): return self.encoder(x)
    def head_modules(self): return [self.encoder.classifier]
    def backbones(self): return [self.encoder.features]
    def last_modules(self): return list(self.encoder.features[-2:])


class VMambaClassifier(Classifier):
    def __init__(self, pretrained=True, weights_path=None):
        super().__init__()
        self.encoder = vmamba_tiny(pretrained, weights_path)
        self.head = nn.Linear(768, 3)
    def forward(self, x): return self.head(self.encoder(x))
    def head_modules(self): return [self.encoder.classifier.norm, self.head]
    def backbones(self): return [self.encoder.patch_embed, self.encoder.layers]
    def last_modules(self): return [self.encoder.layers[-2].downsample, self.encoder.layers[-1]]


class CrossFusion(nn.Module):
    def __init__(self, channels=768, dim=256, heads=4, grid=8):
        super().__init__()
        self.grid = grid
        self.local_proj, self.global_proj = nn.Linear(channels, dim), nn.Linear(channels, dim)
        self.local_norm, self.global_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.local_from_global = nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True)
        self.global_from_local = nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(2*dim), nn.Dropout(.2), nn.Linear(2*dim, 3))
    def forward(self, local_map, global_map):
        # Rejilla común: conserva tokens espaciales y limita el coste cuadrático.
        local = self.local_proj(F.adaptive_avg_pool2d(local_map, self.grid).flatten(2).transpose(1, 2))
        glob = self.global_proj(F.adaptive_avg_pool2d(global_map, self.grid).flatten(2).transpose(1, 2))
        ln, gn = self.local_norm(local), self.global_norm(glob)
        local = local+self.local_from_global(ln, gn, gn, need_weights=False)[0]
        glob = glob+self.global_from_local(gn, ln, ln, need_weights=False)[0]
        return self.output(torch.cat([local.mean(1), glob.mean(1)], dim=1))


class HybridClassifier(Classifier):
    def __init__(self, pretrained=True, weights_path=None):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv = conv.features
        self.mamba = vmamba_tiny(pretrained, weights_path)
        # La fusión recibe mapas anteriores a pooling/clasificación de Mamba.
        # Elimina norm clasificadora sin uso: todos los parámetros activos tienen gradiente.
        self.mamba.classifier = nn.Identity()
        self.fusion = CrossFusion()
    def forward(self, x): return self.fusion(self.conv(x), self.mamba(x))
    def head_modules(self): return [self.fusion]
    def backbones(self): return [self.conv, self.mamba.patch_embed, self.mamba.layers]
    def last_modules(self):
        return list(self.conv[-2:])+[self.mamba.layers[-2].downsample, self.mamba.layers[-1]]


def build_classifier(name, pretrained=True, weights_path=None):
    if name == 'convnext_tiny': return ConvNeXtClassifier(pretrained)
    if name == 'vmamba_tiny': return VMambaClassifier(pretrained, weights_path)
    if name == 'hybrid_cross_attention': return HybridClassifier(pretrained, weights_path)
    raise ValueError('Arquitectura no implementada: '+str(name))

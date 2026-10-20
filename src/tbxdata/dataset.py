"""Lector de clasificación: PIL + NumPy; compatible con DataLoader de PyTorch.

No importa torch. DataLoader puede convertir automáticamente las matrices NumPy
CHW en tensores CPU. Las cajas originales se consultan aparte, sin transformarlas.
"""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from .common import read_csv, safe_path


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['augmentation'].get('clahe'):
        raise ValueError('CLAHE está desactivado en este baseline; requiere una ablación separada.')
    if config['augmentation'].get('horizontal_flip'):
        raise ValueError('El protocolo de esta fase no incluye espejado.')
    if len(config['normalization']['mean'])!=3 or min(config['normalization']['std'])<=0:
        raise ValueError('Normalización RGB inválida.')
    return config


def letterbox(image, size):
    image = image.convert('RGB')
    if image.size == (size,size):
        return image.copy()
    return ImageOps.pad(image,(size,size),method=Image.Resampling.LANCZOS,color=(0,0,0),centering=(0.5,0.5))


class RadiographTransform:
    def __init__(self, config, training=False):
        self.config, self.training = config, training
        self.mean = np.asarray(config['normalization']['mean'],dtype=np.float32)[:,None,None]
        self.std = np.asarray(config['normalization']['std'],dtype=np.float32)[:,None,None]

    def image(self, image, rng=None):
        rng = rng or random.Random(self.config['seed'])
        image = letterbox(image,self.config['image_size'])
        if self.training:
            aug = self.config['augmentation']
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(1-aug['brightness'],1+aug['brightness']))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(1-aug['contrast'],1+aug['contrast']))
            angle = rng.uniform(-aug['rotation_degrees'],aug['rotation_degrees'])
            # expand=True conserva esquinas; letterbox evita recortar el tórax.
            image = image.rotate(angle,resample=Image.Resampling.BILINEAR,expand=True,fillcolor=(0,0,0))
            image = letterbox(image,self.config['image_size'])
        return image

    def __call__(self, image, rng=None):
        rgb = self.image(image,rng)
        a = np.asarray(rgb,dtype=np.float32).transpose(2,0,1)/np.float32(255)
        return np.ascontiguousarray((a-self.mean)/self.std,dtype=np.float32)

    def denormalize(self, array):
        a = np.asarray(array,dtype=np.float32)*self.std+self.mean
        return np.clip(np.rint(a.transpose(1,2,0)*255),0,255).astype(np.uint8)


class TBXDataset:
    """Devuelve (imagen float32 CHW normalizada, etiqueta entera).

    Solo acepta manifests supervisados. Augmentación por (seed, época, ruta):
    reproducible entre workers. En Fase 2 llamar set_epoch antes de cada época y
    usar persistent_workers=False para propagar la época a los workers nuevos.
    """
    def __init__(self, dataset_root, manifest, config, training=None):
        self.root = Path(dataset_root)
        self.rows = read_csv(manifest)
        self.config = load_config(config) if isinstance(config,(str,Path)) else config
        if not self.rows:
            raise ValueError('Manifiesto vacío.')
        splits = {r['split'] for r in self.rows}
        if len(splits)!=1 or not splits <= {'train','validation','test'}:
            raise ValueError('El lector supervisado no admite test oficial ni fuentes externas.')
        self.split = next(iter(splits))
        for r in self.rows:
            if r['class_name'] not in self.config['classes'] or int(r['label']) != self.config['classes'][r['class_name']]:
                raise ValueError('Etiqueta inválida en manifiesto supervisado.')
            safe_path(self.root,r['relative_path'])
        training = self.split=='train' if training is None else training
        if training and self.split!='train':
            raise ValueError('No se permiten aumentos en validation/test.')
        self.transform = RadiographTransform(self.config,training)
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if int(epoch)<0:
            raise ValueError('Época negativa.')
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        seed_bytes = f"{self.config['seed']}:{self.epoch}:{row['relative_path']}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8],'big'))
        with Image.open(safe_path(self.root,row['relative_path'])) as im:
            array = self.transform(im,rng)
        return array,int(row['label'])

    def metadata(self, index):
        """Las cajas se refieren a la imagen ORIGINAL, no al tensor aumentado."""
        row = dict(self.rows[index])
        for key in ['boxes_json','tb_subtypes_json','exact_aliases_json']:
            row[key.removesuffix('_json')] = json.loads(row.pop(key))
        row['box_coordinate_space'] = 'original_512_image_xywh'
        return row


def class_weights(train_manifest, config):
    rows = read_csv(train_manifest)
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('Los pesos se calculan exclusivamente desde train.')
    counts = np.bincount([int(r['label']) for r in rows],minlength=len(config['classes']))
    if min(counts)<=0:
        raise ValueError('Hay clases sin ejemplos en train.')
    return (len(rows)/(len(counts)*counts)).astype(np.float32)

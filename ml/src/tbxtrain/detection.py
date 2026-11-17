"""Inicio Fase 4: conversión explícita de COCO xywh a YOLO, test cerrado.

Una clase de detección 'tb_lesion' reúne todas las cajas TB originales. No se
generan cajas para health/sick. TB sin cajas se excluye, nunca es un negativo.
"""
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from tbxdata.common import read_csv, safe_path, write_csv
from .protocol import atomic_json


def yolo_box(box, width, height):
    x, y, w, h = map(float, box['bbox'])
    if not all(math.isfinite(v) for v in [x,y,w,h,width,height]) or min(width,height) <= 0:
        raise ValueError('Caja/dimensiones no finitas o no positivas.')
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width+.001 or y+h > height+.001:
        raise ValueError('Caja COCO inválida; no se corrige ni recorta silenciosamente.')
    return ((x+w/2)/width, (y+h/2)/height, w/width, h/height)


def prepare_detection(project, dataset, output):
    from tbxdata.validate import validate_manifests
    project, dataset, output = map(Path, (project, dataset, output))
    validate_manifests(project)
    output.mkdir(parents=True, exist_ok=False)
    ledger, counts, smoke_lists = [], {}, {}
    for split in ['train', 'validation']:
        folder = 'train' if split == 'train' else 'val'
        images, labels = output/'images'/folder, output/'labels'/folder
        images.mkdir(parents=True); labels.mkdir(parents=True)
        counter = Counter()
        positives, negatives = [], []
        for row in read_csv(project/f'manifests/{split}.csv'):
            boxes = json.loads(row['boxes_json'])
            if row['class_name'] == 'TB' and not boxes:
                ledger.append({'relative_path': row['relative_path'], 'split': split, 'decision': 'excluded_TB_missing_boxes', 'boxes': 0})
                counter['excluded_TB_missing_boxes'] += 1
                continue
            if row['class_name'] != 'TB' and boxes:
                raise ValueError('Caso no TB con cajas: requiere revisión antes de exportar.')
            name = row['relative_path'].replace('/', '__')
            shutil.copyfile(safe_path(dataset, row['relative_path']), images/name)
            content = []
            for box in boxes:
                content.append('0 '+' '.join(f'{v:.9f}' for v in yolo_box(box, int(row['width']), int(row['height']))))
            (labels/Path(name).with_suffix('.txt')).write_text('\n'.join(content)+('\n' if content else ''))
            decision = 'positive_TB_boxes' if boxes else 'negative_no_TB_boxes'
            (positives if boxes else negatives).append(str((images/name).resolve()))
            ledger.append({'relative_path': row['relative_path'], 'split': split, 'decision': decision, 'boxes': len(boxes)})
            counter[decision] += 1; counter['boxes'] += len(boxes)
        counts[split] = dict(counter)
        smoke_file = output/('smoke_'+folder+'.txt')
        smoke_file.write_text('\n'.join(positives[:3]+negatives[:3])+'\n')
        smoke_lists[folder] = str(smoke_file.resolve())
    # JSON es un subconjunto válido de YAML; evita dependencia para exportar.
    import yaml
    spec = {'path': str(output.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': {0:'tb_lesion'}}
    (output/'dataset.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    smoke_spec = dict(spec, train=smoke_lists['train'], val=smoke_lists['val'])
    (output/'smoke.yaml').write_text(yaml.safe_dump(smoke_spec, sort_keys=False))
    write_csv(output/'conversion_ledger.csv', ledger)
    report = {'status': 'prepared_not_trained', 'counts': counts, 'test_exported': False,
              'class_policy': 'Cajas TB originales unificadas como tb_lesion; subtipos no se usan como clases de detección.',
              'missing_box_policy': 'TB sin cajas excluida; nunca negativo.', 'phase4_complete': False}
    atomic_json(output/'conversion.json', report)
    return report

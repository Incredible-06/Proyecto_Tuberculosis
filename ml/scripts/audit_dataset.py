"""Auditoria descriptiva de TBX11K; no entrena ni modifica datos originales.

Una imagen a la vez, un proceso, pausa por imagen y sin torch ni modelos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def duplicates(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row['path'])
    return [v for v in groups.values() if len(v) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('TBX11K'))
    parser.add_argument('--output', type=Path, default=Path('reports/initial_audit'))
    parser.add_argument('--pause', type=float, default=0.01)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError('La salida debe estar fuera del dataset original.')
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    lists = {p.stem: p.read_text().splitlines() for p in sorted((root/'lists').glob('*.txt'))}
    listsets = {k: set(v) for k, v in lists.items()}
    xmls, xml_errors = {}, []
    for p in sorted((root/'annotations/xml').glob('*.xml')):
        try:
            tree = ET.parse(p).getroot()
            width, height = int(tree.findtext('size/width')), int(tree.findtext('size/height'))
            objects = []
            for obj in tree.findall('object'):
                box = [int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                objects.append({'name': obj.findtext('name'), 'difficult': int(obj.findtext('difficult', '0')), 'box': box})
            xmls[p.stem] = {'size': [width,height], 'objects': objects, 'tags': sorted({e.tag for e in tree.iter()})}
        except Exception as exc:
            xml_errors.append({'file': p.name, 'error': str(exc)})
    coco = {}
    coco_summary = {}
    coco_issues = []
    for p in sorted((root/'annotations/json').glob('*.json')):
        d = json.loads(p.read_text())
        coco[p.stem] = d
        ids = {im['id']: im for im in d['images']}
        cats = {c['id'] for c in d['categories']}
        for ann in d['annotations']:
            im = ids.get(ann['image_id'])
            if im is None or ann['category_id'] not in cats:
                coco_issues.append([p.name, ann['id'], 'referencia invalida'])
                continue
            x,y,w,h = ann['bbox']
            if min(x,y) < 0 or min(w,h) <= 0 or x+w > im['width']+1e-3 or y+h > im['height']+1e-3:
                coco_issues.append([p.name, ann['id'], 'bbox fuera de limites'])
        paths = [im['file_name'] for im in d['images']]
        coco_summary[p.stem] = {'images':len(paths), 'annotations':len(d['annotations']),
            'categories':d['categories'], 'duplicate_image_ids':len(d['images'])-len(ids),
            'duplicate_annotation_ids':len(d['annotations'])-len({a['id'] for a in d['annotations']}),
            'duplicate_paths':len(paths)-len(set(paths)),
            'list_symmetric_difference':sorted(set(paths)^listsets[p.stem]) if p.stem in listsets else None}
    rows, errors = [], []
    files = sorted(p for p in (root/'imgs').rglob('*') if p.is_file())
    for idx, p in enumerate(files, 1):
        rel = p.relative_to(root/'imgs').as_posix()
        row = {'path':rel, 'bytes':p.stat().st_size, 'folder':p.parent.relative_to(root/'imgs').as_posix(),
               'suffix':p.suffix, 'memberships':';'.join(k for k,s in listsets.items() if rel in s),
               'sha256':digest(p), 'width':'', 'height':'', 'mode':'', 'format':'', 'rgb_sha256':'', 'dhash64':''}
        try:
            with Image.open(p) as im:
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                im.verify()
            with Image.open(p) as im:
                im.load()
                rgb = im.convert('RGB')
                h = hashlib.sha256(f'{im.width}x{im.height}:RGB:'.encode())
                h.update(rgb.tobytes())
                row['rgb_sha256'] = h.hexdigest()
                small = rgb.convert('L').resize((9,8), Image.Resampling.BILINEAR)
                px = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
                bits = 0
                for y in range(8):
                    for x in range(8):
                        bits = (bits << 1) | (px[y*9+x] > px[y*9+x+1])
                row['dhash64'] = f'{bits:016x}'
        except Exception as exc:
            errors.append({'file':rel, 'error':str(exc)})
        rows.append(row)
        if idx % 1000 == 0:
            print(f'{idx}/{len(files)} imagenes auditadas, {time.time()-start:.1f}s', flush=True)
        time.sleep(max(0,args.pause))
    with (out/'image_inventory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    actual = {r['path'] for r in rows}
    list_audit = {k:{'count':len(v), 'duplicates':len(v)-len(set(v)), 'missing':sorted(set(v)-actual),
                    'by_folder':dict(Counter(Path(x).parent.as_posix() for x in v))} for k,v in lists.items()}
    splits = {k:listsets['TBX11K_'+k] for k in ('train','val')}
    splits['test'] = listsets['all_test']
    def memberships(paths):
        return sorted(k for k,s in splits.items() if set(paths)&s)
    sha_groups = duplicates(rows, 'sha256')
    pixel_groups = duplicates(rows, 'rgb_sha256')
    dhash_groups = duplicates(rows, 'dhash64')
    image_categories = Counter(tuple(sorted({o['name'] for o in d['objects']})) for d in xmls.values())
    xml_invalid = []
    for stem, d in xmls.items():
        w,h = d['size']
        for o in d['objects']:
            a,b,c,e = o['box']
            if min(a,b)<0 or c<=a or e<=b or c>w or e>h:
                xml_invalid.append({'file':stem, 'object':o})
    cat_ids = {c['name']:c['id'] for c in coco['TBX11K_trainval']['categories']}
    cd = coco['TBX11K_trainval']
    paths_by_id = {im['id']:im['file_name'] for im in cd['images']}
    cb = defaultdict(list)
    for a in cd['annotations']:
        cb[paths_by_id[a['image_id']]].append((a['category_id'],a['bbox']))
    xml_coco_mismatches = []
    excluded_xml_objects = []
    for stem,d in xmls.items():
        w,h = d['size']
        expected = []
        for obj in d['objects']:
            x,y,x2,y2 = obj['box']
            bw,bh = (x2-x)*512/w,(y2-y)*512/h
            if obj['difficult'] or min(bw,bh)<1:
                excluded_xml_objects.append({'file':stem, 'object':obj, 'scaled_wh':[bw,bh]})
                continue
            expected.append((cat_ids[obj['name']], [x*512/w-1, y*512/h-1,bw,bh]))
        observed = cb.get('tb/'+stem+'.png',[])
        unmatched = list(observed)
        for cat,box in expected:
            match = next((i for i,(c,b) in enumerate(unmatched) if c==cat and max(abs(a-z) for a,z in zip(box,b))<0.001), None)
            if match is None:
                xml_coco_mismatches.append({'file':stem, 'expected':[cat,box]})
            else:
                unmatched.pop(match)
        if unmatched:
            xml_coco_mismatches.append({'file':stem,'extra_coco':unmatched})
    summary = {'python':platform.python_version(), 'pillow':Image.__version__,
        'images':len(rows), 'bytes':sum(r['bytes'] for r in rows), 'by_folder':dict(Counter(r['folder'] for r in rows)),
        'dimensions':dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
        'modes':dict(Counter(r['mode'] for r in rows)), 'formats':dict(Counter(r['format'] for r in rows)),
        'lists':list_audit, 'unlisted_images':sorted(actual-set().union(*listsets.values())),
        'split_path_overlaps':{a+'__'+b:sorted(splits[a]&splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
        'trainval_matches_union':{prefix:listsets[prefix+'trainval']==listsets[prefix+'train']|listsets[prefix+'val'] for prefix in ['TBX11K_','all_']},
        'image_errors':errors, 'xml_errors':xml_errors, 'xml_files':len(xmls),
        'xml_sizes':dict(Counter(f"{d['size'][0]}x{d['size'][1]}" for d in xmls.values())),
        'xml_tags':sorted(set().union(*(set(d['tags']) for d in xmls.values()))),
        'xml_object_categories':dict(Counter(o['name'] for d in xmls.values() for o in d['objects'])),
        'xml_image_category_sets':{' + '.join(k) or '(empty)':v for k,v in image_categories.items()},
        'xml_difficult':sum(o['difficult']!=0 for d in xmls.values() for o in d['objects']),
        'xml_invalid_boxes':xml_invalid, 'excluded_xml_objects':excluded_xml_objects,
        'xml_coco_mismatches':xml_coco_mismatches,
        'tb_without_coco_boxes':sorted(p for p in listsets['TBX11K_trainval'] if p.startswith('tb/') and p not in cb),
        'coco':coco_summary, 'coco_issues':coco_issues,
        'exact_file_duplicates':[{'paths':g,'core_splits':memberships(g), 'combined_splits':sorted(k for k in ['all_train','all_val','all_test'] if set(g)&listsets[k])} for g in sha_groups],
        'exact_rgb_duplicates':[{'paths':g,'core_splits':memberships(g), 'combined_splits':sorted(k for k in ['all_train','all_val','all_test'] if set(g)&listsets[k])} for g in pixel_groups],
        'dhash_equal_candidates':[{'paths':g,'core_splits':memberships(g), 'combined_splits':sorted(k for k in ['all_train','all_val','all_test'] if set(g)&listsets[k])} for g in dhash_groups],
        'elapsed_seconds':round(time.time()-start,1),
        'limitations':['No se infieren pacientes a partir de nombres de archivo.',
            'dHash igual es solo un candidato visual; no demuestra identidad ni independencia por paciente.',
            'No se crean particiones ni etiquetas nuevas. El inventario refleja las listas existentes.']}
    (out/'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    # Solo muestras del train oficial; no selecciona ejemplos usando test.
    train = sorted(splits['train'])
    selections = []
    for folder in ['health/','sick/']:
        selections.extend((p,folder[:-1]) for p in train if p.startswith(folder))
        selections = selections[:2] if folder=='health/' else selections[:4]
    for name in ['ActiveTuberculosis','ObsoletePulmonaryTuberculosis','PulmonaryTuberculosis']:
        candidates = [p for p in train if p.startswith('tb/') and {o['name'] for o in xmls[Path(p).stem]['objects']}=={name}]
        selections.extend((p,name) for p in candidates[:2])
    mixed = [p for p in train if p.startswith('tb/') and len({o['name'] for o in xmls[Path(p).stem]['objects']})>1]
    selections.extend((p,'Mixed lesion labels') for p in mixed[:2])
    colors={1:'#ff664f',2:'#47dcff',3:'#ffe66b'}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    tile_w,tile_h,cols=360,410,4
    canvas=Image.new('RGB',(cols*tile_w,((len(selections)+cols-1)//cols)*tile_h),(22,26,33))
    draw=ImageDraw.Draw(canvas)
    sample_records=[]
    for idx,(p,label) in enumerate(selections):
        x0,y0=(idx%cols)*tile_w,(idx//cols)*tile_h
        with Image.open(root/'imgs'/p) as im:
            rgb=im.convert('RGB').resize((352,352))
        canvas.paste(rgb,(x0+4,y0+4))
        for cat,box in cb.get(p,[]):
            x,y,w,h=box
            draw.rectangle((x0+4+x*352/512,y0+4+y*352/512,x0+4+(x+w)*352/512,y0+4+(y+h)*352/512),outline=colors[cat],width=2)
        draw.text((x0+5,y0+360),p,fill='white',font=font)
        draw.text((x0+5,y0+382),label,fill='#c2d2de',font=font)
        sample_records.append({'path':p,'label':label,'boxes':cb.get(p,[])})
    canvas.save(out/'representative_samples.jpg',quality=92)
    (out/'sample_records.json').write_text(json.dumps(sample_records,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['images','by_folder','dimensions','modes','formats','image_errors','xml_object_categories','xml_image_category_sets','tb_without_coco_boxes','xml_coco_mismatches','elapsed_seconds']}, indent=2),flush=True)
    print('DUPLICATES file / RGB / dHash:',len(sha_groups),len(pixel_groups),len(dhash_groups),flush=True)


if __name__=='__main__':
    main()

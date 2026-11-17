"""Resumen ligero y trazabilidad de los resultados de audit_dataset.py."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    out = Path('reports/initial_audit')
    root = Path('TBX11K')
    p = out/'audit_summary.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    rows = list(csv.DictReader((out/'image_inventory.csv').open(encoding='utf-8', newline='')))
    lists = {p.stem:set(p.read_text().splitlines()) for p in (root/'lists').glob('*.txt')}
    by_path = {r['path']:r for r in rows}
    data['coco_image_file_issues'] = []
    for jp in sorted((root/'annotations/json').glob('*.json')):
        for im in json.loads(jp.read_text())['images']:
            actual = by_path.get(im['file_name'])
            if actual is None or (int(actual['width']),int(actual['height'])) != (im['width'],im['height']):
                data['coco_image_file_issues'].append({'json':jp.name,'image':im})
    xml_stems = {p.stem for p in (root/'annotations/xml').glob('*.xml')}
    tb_stems = {Path(p).stem for p in lists['TBX11K_trainval'] if p.startswith('tb/')}
    data['xml_tb_file_symmetric_difference'] = sorted(xml_stems ^ tb_stems)
    for key in ['exact_file_duplicates','exact_rgb_duplicates','dhash_equal_candidates']:
        for group in data[key]:
            group['combined_splits'] = sorted(k for k in ['all_train','all_val','all_test'] if set(group['paths']) & lists[k])
    data['by_folder_details'] = {}
    for folder in sorted({r['folder'] for r in rows}):
        subset = [r for r in rows if r['folder']==folder]
        data['by_folder_details'][folder] = {
            'count':len(subset), 'dimensions':dict(Counter(r['width']+'x'+r['height'] for r in subset)),
            'modes':dict(Counter(r['mode'] for r in subset)), 'formats':dict(Counter(r['format'] for r in subset))}
    data['duplicate_label_conflicts'] = []
    for g in data['exact_rgb_duplicates']:
        labels = {s.split('/')[0] for s in g['paths'] if s.split('/')[0] in ['health','sick','tb']}
        if len(labels)>1:
            data['duplicate_label_conflicts'].append({'paths':g['paths'],'folder_labels':sorted(labels)})
    core = [r for r in rows if r['path'] in lists['TBX11K_trainval']]
    data['core_unique_rgb_images'] = len({r['rgb_sha256'] for r in core})
    data['exact_duplicate_statistics'] = {
        'groups':len(data['exact_rgb_duplicates']),
        'files_in_groups':sum(len(g['paths']) for g in data['exact_rgb_duplicates']),
        'redundant_files':sum(len(g['paths'])-1 for g in data['exact_rgb_duplicates']),
        'core_split_pair_groups':{a+'__'+b:sum(a in g['core_splits'] and b in g['core_splits'] for g in data['exact_rgb_duplicates'])
            for a,b in [('train','val'),('train','test'),('val','test')]}}
    data['core_unique_rgb_by_folder'] = {k:len({r['rgb_sha256'] for r in core if r['folder']==k}) for k in ['health','sick','tb']}
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    sources = sorted([root/'README.md', *root.glob('lists/*.txt'), *root.glob('annotations/xml/*.xml'),
                      *root.glob('annotations/json/*.json'), *root.glob('code/*')])
    provenance = {'note':'Hashes de metadatos originales. Los hashes de imagenes estan en image_inventory.csv.',
        'sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}}
    (out/'metadata_sha256.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for k in ['images','bytes','dimensions','modes','formats','by_folder_details','image_errors',
              'xml_image_category_sets','xml_difficult','xml_invalid_boxes','excluded_xml_objects',
              'coco_issues','tb_without_coco_boxes','duplicate_label_conflicts']:
        print(k, json.dumps(data[k],ensure_ascii=False))
    for k in ['exact_file_duplicates','exact_rgb_duplicates','dhash_equal_candidates']:
        print(k, 'groups',len(data[k]), 'core_cross_split',sum(len(g['core_splits'])>1 for g in data[k]),
              'all_cross_split',sum(len(g['combined_splits'])>1 for g in data[k]))
        print(json.dumps(data[k][:20],ensure_ascii=False))


if __name__=='__main__':
    main()

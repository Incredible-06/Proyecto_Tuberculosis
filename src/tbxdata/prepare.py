"""Genera manifiestos sin decodificar de nuevo el dataset auditado."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from .common import json_text, read_csv, sha256, write_csv, safe_path


class UnionFind:
    def __init__(self, keys):
        self.parent = {k:k for k in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a,b)] = min(a,b)


def allocate_groups(groups, config):
    """Estratifica por clase y asigna grupos indivisibles, no pacientes.

    Cuotas por restos mayores; grupos grandes primero. En cada paso se llena la
    mayor fracción pendiente de su cuota, sin excederla cuando hay espacio.
    """
    splits = list(config['split_ratios'])
    ratios = list(config['split_ratios'].values())
    if abs(sum(ratios)-1) > 1e-9 or any(r <= 0 for r in ratios):
        raise ValueError('Las proporciones deben ser positivas y sumar uno.')
    assignments, targets = {}, {}
    for cls in config['classes']:
        selected = [g for g in groups if g['class_name']==cls]
        n = sum(g['size'] for g in selected)
        raw = [n*r for r in ratios]
        quota = [math.floor(v) for v in raw]
        for i in sorted(range(len(splits)), key=lambda i:(-(raw[i]-quota[i]),i))[:n-sum(quota)]:
            quota[i] += 1
        if min(quota)==0:
            raise ValueError(f'No hay suficientes muestras para estratificar {cls}.')
        targets[cls] = dict(zip(splits,quota))
        def rank(g):
            token = f"{config['seed']}:{cls}:{g['group_id']}".encode()
            return (-g['size'], hashlib.sha256(token).hexdigest())
        counts = [0]*len(splits)
        for group in sorted(selected,key=rank):
            fitting = [i for i in range(len(splits)) if counts[i]+group['size'] <= quota[i]]
            choices = fitting or list(range(len(splits)))
            i = max(choices, key=lambda i:((quota[i]-counts[i])/quota[i],quota[i]-counts[i],-i))
            assignments[group['group_id']] = splits[i]
            counts[i] += group['size']
    return assignments, targets


def build_manifests(project_root, dataset_root, output=None):
    project, root = Path(project_root).resolve(), Path(dataset_root).resolve()
    output = Path(output).resolve() if output else project/'manifests'
    if output == root or root in output.parents:
        raise ValueError('Los manifiestos deben guardarse fuera del dataset original.')
    config_path = project/'configs/data.json'
    config = json.loads(config_path.read_text())
    inv_path = project/'reports/initial_audit/image_inventory.csv'
    inv = read_csv(inv_path)
    by_path = {r['path']:r for r in inv}
    audit = json.loads((project/'reports/initial_audit/audit_summary.json').read_text())
    if audit['image_errors'] or len(inv) != audit['images'] or len(by_path)!=len(inv):
        raise ValueError('La auditoría original no está completa o tiene errores.')
    # Verifica metadatos y tamaños; reutiliza los hashes originales, sin releer imágenes.
    provenance = json.loads((project/'reports/initial_audit/metadata_sha256.json').read_text())
    for relative, expected in provenance['sha256'].items():
        # El prefijo TBX11K identifica la raíz original, no una ruta absoluta.
        path = safe_path(root, relative.split('/',1)[1])
        if sha256(path) != expected:
            raise ValueError(f'Metadato cambiado desde la auditoría: {path}')
    for r in inv:
        p = safe_path(root,'imgs/'+r['path'])
        if not p.is_file() or p.stat().st_size != int(r['bytes']):
            raise ValueError(f'Archivo ausente o tamaño cambiado: {p}')
    review_path = project/'reports/near_duplicates/review_decisions.csv'
    decisions = read_csv(review_path)
    refined = read_csv(project/'reports/near_duplicates/refined_pairs.csv')
    decision_map = {r['pair_id']:r for r in decisions}
    refined_map = {r['pair_id']:r for r in refined}
    if len(decision_map) != len(decisions):
        raise ValueError('Decisiones visuales repetidas.')
    for r in refined:
        if int(r['suspicious']) and r['pair_id'] not in decision_map:
            raise ValueError(f"Falta revisión de {r['pair_id']}")
    uf = UnionFind(by_path)
    exact = defaultdict(list)
    for r in inv:
        exact[r['rgb_sha256']].append(r['path'])
    for paths in exact.values():
        for p in paths[1:]:uf.union(paths[0],p)
    for r in decisions:
        original_pair = refined_map.get(r['pair_id'])
        if original_pair is None or any(r[k]!=original_pair[k] for k in ['path_a','path_b','rgb_sha256_a','rgb_sha256_b']):
            raise ValueError('La decisión visual no corresponde al par refinado.')
        for suffix in ['a','b']:
            if by_path[r['path_'+suffix]]['rgb_sha256'] != r['rgb_sha256_'+suffix]:
                raise ValueError('La decisión visual no coincide con los hashes auditados.')
        if r['decision']=='confirmed_near_duplicate':
            uf.union(r['path_a'],r['path_b'])
        elif r['decision']!='distinct':
            raise ValueError('Resolver las decisiones inciertas antes de dividir.')
    components = defaultdict(list)
    for p in sorted(by_path):components[uf.find(p)].append(p)
    group_for = {}
    for paths in components.values():
        token = '\n'.join(sorted({by_path[p]['rgb_sha256'] for p in paths}))
        group_id = 'g_'+hashlib.sha256(token.encode()).hexdigest()
        for p in paths:group_for[p]=group_id
    core_paths = set((root/'lists/TBX11K_trainval.txt').read_text().splitlines())
    classes = {'health':'health','sick':'sick','tb':'TB'}
    core = [r for r in inv if r['path'] in core_paths]
    representatives = {}
    for r in sorted(core,key=lambda r:r['path']):representatives.setdefault(r['rgb_sha256'],r['path'])
    aliases = defaultdict(list)
    for r in core:aliases[r['rgb_sha256']].append('imgs/'+r['path'])
    grouped = defaultdict(list)
    for rgb,p in representatives.items():grouped[group_for[p]].append(p)
    split_groups = []
    for group_id,paths in sorted(grouped.items()):
        labels = {classes[p.split('/')[0]] for p in paths}
        if len(labels) != 1:
            raise ValueError(f'Conflicto de clases en grupo confirmado: {paths}. Revisar antes de entrenar.')
        split_groups.append({'group_id':group_id,'class_name':next(iter(labels)),'size':len(paths)})
    assignment, targets = allocate_groups(split_groups,config)
    coco_path = root/'annotations/json/TBX11K_trainval.json'
    coco = json.loads(coco_path.read_text())
    coco_images = {im['file_name']:im for im in coco['images']}
    coco_boxes = defaultdict(list)
    for box in coco['annotations']:coco_boxes[box['image_id']].append(box)
    cat_names = {c['id']:c['name'] for c in coco['categories']}
    records = []
    for rgb,p in sorted(representatives.items(),key=lambda kv:kv[1]):
        r,im = by_path[p],coco_images[p]
        boxes = coco_boxes[im['id']]
        cls = classes[p.split('/')[0]]
        gid = group_for[p]
        component = components[uf.find(p)]
        memberships = sorted({s for alias in aliases[rgb] for s in by_path[alias[5:]]['memberships'].split(';')})
        record = {'relative_path':'imgs/'+p,'image_id':Path(p).stem,
            'class_name':cls,'label':config['classes'][cls],'split':assignment[gid],
            'group_id':gid,'sha256':r['sha256'],'rgb_sha256':rgb,'bytes':r['bytes'],
            'width':r['width'],'height':r['height'],'mode':r['mode'],'source':'TBX11K_base',
            'official_memberships':r['memberships'],'alias_official_memberships':';'.join(memberships),
            'exact_aliases_json':json_text(sorted(aliases[rgb])),
            'coco_file':'annotations/json/TBX11K_trainval.json','coco_image_id':im['id'],
            'bbox_count':len(boxes),'tb_subtypes_json':json_text(sorted({cat_names[b['category_id']] for b in boxes})),
            'boxes_json':json_text(boxes),
            'annotation_status':('present' if boxes else 'missing_tb_boxes') if cls=='TB' else 'not_applicable_non_tb',
            'overlaps_official_test':int(any(q.startswith('test/') for q in component)),
            'overlaps_reserved_external':int(any(q.startswith('extra/') for q in component))}
        records.append(record)
    for split in config['split_ratios']:
        write_csv(output/f'{split}.csv',[r for r in records if r['split']==split])
    members = []
    for r in sorted(core,key=lambda r:r['path']):
        rep = representatives[r['rgb_sha256']]
        members.append({'relative_path':'imgs/'+r['path'],'representative_path':'imgs/'+rep,
            'class_name':classes[r['path'].split('/')[0]],'label':config['classes'][classes[r['path'].split('/')[0]]],
            'sha256':r['sha256'],'rgb_sha256':r['rgb_sha256'],'group_id':group_for[r['path']],
            'split':assignment[group_for[r['path']]],'official_memberships':r['memberships'],
            'is_representative':int(r['path']==rep)})
    write_csv(output/'all_labeled_members.csv',members)
    reserved = defaultdict(list)
    overlaps = []
    for r in inv:
        p = r['path']
        if p in core_paths:continue
        kind = 'official_test_unlabeled' if p.startswith('test/') else 'reserved_external'
        component = components[uf.find(p)]
        matches = ['imgs/'+q for q in component if q in core_paths]
        item = {'relative_path':'imgs/'+p,'class_name':'','label':'','split':kind,'group_id':group_for[p],
            'sha256':r['sha256'],'rgb_sha256':r['rgb_sha256'],'bytes':r['bytes'],'width':r['width'],
            'height':r['height'],'mode':r['mode'],'official_memberships':r['memberships'],
            'source':('official_mixed_unknown' if kind=='official_test_unlabeled' else ('mc_shenzhen' if 'mc+shenzhen' in p else 'da_db')),
            'overlaps_internal':int(bool(matches)), 'internal_overlap_paths_json':json_text(matches)}
        reserved[kind].append(item)
        if kind=='reserved_external' and len(component)>1:
            overlaps.append({'relative_path':'imgs/'+p,'group_id':group_for[p],
                             'matched_paths_json':json_text(['imgs/'+q for q in component if q!=p]),
                             'overlaps_internal':int(bool(matches)),
                             'overlaps_official_test':int(any(q.startswith('test/') for q in component))})
    for kind,items in reserved.items():write_csv(output/f'{kind}.csv',items)
    write_csv(output/'external_overlap_report.csv',overlaps,
              ['relative_path','group_id','matched_paths_json','overlaps_internal','overlaps_official_test'])
    stats = {s:dict(Counter(r['class_name'] for r in records if r['split']==s)) for s in config['split_ratios']}
    input_files = [config_path,inv_path,review_path,project/'reports/near_duplicates/refined_pairs.csv',
                   project/'reports/initial_audit/metadata_sha256.json',
                   project/'src/tbxdata/prepare.py',project/'src/tbxdata/common.py']
    meta = {'schema_version':1,'seed':config['seed'],'ratios':config['split_ratios'],
        'classes':config['classes'],'input_sha256':{p.relative_to(project).as_posix():sha256(p) for p in input_files},
        'original_coco_sha256':sha256(coco_path),'unique_labeled_images':len(records),
        'original_labeled_files':len(core),'duplicate_groups_internal':len(split_groups),
        'confirmed_near_pairs':sum(r['decision']=='confirmed_near_duplicate' for r in decisions),
        'targets':targets,'counts':stats,
        'algorithm':'largest_remainder_class_quotas; indivisible_content_groups; descending_size_seeded_sha256_order; proportional_deficit',
        'patient_independence_guaranteed':False,
        'output_sha256':{p.name:sha256(p) for p in sorted(output.glob('*.csv'))}}
    (output/'manifest_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path('.'))
    parser.add_argument('--dataset',type=Path,default=Path('TBX11K'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(build_manifests(args.project,args.dataset,args.output),indent=2))


if __name__=='__main__':main()

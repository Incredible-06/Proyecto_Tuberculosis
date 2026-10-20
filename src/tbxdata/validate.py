"""Comprobaciones de manifiestos; no ejecuta modelos ni analiza de nuevo imágenes."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .common import read_csv, safe_path, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifests(project_root, dataset_root=None, hash_samples_per_class=0, all_image_hashes=False):
    project = Path(project_root)
    root = Path(dataset_root) if dataset_root is not None else None
    config = json.loads((project/'configs/data.json').read_text())
    meta = json.loads((project/'manifests/manifest_metadata.json').read_text())
    for name,expected in meta['output_sha256'].items():
        require(sha256(project/'manifests'/name)==expected,f'Manifiesto modificado: {name}')
    for name,expected in meta['input_sha256'].items():
        require(sha256(project/name)==expected,f'Entrada de preparación modificada: {name}')
    require(meta['classes']==config['classes'] and meta['seed']==config['seed'],'Configuración distinta del split.')
    parts = {s:read_csv(project/'manifests'/f'{s}.csv') for s in ['train','validation','test']}
    records = [r for rows in parts.values() for r in rows]
    require(len(records)==meta['unique_labeled_images'],'Conteo distinto al esperado.')
    for split,rows in parts.items():
        require(rows and all(r['split']==split for r in rows),f'Partición inválida: {split}')
        counts=Counter(r['class_name'] for r in rows)
        require(dict(counts)==meta['counts'][split],f'Conteos cambiados: {split}')
        for r in rows:
            require(r['class_name'] in config['classes'] and int(r['label'])==config['classes'][r['class_name']], 'Etiqueta no válida.')
            folder=r['relative_path'].split('/')
            require(len(folder)==3 and folder[:2] in [['imgs','health'],['imgs','sick'],['imgs','tb']], 'Imagen externa/test en manifiesto supervisado.')
            require({'health':'health','sick':'sick','tb':'TB'}[folder[1]]==r['class_name'],'Clase/ruta inconsistente.')
            require((int(r['width']),int(r['height']),r['mode'])==(512,512,'RGB'),'Dimensiones/modo inesperados.')
            boxes=json.loads(r['boxes_json'])
            require(len(boxes)==int(r['bbox_count']),'Número de cajas incorrecto.')
            if r['annotation_status']=='missing_tb_boxes':
                require(r['relative_path']=='imgs/tb/tb1199.png' and not boxes,'Incidencia TB no documentada.')
            for box in boxes:
                x,y,w,h=box['bbox']
                require(min(x,y)>=0 and min(w,h)>0 and x+w<=512.001 and y+h<=512.001,'Caja fuera de límites.')
    overlap_checks={}
    for field in ['relative_path','sha256','rgb_sha256','group_id']:
        sets={s:{r[field] for r in rows} for s,rows in parts.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            n=len(sets[a]&sets[b]);overlap_checks[f'{field}:{a}:{b}']=n
            require(n==0,f'Fuga de datos por {field}: {a}/{b}')
    require(len({r['relative_path'] for r in records})==len(records),'Rutas supervisadas repetidas.')
    require(len({r['rgb_sha256'] for r in records})==len(records),'Copias exactas no deduplicadas.')
    members=read_csv(project/'manifests/all_labeled_members.csv')
    paths={r['relative_path']:r for r in records}
    require(len(members)==meta['original_labeled_files'] and len({r['relative_path'] for r in members})==len(members),'Cobertura de archivos etiquetados incorrecta.')
    member_by_path={r['relative_path']:r for r in members}
    for m in members:
        rep=paths.get(m['representative_path'])
        require(rep is not None,'Representante ausente.')
        require(all(m[k]==rep[k] for k in ['split','group_id','rgb_sha256','label','class_name']),'Alias con asignación inconsistente.')
        require(m['relative_path'] in json.loads(rep['exact_aliases_json']),'Alias sin trazabilidad.')
    for pair in read_csv(project/'reports/near_duplicates/review_decisions.csv'):
        if pair['decision']=='confirmed_near_duplicate':
            a,b=(member_by_path.get('imgs/'+pair['path_'+k]) for k in ['a','b'])
            if a and b:
                require(a['split']==b['split'] and a['group_id']==b['group_id'],'Duplicado cercano cruza particiones.')
    reserved={k:read_csv(project/'manifests'/f'{k}.csv') for k in ['official_test_unlabeled','reserved_external']}
    for kind,rows in reserved.items():
        require(all(r['label']=='' and r['class_name']=='' and r['split']==kind for r in rows),'Etiqueta inferida en imágenes reservadas.')
        prefix='imgs/test/' if kind=='official_test_unlabeled' else 'imgs/extra/'
        require(all(r['relative_path'].startswith(prefix) for r in rows),'Fuente reservada incorrecta.')
    inventory=read_csv(project/'reports/initial_audit/image_inventory.csv')
    require({m['relative_path'] for m in members} | {r['relative_path'] for rows in reserved.values() for r in rows} == {'imgs/'+r['path'] for r in inventory},'El inventario no está cubierto completamente.')
    verified=[]
    full_hash_count=0
    if root is not None:
        require(sha256(root/'annotations/json/TBX11K_trainval.json')==meta['original_coco_sha256'],'COCO original modificado.')
        coco=json.loads((root/'annotations/json/TBX11K_trainval.json').read_text())
        names={im['id']:im['file_name'] for im in coco['images']}
        boxes_by_id=defaultdict(list)
        for box in coco['annotations']:boxes_by_id[box['image_id']].append(box)
        for r in records:
            iid=int(r['coco_image_id'])
            require(names[iid]==r['relative_path'][5:] and boxes_by_id[iid]==json.loads(r['boxes_json']),'Cajas COCO alteradas en manifiesto.')
        for r in inventory:
            p=safe_path(root,'imgs/'+r['path'])
            require(p.is_file() and p.stat().st_size==int(r['bytes']),f'Archivo ausente/cambiado: {p}')
            if all_image_hashes:
                require(sha256(p)==r['sha256'],f'Hash cambiado desde la auditoría: {p}')
                full_hash_count+=1
        require({m['relative_path'][5:] for m in members}==set((root/'lists/TBX11K_trainval.txt').read_text().splitlines()),'Cobertura distinta de la lista base.')
        if hash_samples_per_class and not all_image_hashes:
            for split,rows in parts.items():
                for cls in config['classes']:
                    selected=sorted((r for r in rows if r['class_name']==cls),key=lambda r:hashlib.sha256((str(config['seed'])+r['relative_path']).encode()).hexdigest())[:hash_samples_per_class]
                    for r in selected:
                        require(sha256(safe_path(root,r['relative_path']))==r['sha256'],'Hash de muestra distinto a la auditoría.')
                        verified.append(r['relative_path'])
    return {'status':'passed','counts':meta['counts'],'overlap_checks':overlap_checks,
        'supervised_images':len(records),'supervised_groups':meta['duplicate_groups_internal'],
        'official_test_unlabeled':len(reserved['official_test_unlabeled']),
        'reserved_external':len(reserved['reserved_external']),
        'hash_verified_samples':verified,
        'all_image_hashes_verified':full_hash_count,
        'hash_scope':('SHA256 de todos los archivos contra la auditoría reutilizada.' if all_image_hashes else 'SHA256 de muestras; tamaños/existencia para todos. No es una reauditoría completa de píxeles.'),
        'patient_independence_guaranteed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path)
    p.add_argument('--hash-samples',type=int,default=0)
    p.add_argument('--all-image-hashes',action='store_true',help='Opcional para verificar la transferencia completa en Colab.')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report=validate_manifests(args.project,args.dataset,args.hash_samples,args.all_image_hashes)
    text=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()

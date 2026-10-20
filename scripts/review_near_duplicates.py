"""Refina SOLO los candidatos de la auditoría existente. No entrena modelos.

pHash DCT de 256 bits + SSIM local 7x7 + correlación y error de píxeles.
Las decisiones humanas se registran por separado; este script no confirma pares.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
import itertools
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def phash(im):
    a = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    n = np.arange(32, dtype=np.float32)
    basis = np.cos(np.pi / 32 * (n[None, :] + 0.5) * n[:, None])
    dct = (basis @ a @ basis.T)[:16, :16].reshape(-1)
    bits = dct > np.median(dct[1:])
    bits[0] = False  # excluye el término DC del contenido
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def local_mean(a, window=7):
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[window:, window:] - integral[:-window, window:]
            - integral[window:, :-window] + integral[:-window, :-window]) / window**2


def similarity(a, b, window=7):
    a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
    ma, mb = local_mean(a, window), local_mean(b, window)
    va = np.maximum(0, local_mean(a*a, window) - ma*ma)
    vb = np.maximum(0, local_mean(b*b, window) - mb*mb)
    cov = local_mean(a*b, window) - ma*mb
    ssim = np.mean(((2*ma*mb + 0.01**2)*(2*cov + 0.03**2)) /
                   ((ma*ma + mb*mb + 0.01**2)*(va + vb + 0.03**2)))
    ac, bc = a-a.mean(), b-b.mean()
    den = np.linalg.norm(ac)*np.linalg.norm(bc)
    ncc = float(np.sum(ac*bc)/den) if den else float(np.array_equal(a,b))
    return float(ssim), ncc, float(np.abs(a-b).mean()), float(np.sqrt(np.mean((a-b)**2)))


def contact_sheet(pairs, root, output, title):
    if not pairs:
        return
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    for page in range((len(pairs)+7)//8):
        subset = pairs[page*8:page*8+8]
        canvas = Image.new('RGB', (1100, 50+len(subset)*190), '#151b22')
        draw = ImageDraw.Draw(canvas)
        draw.text((12,10), title+' | page '+str(page+1), fill='white', font=font)
        for i, pair in enumerate(subset):
            y = 45+i*190
            for j, field in enumerate(['path_a','path_b']):
                with Image.open(root/'imgs'/pair[field]) as im:
                    canvas.paste(im.convert('RGB').resize((170,170)), (8+j*180,y))
            text = f"{pair['pair_id']}\n{pair['path_a']}\n{pair['path_b']}\n"
            text += f"pHash={pair['phash_distance']} | SSIM={float(pair['ssim128']):.5f}\nNCC={float(pair['ncc128']):.5f} | MAE={float(pair['mae128']):.5f}"
            draw.multiline_text((375,y+4), text, fill='white', font=font, spacing=7)
        canvas.save(output/f'{title}_{page+1:02d}.jpg',quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('TBX11K'))
    args = parser.parse_args()
    out = Path('reports/near_duplicates')
    out.mkdir(parents=True,exist_ok=True)
    config = json.loads(Path('configs/data.json').read_text())['near_duplicate_review']
    audit = json.loads(Path('reports/initial_audit/audit_summary.json').read_text())
    inventory = {r['path']:r for r in csv.DictReader(Path('reports/initial_audit/image_inventory.csv').open())}
    groups = audit['dhash_equal_candidates']
    paths = sorted({p for g in groups for p in g['paths']})
    features = {}
    for i,path in enumerate(paths,1):
        with Image.open(args.root/'imgs'/path) as im:
            gray = im.convert('L')
            features[path] = (phash(gray), np.asarray(gray.resize((config['review_size'],)*2,Image.Resampling.LANCZOS)))
        if i % 1000 == 0:
            print('Features',i,'/',len(paths),flush=True)
        time.sleep(0.005)
    rows, rejects, exact = [],0,0
    rejected_pairs = []
    for group in groups:
        for a,b in itertools.combinations(sorted(group['paths']),2):
            if inventory[a]['rgb_sha256'] == inventory[b]['rgb_sha256']:
                exact += 1
                continue
            pa,ia = features[a]
            pb,ib = features[b]
            dist = (pa^pb).bit_count()
            if dist > config['phash_max_distance']:
                rejects += 1
                rejected_pairs.append((dist,a,b))
                continue
            ssim,ncc,mae,rmse = similarity(ia,ib,config['ssim_window'])
            suspicious = (dist <= config['suspicious_phash_max_distance'] and
                          ssim >= config['suspicious_ssim_min'] and ncc >= config['suspicious_ncc_min'])
            row = {'pair_id':f'near_{len(rows)+1:05d}', 'path_a':a,'path_b':b,
                   'rgb_sha256_a':inventory[a]['rgb_sha256'],'rgb_sha256_b':inventory[b]['rgb_sha256'],
                   'phash_distance':dist,'ssim128':ssim,'ncc128':ncc,'mae128':mae,'rmse128':rmse,
                   'suspicious':int(suspicious)}
            rows.append(row)
    rows.sort(key=lambda r:(-r['ssim128'],r['phash_distance'],r['path_a'],r['path_b']))
    for i,row in enumerate(rows,1):row['pair_id']=f'near_{i:05d}'
    fields = ['pair_id','path_a','path_b','rgb_sha256_a','rgb_sha256_b','phash_distance','ssim128','ncc128','mae128','rmse128','suspicious']
    with (out/'refined_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    feature_records = [{'path':p,'phash256':f'{features[p][0]:064x}'} for p in paths]
    with (out/'phash_features.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','phash256']);w.writeheader();w.writerows(feature_records)
    suspicious = [r for r in rows if r['suspicious']]
    contact_sheet(suspicious,args.root,out,'suspicious')
    # Controles de mayor parecido fuera del umbral; no se confirman por similitud.
    contact_sheet([r for r in rows if not r['suspicious']][:16],args.root,out,'controls')
    controls=[]
    for i,(dist,a,b) in enumerate(sorted(rejected_pairs)[:8],1):
        ssim,ncc,mae,_=similarity(features[a][1],features[b][1],config['ssim_window'])
        controls.append({'pair_id':f'control_{i:02d}','path_a':a,'path_b':b,
            'phash_distance':dist,'ssim128':ssim,'ncc128':ncc,'mae128':mae,
            'decision':'rejected_metric_control'})
    if controls:
        with (out/'rejected_controls.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)
        contact_sheet(controls,args.root,out,'rejected_controls')
    summary={'candidate_groups':len(groups),'images_read':len(paths),
             'candidate_pairs':sum(len(g['paths'])*(len(g['paths'])-1)//2 for g in groups),
             'exact_pairs_already_known':exact,'rejected_by_phash':rejects,
             'pairs_with_structural_metrics':len(rows),'suspicious_pairs':len(suspicious),
             'method':config,'scope':'Solo grupos de dHash igual existentes; no busqueda exhaustiva ni identificacion de pacientes.'}
    (out/'method_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    for row in rows[:30]:print(row,flush=True)


if __name__=='__main__':main()

"""Comprobación a resolución original de los 44 pares refinados; sin barrido nuevo."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from review_near_duplicates import similarity


def main():
    out=Path('reports/near_duplicates')
    rows=list(csv.DictReader((out/'refined_pairs.csv').open()))
    details=[]
    for r in rows:
        arrays=[]
        for field in ['path_a','path_b']:
            with Image.open(Path('TBX11K/imgs')/r[field]) as im:arrays.append(np.asarray(im.convert('L')))
        a,b=arrays
        ssim,ncc,mae,rmse=similarity(a,b)
        diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
        details.append({'pair_id':r['pair_id'],'ssim512':ssim,'ncc512':ncc,'mae512':mae,
                        'rmse512':rmse,'fraction_diff_gt5':float((diff>5).mean()),
                        'fraction_identical':float((diff==0).mean())})
    with (out/'full_resolution_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    selected=rows[-3:]
    canvas=Image.new('RGB',(1536,3*545),'#151b22')
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(selected):
        ims=[]
        for j,field in enumerate(['path_a','path_b']):
            with Image.open(Path('TBX11K/imgs')/r[field]) as im:ims.append(im.convert('RGB'))
            canvas.paste(ims[-1],(j*512,i*545+25))
            draw.text((j*512+4,i*545+4),r[field],fill='white')
        a,b=[np.asarray(im,dtype=np.int16) for im in ims]
        delta=np.clip(np.abs(a-b)*4,0,255).astype(np.uint8)
        canvas.paste(Image.fromarray(delta),(1024,i*545+25))
        draw.text((1028,i*545+4),r['pair_id']+' | absolute difference x4',fill='white')
    canvas.save(out/'detail_lowest_similarity.png')
    print('Full-res SSIM range',min(d['ssim512'] for d in details),max(d['ssim512'] for d in details))
    print('Max fraction difference >5/255',max(d['fraction_diff_gt5'] for d in details))


if __name__=='__main__':main()

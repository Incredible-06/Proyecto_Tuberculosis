"""Figuras de auditoría con Pillow; no requiere Matplotlib ni modelos."""
import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .common import read_csv, safe_path
from .dataset import RadiographTransform, load_config


def font(size=16):
    for path in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def distribution(project, output):
    project,output=Path(project),Path(output)
    config=load_config(project/'configs/data.json')
    splits=['train','validation','test']
    counts={s:Counter(r['class_name'] for r in read_csv(project/'manifests'/f'{s}.csv')) for s in splits}
    canvas=Image.new('RGB',(1080,530),'white');draw=ImageDraw.Draw(canvas)
    draw.text((25,15),'TBX11K: contenidos unicos por clase y particion',fill='#182536',font=font(23))
    colors=['#276592','#dc8b25','#368375']
    max_count=max(n for c in counts.values() for n in c.values())
    for ci,cls in enumerate(config['classes']):
        left=60+ci*350
        for si,split in enumerate(splits):
            x=left+si*80;y=430-counts[split][cls]/max_count*320
            draw.rectangle((x,y,x+60,430),fill=colors[si])
            draw.text((x,y-24),str(counts[split][cls]),fill='#182536',font=font())
            draw.text((x,440),split,fill='#182536',font=font(12))
        draw.text((left+65,475),cls,fill='#182536',font=font(21))
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return canvas


def samples(project, dataset_root, output):
    project,root=Path(project),Path(dataset_root)
    rows=read_csv(project/'manifests/train.csv')
    selected=[]
    for cls in ['health','sick','TB']:
        subset=[r for r in rows if r['class_name']==cls]
        if cls=='TB':
            # Incluye un ejemplo mixto si existe en train.
            subset.sort(key=lambda r:(-len(json.loads(r['tb_subtypes_json'])),r['relative_path']))
        selected.extend(subset[:2])
    canvas=Image.new('RGB',(1020,760),'#151b22');draw=ImageDraw.Draw(canvas)
    colors={1:'#ff765c',2:'#48d6f2',3:'#e7d35d'}
    for i,r in enumerate(selected):
        x=(i%3)*340;y=(i//3)*380
        with Image.open(safe_path(root,r['relative_path'])) as im:
            canvas.paste(im.convert('RGB').resize((330,330)),(x+5,y+5))
        for b in json.loads(r['boxes_json']):
            a,c,w,h=b['bbox'];scale=330/512
            draw.rectangle((x+5+a*scale,y+5+c*scale,x+5+(a+w)*scale,y+5+(c+h)*scale),outline=colors[b['category_id']],width=2)
        draw.text((x+5,y+340),r['relative_path'],fill='white',font=font(13))
        draw.text((x+5,y+360),r['class_name']+f" | boxes={r['bbox_count']}",fill='#b9d0dd',font=font(13))
    Path(output).parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return canvas


def transformations(project, dataset_root, output):
    project,root=Path(project),Path(dataset_root)
    config=load_config(project/'configs/data.json')
    rows=read_csv(project/'manifests/train.csv')
    train=RadiographTransform(config,training=True)
    evaluation=RadiographTransform(config,training=False)
    canvas=Image.new('RGB',(1280,3*355),'#151b22');draw=ImageDraw.Draw(canvas)
    for ci,cls in enumerate(config['classes']):
        r=next(r for r in rows if r['class_name']==cls)
        with Image.open(safe_path(root,r['relative_path'])) as im:
            views=[evaluation.image(im)]+[train.image(im,random.Random(config['seed']+ci*10+j)) for j in range(3)]
        for j,view in enumerate(views):
            x=j*320;y=ci*355
            canvas.paste(view.resize((310,310)),(x+5,y+25))
            label='determinista' if j==0 else f'train ejemplo {j}'
            draw.text((x+5,y+5),cls+' | '+label,fill='white',font=font(14))
            draw.text((x+5,y+338),Path(r['relative_path']).name,fill='#b9d0dd',font=font(12))
    Path(output).parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return canvas


def generate_figures(project, dataset_root, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    distribution(project,output/'class_distribution.png')
    samples(project,dataset_root,output/'training_samples.png')
    transformations(project,dataset_root,output/'training_transformations.png')
    return [str(output/name) for name in ['class_distribution.png','training_samples.png','training_transformations.png']]

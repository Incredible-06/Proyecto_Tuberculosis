"""Revisión de evidencia descargada y formato del informe; sin inferencia."""
import csv
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'reports/phase5_review/drive_evidence'
def read(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):return json.loads((E/name).read_text(encoding='utf-8'))


def main():
    notebook=load('06_fase5_ejecutado.ipynb')
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==11 and [c['execution_count'] for c in cells]==list(range(1,12))
    assert not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[]))
    status=load('preparation_status.json')
    assert status['candidate']=='hybrid_cross_attention'
    assert not status['test_evaluated'] and not status['external_evaluated'] and not status['decisions_frozen']
    assert not status['detector_diagnostics_executed']
    assert all(r['passed'] for r in load('load_smoke.json'))
    assert load('transfer_integrity.json')['all_image_hashes_verified']==12278
    p=read(E/'candidate_validation_predictions.csv')
    cm=[[0]*3 for _ in range(3)]
    for row in p:
        label=int(row['label']);score=[float(row['p_'+name]) for name in ['health','sick','TB']]
        assert math.isclose(sum(score),1,abs_tol=1e-5)
        prediction=max(range(3),key=lambda i:score[i]);assert prediction==int(row['prediction'])
        cm[label][prediction]+=1
    assert len(p)==1241 and cm==[[568,2,0],[8,539,4],[6,7,107]]
    f1=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in range(3))/3
    assert math.isclose(f1,.9644699190877536,abs_tol=1e-12)
    thresholds=read(E/'classifier_TB_thresholds_validation.csv')
    assert len(thresholds)==len({float(r['p_TB']) for r in p}|{0.,1.})
    for r in thresholds:
        t=float(r['threshold'])
        tp=sum(int(v['label'])==2 and float(v['p_TB'])>=t for v in p)
        fp=sum(int(v['label'])!=2 and float(v['p_TB'])>=t for v in p)
        assert [int(r[k]) for k in ['tp','fp','fn','tn']]==[tp,fp,120-tp,1121-fp]
        assert math.isclose(float(r['sensitivity']),tp/120,abs_tol=1e-12)
        assert math.isclose(float(r['specificity']),(1121-fp)/1121,abs_tol=1e-12)
    manifest=load('figures_manifest.json');assert len(manifest['figures'])==12
    figures=['classifier_comparison.png']+[prefix+'_'+kind+'.png'
        for prefix in ['convnext_tiny','vmamba_tiny','hybrid_cross_attention'] for kind in ['training','confusion']]
    for name in figures:
        assert digest(E/name)==manifest['output_sha256'][name]
    for original,h in manifest['source_sha256'].items():
        name=Path(original).name
        if name=='validation.csv': path=ROOT/'manifests/validation.csv'
        elif 'convnext_tiny' in original:path=ROOT/'reports/phase5_readiness/drive_evidence'/('convnext_'+name)
        elif 'vmamba_tiny' in original:path=ROOT/'reports/phase5_readiness/drive_evidence'/('vmamba_'+name)
        elif 'hybrid_cross_attention' in original:path=ROOT/'reports/phase5_readiness/drive_evidence'/('hybrid_'+name)
        elif name=='results.csv':path=ROOT/'reports/phase5_readiness/drive_evidence/detector_results.csv'
        else:path=ROOT/'reports/phase5_readiness/drive_evidence/detector_detector_summary.json'
        assert digest(path)==h
    docx=ROOT/'docs/Informe_Avance_Fase_5.docx'
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    W='{'+ns['w']+'}'
    with ZipFile(docx) as z:
        for member in ['word/document.xml','word/styles.xml','word/footer1.xml']:
            tree=ET.fromstring(z.read(member))
            if member=='word/styles.xml':
                used={'Normal','Title','Heading1','Footer'}
                selected=ET.Element('usedStyles')
                for style in tree.findall('w:style',ns):
                    if style.get(W+'styleId') in used:selected.append(style)
                tree=selected
            for color in tree.findall('.//w:color',ns):assert color.get(W+'val')=='000000'
            for size in tree.findall('.//w:sz',ns):assert size.get(W+'val')=='24'
            for font in tree.findall('.//w:rFonts',ns):
                for key in ['ascii','hAnsi','eastAsia','cs']:
                    if font.get(W+key):assert font.get(W+key)=='Times New Roman'
        media=[name for name in z.namelist() if name.startswith('word/media/')]
        media_hashes={hashlib.sha256(z.read(name)).hexdigest() for name in media}
        assert len(media)==7
        assert media_hashes=={manifest['output_sha256'][name] for name in figures}
    report=dict(status='passed',run_id=load('session.json')['run_id'],
        executed_code_cells=len(cells),notebook_errors=0,validation_images=len(p),
        confusion_matrix=cm,macro_f1=f1,threshold_points_verified=len(thresholds),
        figures_exported=12,downloaded_figures_sha256_verified=7,all_figure_sources_match_local_review=True,
        embedded_figures_match_original_exports=True,color_figures_used=figures,
        classifier_load_smokes_passed=3,full_validation_inference_reexecuted=False,
        detector_diagnostics_executed=False,decisions_frozen=False,test_evaluated=False,external_evaluated=False,
        docx=str(docx),docx_sha256=digest(docx),font='Times New Roman',font_size_pt=12,text_color='000000',
        pages_rendered=5,cover_page=False,
        evidence_sha256={p.name:digest(p) for p in sorted(E.iterdir()) if p.is_file()},
        sources=[dict(title='Notebook 06 ejecutado',url='https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD'),
                 dict(title='Resultados de fase 5',url='https://drive.google.com/drive/folders/17JBPtxHx5jL0SEnPY-T56MEt6YM7URgW')])
    target=ROOT/'reports/phase5_review/review.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'evidence_sha256','sources'}},indent=2))


if __name__=='__main__':main()

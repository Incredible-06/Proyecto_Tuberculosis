"""Informe breve desde evidencia ejecutada; no modifica notebooks ni modelos."""
import csv
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'reports/phase5_review/drive_evidence'
VIT_EVIDENCE = ROOT/'reports/phase5_review/vit_evidence'
HYBRID_VIT_EVIDENCE = ROOT/'reports/phase5_review/hybrid_vit_evidence'
ASSETS = ROOT/'reports/phase5_review/report_assets'
TARGET = ROOT/'docs/Informe_Avance_Fase_5.docx'
BLACK = RGBColor(0,0,0)


def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def graph_source():
    """SVG en negro, Times New Roman 12 pt al insertar a 16 cm de ancho."""
    ASSETS.mkdir(parents=True,exist_ok=True)
    width,height = 604.7244,240
    left,right,top,bottom = 62,584,14,157
    def x(epoch): return left+(epoch-1)/11*(right-left)
    def y(value): return bottom-(value-.7)/.3*(bottom-top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Times New Roman" font-size="16" fill="black">']
    for value in [.7,.8,.9,1.]:
        yy=y(value)
        parts.append(f'<path d="M {left},{yy} H {right}" fill="none" stroke="#D9D9D9"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<path d="M {left},{top} V {bottom} H {right}" fill="none" stroke="black"/>')
    for epoch in range(1,13):
        xx=x(epoch)
        parts.append(f'<path d="M {xx},{bottom} v 4" stroke="black"/>')
        parts.append(f'<text x="{xx}" y="{bottom+23}" text-anchor="middle">{epoch}</text>')
    parts.append('<text x="16" y="95" text-anchor="middle" transform="rotate(-90 16 95)">Macro F1</text>')
    parts.append(f'<text x="{(left+right)/2}" y="{bottom+47}" text-anchor="middle">Época de entrenamiento</text>')
    legend_x=[69,253,429]
    for i,(short,label,dash) in enumerate([('convnext','ConvNeXt',''),('vmamba','VMamba','8 4'),('hybrid','Híbrido','2 4')]):
        rows=read(ROOT/f'reports/phase5_readiness/drive_evidence/{short}_history.csv')
        values=[float(r['validation_macro_f1']) for r in rows]
        if min(values)<.7 or max(values)>1: raise ValueError('Rango de figura insuficiente.')
        points=' '.join(f"{x(int(r['epoch']))},{y(float(r['validation_macro_f1']))}" for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="{dash}"/>')
        for r in rows:
            parts.append(f'<circle cx="{x(int(r["epoch"]))}" cy="{y(float(r["validation_macro_f1"]))}" r="2.7" fill="black"/>')
        xx=legend_x[i]
        parts.append(f'<path d="M {xx},215 h 34" stroke="black" stroke-width="2" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{xx+42}" y="220">{escape(label)}</text>')
    parts.append('</g></svg>')
    (ASSETS/'training_comparison.svg').write_text('\n'.join(parts),encoding='utf-8')
    print('SVG de entrenamiento preparado.')


def font(run,bold=None):
    run.font.name='Times New Roman'; run.font.size=Pt(12); run.font.color.rgb=BLACK
    fonts=run._element.get_or_add_rPr().get_or_add_rFonts()
    for key in ['ascii','hAnsi','eastAsia','cs']: fonts.set(qn('w:'+key),'Times New Roman')
    if bold is not None: run.bold=bold
    return run


def main():
    status=json.loads((EVIDENCE/'preparation_status.json').read_text(encoding='utf-8'))
    vit_payload=json.loads((VIT_EVIDENCE/'vit_report_payload.json').read_text(encoding='utf-8'))
    hyb_vit_payload=json.loads((HYBRID_VIT_EVIDENCE/'hybrid_vit_report_payload.json').read_text(encoding='utf-8'))
    assert not status['test_evaluated'] and not status['decisions_frozen']
    assert not vit_payload['test_evaluated'] and not hyb_vit_payload['test_evaluated']
    assert status['detector_diagnostics_executed'] is False
    comparison=read(HYBRID_VIT_EVIDENCE/'validation_comparison_5models.csv')
    assert len(comparison)==5
    doc=Document()
    section=doc.sections[0]
    section.page_width=Cm(21); section.page_height=Cm(29.7)
    section.top_margin=Cm(2.3); section.bottom_margin=Cm(2.2)
    section.left_margin=Cm(2.5); section.right_margin=Cm(2.5)
    section.footer_distance=Cm(1)
    for style in doc.styles:
        if style.type in [1,2]:
            style.font.name='Times New Roman'; style.font.size=Pt(12); style.font.color.rgb=BLACK
            style.font.underline=False
            rpr=style.element.get_or_add_rPr()
            for color in rpr.findall(qn('w:color')):
                for attr in list(color.attrib):
                    if attr != qn('w:val'): del color.attrib[attr]
                color.set(qn('w:val'),'000000')
            for border in list(style.element.iter(qn('w:pBdr'))): border.getparent().remove(border)
    normal=doc.styles['Normal'].paragraph_format
    normal.line_spacing=1.5; normal.space_after=Pt(6)
    for name in ['Title','Heading 1','Heading 2']:
        style=doc.styles[name]; style.font.bold=True
        style.paragraph_format.space_before=Pt(8); style.paragraph_format.space_after=Pt(5)
        style.paragraph_format.line_spacing=1.15
        style.paragraph_format.keep_with_next=True
    doc.styles['Title'].paragraph_format.space_before=Pt(0)
    doc.styles['Title'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    markdown=[]

    def p(text,style=None):
        par=doc.add_paragraph(style=style)
        par.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY if style is None else WD_ALIGN_PARAGRAPH.LEFT
        font(par.add_run(text))
        par.paragraph_format.widow_control=True
        markdown.append(text+'\n')
        return par

    def h(text,page=False):
        par=p(text,'Heading 1'); par.paragraph_format.page_break_before=page
        markdown[-1]='## '+text+'\n'
        return par

    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
        for i,width in enumerate(widths): t.columns[i].width=Cm(width)
        for i,text in enumerate(headers): t.rows[0].cells[i].text=text
        for row in rows:
            cells=t.add_row().cells
            for i,text in enumerate(row):cells[i].text=str(text)
        for ri,row in enumerate(t.rows):
            props=row._tr.get_or_add_trPr();props.append(OxmlElement('w:cantSplit'))
            if ri==0: props.append(OxmlElement('w:tblHeader'))
            for ci,cell in enumerate(row.cells):
                cell.width=Cm(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr=cell._tc.get_or_add_tcPr()
                borders=OxmlElement('w:tcBorders')
                for side in ['top','bottom','left','right']:
                    edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single')
                    edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
                pr.append(borders)
                margins=OxmlElement('w:tcMar')
                for side in ['top','bottom','left','right']:
                    margin=OxmlElement('w:'+side);margin.set(qn('w:w'),'80');margin.set(qn('w:type'),'dxa');margins.append(margin)
                pr.append(margins)
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F2F2F2');pr.append(shade)
                for par in cell.paragraphs:
                    par.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                    par.paragraph_format.line_spacing=1.0
                    par.paragraph_format.space_before=Pt(2);par.paragraph_format.space_after=Pt(2)
                    for run in par.runs:font(run,bold=ri==0)
        markdown.append('| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+
                        '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
        doc.add_paragraph().paragraph_format.space_after=Pt(0)

    def link(label,url):
        par=doc.add_paragraph();par.paragraph_format.line_spacing=1.15
        par.paragraph_format.space_after=Pt(3)
        relation=par.part.relate_to(url,RT.HYPERLINK,is_external=True)
        element=OxmlElement('w:hyperlink');element.set(qn('r:id'),relation)
        run=OxmlElement('w:r');props=OxmlElement('w:rPr')
        for tag,attrs in [('rFonts',{'ascii':'Times New Roman','hAnsi':'Times New Roman'}),
                          ('color',{'val':'000000'}),('sz',{'val':'24'}),('u',{'val':'none'})]:
            item=OxmlElement('w:'+tag)
            for key,value in attrs.items():item.set(qn('w:'+key),value)
            props.append(item)
        run.append(props);text=OxmlElement('w:t');text.text=label;run.append(text);element.append(run);par._p.append(element)
        markdown.append('['+label+']('+url+')\n')

    figure_number=0
    def figure(name,caption,folder=EVIDENCE):
        nonlocal figure_number
        figure_number+=1
        path=folder/name
        par=doc.add_paragraph();par.alignment=WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.line_spacing=1;par.paragraph_format.space_after=Pt(2)
        par.paragraph_format.keep_with_next=True
        par.add_run().add_picture(str(path),width=Cm(16))
        cap=p(f'Figura {figure_number}. '+caption)
        cap.paragraph_format.line_spacing=1.0;cap.paragraph_format.space_after=Pt(6)
        markdown.append('!['+caption+']('+str(path)+')\n')

    title=doc.add_paragraph(style='Title');font(title.add_run('Resultados de validación y avance de la fase 5'),True)
    markdown.append('# Resultados de validación y avance de la fase 5\n')
    p('La comparación controlada incorpora ahora cinco clasificadores: ConvNeXt Tiny, VMamba Tiny, el híbrido original (ConvNeXt + VMamba), ViT-B/16 y el nuevo híbrido ConvNeXt + ViT-B/16. El híbrido original mantiene el primer lugar por el criterio predefinido de Macro F1 (0,9645 frente a 0,9644 de ViT-B/16 y 0,9630 del híbrido ConvNeXt + ViT-B/16). ViT-B/16 obtuvo la mayor sensibilidad TB por argmax (110 de 120 imágenes, 91,67 %), mientras el nuevo híbrido ConvNeXt + ViT-B/16 empató la mayor exactitud global (97,82 %) y alcanzó la mayor precisión TB (97,22 %), ROC AUC (0,9975) y Average Precision (0,9806). La fase 5 sigue en desarrollo; aún faltan decisiones y pruebas para cerrar la evaluación.')
    p('La revisión integra el notebook 06 ejecutado en Colab (preparación de fase 5), el notebook 04b (ViT-B/16) y el notebook 04c (híbrido ConvNeXt + ViT-B/16), ambos ejecutados en GPU NVIDIA L4 bajo el mismo protocolo controlado (semilla 20261019, 512 píxeles y hasta 12 épocas). La validación contiene 1.241 radiografías: 570 sin enfermedad, 551 con otras enfermedades y 120 con tuberculosis. Las figuras a color son las imágenes originales exportadas por las ejecuciones. El conjunto de prueba permanece reservado.')
    h('Comparación de los clasificadores')
    p('La exactitud indica cuántas imágenes se clasificaron correctamente. Macro F1 resume el desempeño dando el mismo peso a las tres clases. Para TB también se muestra cuántas de sus 120 imágenes fueron reconocidas.')
    names=['ConvNeXt Tiny','VMamba Tiny','Híbrido (ConvNeXt+VMamba)','ViT-B/16','Híbrido (ConvNeXt+ViT-B/16)']
    metrics=[]
    for name,row in zip(names,comparison):
        fmt=lambda v:f'{100*float(v):.2f}'.replace('.',',')+' %'
        metrics.append([name,fmt(row['accuracy']),f"{float(row['macro_f1']):.4f}".replace('.',','),
                        str(round(float(row['TB_recall'])*120))+' de 120'])
    table(['Modelo','Exactitud','Macro F1','TB reconocidas'],metrics,[5.6,3.3,3.3,3.8])
    figure('classifier_comparison.png','Comparación de los cinco clasificadores en validación.',folder=HYBRID_VIT_EVIDENCE)
    p('Los tres modelos superiores (híbrido ConvNeXt+VMamba, ViT-B/16 e híbrido ConvNeXt+ViT-B/16) superan con claridad a ConvNeXt Tiny y VMamba Tiny. El híbrido original y ViT-B/16 quedan prácticamente empatados en Macro F1 (0,9645 frente a 0,9644), con mayor sensibilidad TB en ViT-B/16 (110 frente a 107). Por su parte, el híbrido ConvNeXt + ViT-B/16 iguala la exactitud máxima (97,82 %) con solo tres falsas alarmas TB y logra la mejor separación probabilística de TB (ROC AUC de 0,9975 y AP de 0,9806), lo que abre margen para ajustar su umbral operativo. Todavía falta comprobar el desempeño en el conjunto de prueba.')

    h('Resultados de ConvNeXt Tiny',page=True)
    p('ConvNeXt obtuvo 97,10 % de exactitud y Macro F1 de 0,9427. Completó 12 épocas de entrenamiento y su mejor versión fue la época 11. Una época equivale a una pasada por los datos de entrenamiento.')
    figure('convnext_tiny_training.png','Pérdidas y métricas de ConvNeXt durante el entrenamiento.')
    p('Reconoció 94 de las 120 imágenes TB, una sensibilidad de 78,33 %. Omitió 26: cuatro se clasificaron como health y 22 como sick. Además, tres imágenes sick recibieron la etiqueta TB. Su principal dificultad fue distinguir TB de otras enfermedades.')
    figure('convnext_tiny_confusion.png','Errores de ConvNeXt con conteos y porcentajes por clase real.')
    p('En las matrices, las filas indican la clase real y las columnas la predicción. Health corresponde a sin enfermedad; sick, a otras enfermedades. Los valores de la diagonal son aciertos.')

    h('Resultados de VMamba Tiny',page=True)
    p('VMamba obtuvo 96,86 % de exactitud y Macro F1 de 0,9497. Completó 12 épocas y su mejor versión fue la época 11. Aunque su exactitud general fue ligeramente menor que la de ConvNeXt, mejoró el reconocimiento de TB.')
    figure('vmamba_tiny_training.png','Pérdidas y métricas de VMamba durante el entrenamiento.')
    p('Reconoció 101 de las 120 imágenes TB, una sensibilidad de 84,17 %. Omitió 19: 12 se clasificaron como health y siete como sick. Cuatro imágenes sick recibieron la etiqueta TB. El balance entre clases mejoró, pero todavía hubo omisiones importantes de TB.')
    figure('vmamba_tiny_confusion.png','Errores de VMamba con conteos y porcentajes por clase real.')

    h('Resultados del modelo híbrido (ConvNeXt + VMamba)',page=True)
    p('El híbrido original obtuvo 97,82 % de exactitud y Macro F1 de 0,9645. Terminó en la época 7 por parada anticipada y se conservó la época 3, que fue la mejor. Las curvas muestran que la pérdida de validación aumentó después, por lo que se mantiene esa versión.')
    figure('hybrid_cross_attention_training.png','Pérdidas y métricas del híbrido ConvNeXt + VMamba durante el entrenamiento.')
    p('Acertó 1.214 de las 1.241 imágenes y reconoció 107 de las 120 TB, una sensibilidad de 89,17 %. Omitió 13: seis se clasificaron como health y siete como sick. Además, cuatro imágenes sick recibieron la etiqueta TB. Sigue siendo el candidato principal por Macro F1, aunque esos errores todavía requieren revisión.')
    figure('hybrid_cross_attention_confusion.png','Errores del híbrido ConvNeXt + VMamba con conteos y porcentajes por clase real.')

    h('Resultados de ViT-B/16',page=True)
    p('ViT-B/16 obtuvo 97,74 % de exactitud y Macro F1 de 0,9644. Se entrenó con pesos oficiales ImageNet adaptados mediante interpolación bicúbica de embeddings posicionales de 224 a 512 píxeles; completó 12 épocas y su mejor versión fue la época 11.')
    figure('vit_b_16_training.png','Pérdidas y métricas de ViT-B/16 durante el entrenamiento.',folder=VIT_EVIDENCE)
    p('Acertó 1.213 de las 1.241 imágenes y reconoció 110 de las 120 TB, la sensibilidad por argmax más alta entre los cinco clasificadores (91,67 %). Omitió 10: tres se clasificaron como health y siete como sick. Ninguna imagen health fue confundida con TB y siete imágenes sick recibieron la etiqueta TB.')
    figure('vit_b_16_confusion.png','Errores de ViT-B/16 con conteos y porcentajes por clase real.',folder=VIT_EVIDENCE)

    h('Resultados del híbrido ConvNeXt + ViT-B/16',page=True)
    p('El nuevo híbrido ConvNeXt-Tiny + ViT-B/16 con atención cruzada bidireccional obtuvo 97,82 % de exactitud y Macro F1 de 0,9630. Terminó en la época 6 por parada anticipada y conservó la época 2 (fin del warmup de la fusión y la norma final), pues al descongelar las últimas etapas de ambas ramas en la época 3 la pérdida de validación aumentó de 0,159 a 0,522.')
    figure('hybrid_convnext_vit_training.png','Pérdidas y métricas del híbrido ConvNeXt + ViT-B/16 durante el entrenamiento.',folder=HYBRID_VIT_EVIDENCE)
    p('Acertó 1.214 de las 1.241 imágenes y reconoció 105 de las 120 TB (87,50 % de sensibilidad), omitiendo 15 (cuatro health y 11 sick). Generó solo tres falsas alarmas TB (una health y dos sick), logrando la mayor precisión TB (97,22 %), especificidad (99,73 %), ROC AUC (0,9975) y Average Precision (0,9806) de los cinco modelos.')
    figure('hybrid_convnext_vit_confusion.png','Errores del híbrido ConvNeXt + ViT-B/16 con conteos y porcentajes por clase real.',folder=HYBRID_VIT_EVIDENCE)

    h('Umbrales y estado de la evaluación',page=True)
    p('Se exploró la probabilidad mínima p_TB para emitir una alerta TB en los tres modelos punteros (híbrido ConvNeXt+VMamba, ViT-B/16 e híbrido ConvNeXt+ViT-B/16), por separado de la clase original por argmax. Los valores siguientes son ejemplos de validación y todavía no están fijados.')
    table(['Modelo','Umbral','TB identificadas','TB omitidas','Falsas alarmas'],
          [['Híbrido (C+VM)','0,10',111,9,5],['Híbrido (C+VM)','0,50',107,13,4],['Híbrido (C+VM)','0,90',100,20,1],
           ['ViT-B/16','0,10',115,5,22],['ViT-B/16','0,50',110,10,6],['ViT-B/16','0,90',100,20,2],
           ['Híbrido (C+ViT)','0,10',111,9,15],['Híbrido (C+ViT)','0,30',108,12,5],['Híbrido (C+ViT)','0,50',105,15,2],['Híbrido (C+ViT)','0,90',88,32,0]],
          [3.8,2.2,3.5,3.2,3.3])
    p('En el híbrido ConvNeXt+VMamba, bajar de 0,50 a 0,10 elevó a 111 los casos TB con cinco falsas alarmas. En ViT-B/16, el umbral 0,50 identifica 110 casos con seis falsas alarmas y bajar a 0,10 alcanza 115 de 120 TB con 22 falsas alarmas. En el híbrido ConvNeXt+ViT-B/16, bajar a 0,30 recupera 108 casos TB con solo cinco falsas alarmas y subir a 0,90 elimina las falsas alarmas a costa de omitir 32 casos. Falta acordar el equilibrio entre omisiones, carga de revisión y la selección final del clasificador.')

    h('Detector de lesiones y siguientes pasos')
    p('YOLO11n conserva los resultados de la fase anterior: mAP50 de 0,484 y mAP50–95 de 0,216. Estas medidas evalúan las cajas y su ubicación, no el porcentaje de pacientes diagnosticados. En el punto reportado por la herramienta recuperó cerca del 45,1 % de las lesiones anotadas; su capacidad de localización sigue siendo limitada.')
    p('El diagnóstico nuevo del detector no se activó en estas ejecuciones. Falta revisar cajas y errores, comprobar las explicaciones visuales y el coste computacional de los candidatos, elegir el modelo final entre los tres punteros, fijar los umbrales y evaluar la concordancia. La regla propuesta conserva la alerta del clasificador aunque no haya cajas y envía esa discrepancia a revisión.')
    p('Antes de abrir el test deben quedar fijados el modelo, los pesos, el preprocesamiento, los umbrales y la concordancia. La comparación usa una sola semilla y validación reutilizada, sin intervalos de confianza ni independencia por paciente garantizada. Por ello, la fase 5 sigue abierta y los conjuntos de prueba y externos permanecen reservados.')
    h('Fuentes revisadas')
    link('Notebook 06 ejecutado en Google Colab','https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD')
    link('Resultados y registros de la ejecución de fase 5','https://drive.google.com/drive/folders/17JBPtxHx5jL0SEnPY-T56MEt6YM7URgW')
    link('Figuras y tablas exportadas de fase 5','https://drive.google.com/drive/folders/1r0NumlzRBzGZiQicnYeHEw_k7RWAXjZS')
    link('Notebook 04b (ViT-B/16) ejecutado en Google Colab','https://colab.research.google.com/drive/1tvKNsTzirSa4G7wXO_l7xZqp3YtoM0fU')
    link('Corrida y registros de ViT-B/16 en Drive','https://drive.google.com/drive/folders/1yqZz22Wm6EQgV9zekJtt0mHJTxuWGoF8')
    link('Figuras y tablas exportadas de ViT-B/16','https://drive.google.com/drive/folders/1m5WrwO8KpmZq5XYJEt3KO22iOcb1Igh6')
    link('Notebook 04c (Híbrido ConvNeXt + ViT-B/16) ejecutado en Google Colab','https://colab.research.google.com/drive/1hr1r-vRpuHDOUV9KmRFlh9AzL-k4Tbvw')
    link('Corrida y registros de Híbrido ConvNeXt + ViT-B/16 en Drive','https://drive.google.com/drive/folders/1h3QICmhofofrIFbhMY3tOSdkBrx8xVOT')
    link('Figuras y tablas exportadas de Híbrido ConvNeXt + ViT-B/16 y comparación de 5 modelos','https://drive.google.com/drive/folders/1NsRcHaE7VCyz2drTjTIAF_JasZvgE0tF')

    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run('Página '))
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
    run=OxmlElement('w:r');t=OxmlElement('w:t');t.text='1';run.append(t);field.append(run);footer._p.append(field)
    doc.core_properties.title='Resultados de validación y avance de la fase 5'
    doc.core_properties.subject='Comparación de clasificadores y estado de la evaluación de lesiones TB'
    doc.core_properties.author=''
    doc.core_properties.keywords='fase 5, validación, tuberculosis'
    doc.save(TARGET)
    (ROOT/'docs/Informe_Avance_Fase_5.md').write_text('\n'.join(markdown),encoding='utf-8')
    print(TARGET)


if __name__=='__main__':
    if '--graph-source' in sys.argv: graph_source()
    else: main()

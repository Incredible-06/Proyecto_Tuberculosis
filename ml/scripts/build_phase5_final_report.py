"""Genera el Informe Final de Fase 5 (.docx y .md) desde evidencia ejecutada y verificada."""
import csv
import json
from pathlib import Path
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / 'reports/phase5_diag/phase5_diag_20261116T042213Z_ac3bcb'
FINAL_DIR = ROOT / 'reports/phase5_final/phase5_final_20261116T051424Z_f5c880'
HYBRID_VIT_EVIDENCE = ROOT / 'reports/phase5_review/hybrid_vit_evidence'
TARGET_DOCX = ROOT / 'docs/Informe_Final_Fase_5.docx'
TARGET_MD = ROOT / 'docs/Informe_Final_Fase_5.md'
BLACK = RGBColor(0, 0, 0)


def read_csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def fmt_pct(v):
    return f'{100 * float(v):.2f}'.replace('.', ',') + ' %'


def fmt_num(v, digits=4):
    return f'{float(v):.{digits}f}'.replace('.', ',')


def font(run, bold=None):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.color.rgb = BLACK
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
        fonts.set(qn('w:' + key), 'Times New Roman')
    if bold is not None:
        run.bold = bold
    return run


def main():
    val_5 = read_csv_rows(DIAG_DIR / 'validation_5models_verified.csv')
    boot_cls = json.loads((DIAG_DIR / 'bootstrap_uncertainty.json').read_text(encoding='utf-8'))['classifiers']
    det_table = read_csv_rows(DIAG_DIR / 'detector_thresholds_detailed.csv')
    cost_rows = read_csv_rows(DIAG_DIR / 'computational_cost.csv')
    test_summary = json.loads((FINAL_DIR / 'test_evaluation/test_evaluation_summary.json').read_text(encoding='utf-8'))
    ext_summary = json.loads((FINAL_DIR / 'external_evaluation/external_evaluation_summary.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'configs/phase5_frozen_contract.json').read_text(encoding='utf-8'))
    deploy_manifest = json.loads((ROOT / 'models/frozen_phase5/deployment_manifest.json').read_text(encoding='utf-8'))

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.3)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.footer_distance = Cm(1)

    for style in doc.styles:
        if style.type in [1, 2]:
            style.font.name = 'Times New Roman'
            style.font.size = Pt(12)
            style.font.color.rgb = BLACK
            style.font.underline = False
            rpr = style.element.get_or_add_rPr()
            for color in rpr.findall(qn('w:color')):
                for attr in list(color.attrib):
                    if attr != qn('w:val'):
                        del color.attrib[attr]
                color.set(qn('w:val'), '000000')
            for border in list(style.element.iter(qn('w:pBdr'))):
                border.getparent().remove(border)

    normal = doc.styles['Normal'].paragraph_format
    normal.line_spacing = 1.5
    normal.space_after = Pt(6)
    for name in ['Title', 'Heading 1', 'Heading 2']:
        style = doc.styles[name]
        style.font.bold = True
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.keep_with_next = True
    doc.styles['Title'].paragraph_format.space_before = Pt(0)
    doc.styles['Title'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    markdown = []

    def p(text, style=None):
        par = doc.add_paragraph(style=style)
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if style is None else WD_ALIGN_PARAGRAPH.LEFT
        font(par.add_run(text))
        par.paragraph_format.widow_control = True
        markdown.append(text + '\n')
        return par

    def h(text, page=False):
        par = p(text, 'Heading 1')
        par.paragraph_format.page_break_before = page
        markdown[-1] = '## ' + text + '\n'
        return par

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, width in enumerate(widths):
            t.columns[i].width = Cm(width)
        for i, text in enumerate(headers):
            t.rows[0].cells[i].text = text
        for row in rows:
            cells = t.add_row().cells
            for i, text in enumerate(row):
                cells[i].text = str(text)
        for ri, row in enumerate(t.rows):
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement('w:cantSplit'))
            if ri == 0:
                props.append(OxmlElement('w:tblHeader'))
            for ci, cell in enumerate(row.cells):
                cell.width = Cm(widths[ci])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for side in ['top', 'bottom', 'left', 'right']:
                    edge = OxmlElement('w:' + side)
                    edge.set(qn('w:val'), 'single')
                    edge.set(qn('w:sz'), '4')
                    edge.set(qn('w:color'), 'D9D9D9')
                    borders.append(edge)
                pr.append(borders)
                margins = OxmlElement('w:tcMar')
                for side in ['top', 'bottom', 'left', 'right']:
                    margin = OxmlElement('w:' + side)
                    margin.set(qn('w:w'), '80')
                    margin.set(qn('w:type'), 'dxa')
                    margins.append(margin)
                pr.append(margins)
                if ri == 0:
                    shade = OxmlElement('w:shd')
                    shade.set(qn('w:fill'), 'F2F2F2')
                    pr.append(shade)
                for par in cell.paragraphs:
                    par.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    par.paragraph_format.line_spacing = 1.0
                    par.paragraph_format.space_before = Pt(2)
                    par.paragraph_format.space_after = Pt(2)
                    for run in par.runs:
                        font(run, bold=ri == 0)
        markdown.append(
            '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n'
            + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'
        )
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def link(label, url):
        par = doc.add_paragraph()
        par.paragraph_format.line_spacing = 1.15
        par.paragraph_format.space_after = Pt(3)
        relation = par.part.relate_to(url, RT.HYPERLINK, is_external=True)
        element = OxmlElement('w:hyperlink')
        element.set(qn('r:id'), relation)
        run = OxmlElement('w:r')
        props = OxmlElement('w:rPr')
        for tag, attrs in [
            ('rFonts', {'ascii': 'Times New Roman', 'hAnsi': 'Times New Roman'}),
            ('color', {'val': '000000'}),
            ('sz', {'val': '24'}),
            ('u', {'val': 'none'}),
        ]:
            item = OxmlElement('w:' + tag)
            for key, value in attrs.items():
                item.set(qn('w:' + key), value)
            props.append(item)
        run.append(props)
        text = OxmlElement('w:t')
        text.text = label
        run.append(text)
        element.append(run)
        par._p.append(element)
        markdown.append(f'[{label}]({url})\n')

    figure_number = 0

    def figure(path, caption):
        nonlocal figure_number
        figure_number += 1
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.line_spacing = 1
        par.paragraph_format.space_after = Pt(2)
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(path), width=Cm(16))
        cap = p(f'Figura {figure_number}. {caption}')
        cap.paragraph_format.line_spacing = 1.0
        cap.paragraph_format.space_after = Pt(6)
        markdown.append(f'![{caption}]({path})\n')

    title = doc.add_paragraph(style='Title')
    font(title.add_run('Informe final de fase 5: selección, explicabilidad, evaluación congelada y generalización'), True)
    markdown.append('# Informe final de fase 5: selección, explicabilidad, evaluación congelada y generalización\n')

    p(
        'La fase 5 completa el cierre experimental del sistema de apoyo diagnóstico para tuberculosis en radiografías de tórax. '
        'En primer lugar se completó el diagnóstico integral sobre el conjunto de validación (1.241 imágenes en 1.237 grupos de contenido) '
        'para los cinco clasificadores entrenados bajo protocolo controlado (ConvNeXt-Tiny, VMamba-Tiny, Híbrido ConvNeXt+VMamba, ViT-B/16 e Híbrido ConvNeXt+ViT-B/16) '
        'y el detector baseline YOLO11n, incorporando intervalos de confianza por bootstrap estratificado agrupado por group_id, '
        'coste computacional comparable en GPU NVIDIA L4, revisión visual determinista de errores y explicabilidad verificada con Grad-CAM, Attention Rollout y CrossFusion.'
    )
    p(
        'Con esa evidencia se congeló el contrato inmutable antes de abrir el conjunto de prueba reservado: clasificador ViT-B/16 '
        '(umbral operativo p_TB >= 0,30 conservando argmax y probabilidades originales), detector auxiliar YOLO11n (umbral de confianza >= 0,25) '
        'y regla de concordancia aditiva en cuatro estados sin supresión de alertas del clasificador. '
        'En la evaluación única sobre test interno (1.241 radiografías), el sistema congelado alcanzó 97,18 % de exactitud, Macro-F1 de 0,9551, '
        'TB ROC AUC de 0,9941 y una sensibilidad combinada TB de 95,83 % (115 de 120 casos), con apenas 0,98 % de falsas alertas en imágenes no TB. '
        'Por contraste, la evaluación externa en Montgomery (138) y Shenzhen (662) evidenció un fuerte domain shift hospitalario sobre el clasificador global '
        '(especificidad de 25,00 % y 12,88 %), mientras el detector localizado YOLO11n mantuvo AUC de 0,811 y 0,804 con especificidad de 100,0 % y 97,5 %.'
    )

    h('1. Comparación de validación e incertidumbre por grupos (group_id)')
    p(
        'Sobre las 1.241 imágenes de validación se verificaron los cinco checkpoints contra sus hashes de procedencia originales '
        '(5/5 pruebas smoke superadas con delta máximo de probabilidad de 5,71e-4 en NVIDIA L4 con backend oflex). '
        'Para cuantificar incertidumbre respetando los pares de duplicados cercanos, se aplicó bootstrap estratificado por group_id (1.237 grupos, 2.000 réplicas, semilla 20261019).'
    )

    val_rows_tbl = []
    for r in val_5:
        arch = r['architecture']
        b_m = boot_cls['models'][arch]['metrics']
        val_rows_tbl.append([
            r['display_name'],
            fmt_pct(r['accuracy']),
            f"{fmt_num(r['macro_f1'])} [{fmt_num(b_m['macro_f1']['ci_low'])}, {fmt_num(b_m['macro_f1']['ci_high'])}]",
            f"{fmt_pct(r['tb_recall'])} [{fmt_pct(b_m['tb_recall']['ci_low'])}, {fmt_pct(b_m['tb_recall']['ci_high'])}]",
            fmt_num(r['tb_roc_auc']),
        ])
    table(
        ['Modelo', 'Exactitud', 'Macro F1 (IC 95 %)', 'Sensibilidad TB argmax (IC 95 %)', 'TB AUC'],
        val_rows_tbl,
        [4.2, 2.2, 4.0, 3.8, 1.8],
    )
    figure(HYBRID_VIT_EVIDENCE / 'classifier_comparison.png', 'Comparación de los cinco clasificadores en validación.')
    p(
        'Los deltas pareados de bootstrap entre los tres candidatos punteros (Híbrido C+VM, ViT-B/16 e Híbrido C+ViT) muestran que sus intervalos de confianza del 95 % cruzan cero '
        '(por ejemplo, Híbrido C+VM menos ViT-B/16: Δ Macro-F1 = +0,00003 [-0,0174, +0,0162] y Δ sensibilidad TB = -2,50 pp [-8,33, +3,33]), '
        'confirmando un empate estadístico en Macro-F1 sobre esta validación con una sola semilla.'
    )

    h('2. Diagnóstico del detector YOLO11n, concordancia exploratoria y coste', page=True)
    p(
        'Se ejecutó YOLO11n sobre las 1.241 radiografías de validación (512 px, FP32, NMS IoU=0,7). '
        'En el punto de confianza 0,25 recuperó 56,82 % de las 176 lesiones anotadas (IoU >= 0,5), detectó al menos una caja en 74,17 % de los pacientes TB '
        'y mantuvo 0 falsas alarmas en health (0/570) y solo 7/551 en sick (0,62 % en no-TB).'
    )
    det_sel = [r for r in det_table if float(r['threshold']) in {0.1, 0.25, 0.5}]
    table(
        ['Umbral conf', 'Recall lesión', 'Precisión caja', 'Sens. imagen (cualq.)', 'Sens. imagen (loc.)', 'Falsa alarma no-TB'],
        [[
            fmt_num(r['threshold'], 2),
            fmt_pct(r['lesion_recall']),
            fmt_pct(r['box_precision']),
            fmt_pct(r['image_sensitivity_any_box']),
            fmt_pct(r['image_sensitivity_localized']),
            fmt_pct(r['negative_image_alert_rate']),
        ] for r in det_sel],
        [2.2, 2.6, 2.6, 3.0, 2.9, 2.7],
    )
    figure(DIAG_DIR / 'detector_diagnostic.png', 'Curvas por umbral y FROC exploratoria de YOLO11n en validación.')

    p(
        'La medición comparable en GPU NVIDIA L4 (FP32, 512×512) mostró que Híbrido (C+VM) y ViT-B/16 tienen latencia prácticamente idéntica por imagen '
        '(30,0 ms y 31,3 ms en batch=1), mientras Híbrido (C+ViT) requiere 40,7 ms y 115,2 M de parámetros. YOLO11n añade 19,4 ms y 2,59 M de parámetros.'
    )
    cost_tbl = []
    for r in cost_rows:
        if r['component'] in {'classifier', 'detector'}:
            cost_tbl.append([
                r['display_name'],
                f"{int(r['total_params']) / 1e6:.2f} M".replace('.', ','),
                f"{float(r['checkpoint_mb']):.1f} MB".replace('.', ','),
                f"{float(r['b1_mean_ms']):.1f} ms".replace('.', ','),
                f"{float(r['b1_vram_mb']):.0f} MB",
                f"{float(r['b8_img_s']):.1f} img/s".replace('.', ','),
            ])
    table(
        ['Componente', 'Parámetros', 'Peso disco', 'Latencia b=1', 'VRAM pico b=1', 'Throughput b=8'],
        cost_tbl,
        [4.5, 2.2, 2.2, 2.3, 2.4, 2.4],
    )

    h('3. Explicabilidad verificada y revisión visual de errores', page=True)
    p(
        'Se implementó Grad-CAM por rama para los cinco modelos, Attention Rollout (32×32) para las ramas ViT y mapas de atención cruzada (8×8) en CrossFusion. '
        'Los 60 chequeos ejecutados (12 casos representativos de validación × 5 arquitecturas) superaron el 100 % de las pruebas de finitud, '
        'no nulidad de gradientes (incluso a través del kernel oflex de VMamba), contraste entre clase TB y health (correlación negativa entre -0,06 y -0,33) '
        'y aleatorización de la cabeza/fusión sobre copia en memoria (correlación cercana a cero o negativa frente al checkpoint entrenado).'
    )
    p(
        'En la alineación auxiliar frente a cajas GT en los siete casos TB evaluados, ViT-B/16 obtuvo el mayor enriquecimiento promedio de relevancia dentro de lesiones anotadas '
        '(1,42× frente a área uniforme), por encima de ConvNeXt-Tiny (1,31×), Híbrido C+ViT (1,22×), Híbrido C+VM (1,13×) y VMamba-Tiny (0,86×). '
        'La revisión visual de los 67 casos seleccionados (FN, FP y discrepancias) mostró que en errores compartidos difíciles (como tb0003.png) '
        'los mapas se desvían hacia regiones basales/diafragmáticas en lugar de las cajas apicales.'
    )
    figure(DIAG_DIR / 'explainability_panels/explain_01_tb0378.png', 'Comparación de mapas de explicabilidad en acierto TB (tb0378.png).')
    figure(DIAG_DIR / 'visual_review_panels/visual_review_page_01.png', 'Revisión visual de discrepancias clasificador–detector con cajas GT (verde).')

    h('4. Contrato congelado y evaluación única sobre test reservado', page=True)
    p(
        'Con base en el empate estadístico en Macro-F1, la mayor sensibilidad TB por argmax (91,67 %) y combinada (96,67 % a p_TB >= 0,30), '
        'el mayor enriquecimiento en cajas GT (1,42×) y la portabilidad en PyTorch estándar sin dependencia CUDA custom (oflex) para api/ y web/, '
        'se seleccionó y congeló ViT-B/16 (época 11) con umbral operativo p_TB >= 0,30 junto a YOLO11n (conf >= 0,25) bajo la regla aditiva de cuatro estados '
        '(configs/phase5_frozen_contract.json). La evaluación se ejecutó una sola vez sobre las 1.241 radiografías reservadas de manifests/test.csv.'
    )

    cls_m = test_summary['classifier_metrics']
    op_m = test_summary['classifier_operational_threshold']
    det_m = test_summary['detector_operational_point']
    conc_m = test_summary['concordance_summary']
    b_test_cls = test_summary['bootstrap_ci']['classifier_argmax']
    b_test_op = test_summary['bootstrap_ci']['operational_and_concordance']

    table(
        ['Métrica en test reservado (n = 1.241)', 'Valor puntual', 'IC 95 % (bootstrap group_id)'],
        [
            ['Exactitud ViT-B/16 (argmax)', fmt_pct(cls_m['accuracy']), f"[{fmt_pct(b_test_cls['accuracy']['ci_low'])}, {fmt_pct(b_test_cls['accuracy']['ci_high'])}]"],
            ['Macro-F1 ViT-B/16 (argmax)', fmt_num(cls_m['macro_f1']), f"[{fmt_num(b_test_cls['macro_f1']['ci_low'])}, {fmt_num(b_test_cls['macro_f1']['ci_high'])}]"],
            ['TB ROC AUC / Average Precision', f"{fmt_num(cls_m['per_class']['TB']['roc_auc'])} / {fmt_num(cls_m['per_class']['TB']['average_precision'])}", f"[{fmt_num(b_test_cls['tb_roc_auc']['ci_low'])}, {fmt_num(b_test_cls['tb_roc_auc']['ci_high'])}]"],
            ['Sensibilidad TB clasificador (argmax)', fmt_pct(cls_m['per_class']['TB']['recall']), f"[{fmt_pct(b_test_cls['tb_recall']['ci_low'])}, {fmt_pct(b_test_cls['tb_recall']['ci_high'])}]"],
            ['Sensibilidad TB clasificador (p_TB >= 0,30)', fmt_pct(op_m['sensitivity']), f"[{fmt_pct(b_test_op['cls_thresh_sensitivity']['ci_low'])}, {fmt_pct(b_test_op['cls_thresh_sensitivity']['ci_high'])}]"],
            ['Especificidad TB clasificador (p_TB >= 0,30)', fmt_pct(op_m['specificity']), f"[{fmt_pct(b_test_op['cls_thresh_specificity']['ci_low'])}, {fmt_pct(b_test_op['cls_thresh_specificity']['ci_high'])}]"],
            ['YOLO11n (conf >= 0,25) recall lesión / sens. caja', f"{fmt_pct(det_m['lesion_recall'])} / {fmt_pct(det_m['image_sensitivity_any_box'])}", f"[{fmt_pct(b_test_op['detector_lesion_recall']['ci_low'])}, {fmt_pct(b_test_op['detector_lesion_recall']['ci_high'])}]"],
            ['Sensibilidad TB combinada (concordancia)', f"{fmt_pct(conc_m['combined_tb_sensitivity'])} (115/120)", f"[{fmt_pct(b_test_op['combined_tb_sensitivity']['ci_low'])}, {fmt_pct(b_test_op['combined_tb_sensitivity']['ci_high'])}]"],
            ['Tasa de revisión en no-TB (concordancia)', f"{fmt_pct(conc_m['non_tb_review_rate'])} (11/1.121)", f"[{fmt_pct(b_test_op['combined_non_tb_review_rate']['ci_low'])}, {fmt_pct(b_test_op['combined_non_tb_review_rate']['ci_high'])}]"],
        ],
        [7.2, 4.2, 4.6],
    )
    figure(FINAL_DIR / 'test_evaluation/test_confusion_matrix.png', 'Matriz de confusión del contrato congelado (ViT-B/16) en test reservado.')
    figure(FINAL_DIR / 'test_evaluation/test_roc_pr.png', 'Curvas ROC OVR y Precisión-Recall TB en test reservado.')
    figure(FINAL_DIR / 'test_evaluation/test_concordance.png', 'Distribución de estados de concordancia en test reservado (p_TB >= 0,30 y YOLO11n >= 0,25).')
    p(
        'En el test reservado, YOLO11n rescató 7 casos TB adicionales que el clasificador había dejado por debajo de 0,30, '
        'llevando la detección combinada a 115/120 (95,83 %) con solo 5 casos TB omitidos, 2 falsas alertas en health (0,35 %) y 9 en sick (1,63 %). '
        'La consistencia entre validación (96,67 % combinado) y test (95,83 % combinado) confirma que el contrato congelado no sufrió sobreajuste.'
    )

    h('5. Evaluación externa (Montgomery y Shenzhen) y análisis de domain shift', page=True)
    p(
        'La auditoría previa confirmó 0 duplicados visuales entre Montgomery (138) / Shenzhen (662) y las particiones train/validation de TBX11K '
        '(las coincidencias exactas se encuentran en extra/mc+shenzhen y en test oficial no etiquetado, excluidos del entrenamiento). '
        'Dado que estos conjuntos solo disponen de etiquetas binarias (normal frente a anormal compatible con TB), se evaluaron p_TB >= 0,30, '
        'la concordancia combinada y la referencia 1 - p_health, comparándolos además con el detector YOLO11n aislado.'
    )
    mc = ext_summary['datasets']['montgomery']['clean_vs_train_val_subset']
    sz = ext_summary['datasets']['shenzhen']['clean_vs_train_val_subset']
    table(
        ['Conjunto externo', 'ViT-B/16 p_TB>=0,30 (Sens / Espec / AUC)', 'YOLO11n conf>=0,25 solo (Sens / Espec / AUC)'],
        [
            [
                'Montgomery (n = 138: 58 TB, 80 normal)',
                f"{fmt_pct(mc['p_TB_frozen_threshold']['sensitivity'])} / {fmt_pct(mc['p_TB_frozen_threshold']['specificity'])} / {fmt_num(mc['p_TB_frozen_threshold']['roc_auc'], 3)}",
                '43,10 % / 100,00 % (0/80 FP) / 0,811',
            ],
            [
                'Shenzhen (n = 662: 336 TB, 326 normal)',
                f"{fmt_pct(sz['p_TB_frozen_threshold']['sensitivity'])} / {fmt_pct(sz['p_TB_frozen_threshold']['specificity'])} / {fmt_num(sz['p_TB_frozen_threshold']['roc_auc'], 3)}",
                '49,11 % / 97,55 % (8/326 FP) / 0,804',
            ],
        ],
        [4.8, 5.6, 5.6],
    )
    p(
        'El contraste entre ambos componentes es un hallazgo relevante: en imágenes normales externas, ViT-B/16 emitió p_health cercana a cero '
        '(media 0,0046 en Montgomery y 0,0440 en Shenzhen), clasificando la mayoría de radiografías normales como sick o TB (especificidad de 25,00 % y 12,88 %). '
        'Esto refleja un domain shift clásico de TBX11K, donde las clases health, sick y TB provienen de cohortes hospitalarias y formatos distintos '
        '(contraste, proporciones y bordes tras letterbox), induciendo atajos de dominio en el clasificador de imagen completa. '
        'En contraste, YOLO11n —entrenado exclusivamente con cajas locales de lesión TB y sin distinguir entre dominios negativos— conservó AUC externa de 0,811 y 0,804 '
        'y especificidad de 100,0 % en Montgomery y 97,55 % en Shenzhen a conf >= 0,25.'
    )
    p(
        'Por rigor metodológico, la fase 5 se cierra con estos resultados sin reabrir el test interno. Para despliegue fuera del dominio TBX11K '
        'se documenta la necesidad de calibración por sitio o una futura fase 6 basada en segmentación pulmonar, normalización de contraste controlada (p. ej. ablación con CLAHE) '
        'y entrenamiento multi-hospital.'
    )

    h('6. Exportación final y fuentes verificadas')
    p(
        f"El paquete congelado quedó exportado y verificado por SHA-256 en ml/models/frozen_phase5/ "
        f"(vit_b_16_best.pt: {deploy_manifest['classifier_sha256'][:16]}..., yolo11n_best.pt: {deploy_manifest['detector_sha256'][:16]}..., "
        f"contrato: {contract['schema']}) para su integración en api/ y web/."
    )
    link('Notebook 07 ejecutado (diagnóstico, Grad-CAM, coste e incertidumbre)', 'https://colab.research.google.com/drive/1YeJmyXyk9Iwg6_GS5BAh6GvQFtkrR3gY')
    link('Resultados Parte 1 en Drive (phase5_diag)', 'https://drive.google.com/drive/folders/12zsM2dgbgWwsh2iJOxH7Q-ObVJsjIoBB')
    link('Notebook 08 ejecutado (evaluación única test y externos)', 'https://colab.research.google.com/drive/1bLNyXqWGrUULrfAWZmxCrTQhzZGbBveP')
    link('Resultados finales Parte 2 en Drive (phase5_final)', 'https://drive.google.com/drive/folders/1FwpwLNKmJryjRYkXdFYvOcIs7PfpbuSR')
    link('Notebook 06 ejecutado (preparación inicial fase 5)', 'https://colab.research.google.com/drive/1XMzT95x1p9WSBNex3yyXuQiySZODGyvD')
    link('Notebook 04b (ViT-B/16) ejecutado en Colab', 'https://colab.research.google.com/drive/1tvKNsTzirSa4G7wXO_l7xZqp3YtoM0fU')
    link('Notebook 04c (Híbrido ConvNeXt + ViT-B/16) ejecutado en Colab', 'https://colab.research.google.com/drive/1hr1r-vRpuHDOUV9KmRFlh9AzL-k4Tbvw')

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run('Página '))
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    run = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = '1'
    run.append(t)
    field.append(run)
    footer._p.append(field)

    doc.core_properties.title = 'Informe final de fase 5'
    doc.core_properties.subject = 'Selección, explicabilidad, evaluación congelada y generalización externa'
    doc.core_properties.author = ''
    doc.core_properties.keywords = 'fase 5, test, explicabilidad, tuberculosis, ViT-B/16, YOLO11n'
    doc.save(TARGET_DOCX)
    TARGET_MD.write_text('\n'.join(markdown), encoding='utf-8')
    print('Informe final generado:', TARGET_DOCX, 'y', TARGET_MD)


if __name__ == '__main__':
    main()

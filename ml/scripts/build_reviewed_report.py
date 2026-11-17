"""Informe breve con carátula existente y evidencias auténticas del proyecto."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
ASSETS = DOCS / 'assets'


def main():
    # La versión editada por el usuario es la fuente vigente. Conservar sus
    # cambios en reconstrucciones posteriores del informe.
    user_reference = ROOT / 'results/report_reference/Informe_Con_Cambios_Del_Usuario.docx'
    if user_reference.is_file():
        from improve_report_for_reader import main as improve_for_reader
        improve_for_reader()
        return
    current = ROOT / 'reports/colab_20261021'
    latest = json.loads((current / 'verification.json').read_text(encoding='utf-8'))
    assert latest['status'] == 'passed'
    env = json.loads((current / latest['environment_run'] / 'environment_setup.json').read_text(encoding='utf-8'))
    integrity = json.loads((current / latest['environment_run'] / 'transfer_integrity.json').read_text(encoding='utf-8'))
    external = json.loads((ROOT / 'reports/external_sources/preparation_summary.json').read_text(encoding='utf-8'))
    review = json.loads((ROOT / 'reports/revision_20261021/verification.json').read_text(encoding='utf-8'))
    assert review['status'] == 'passed' and external['total_images'] == 800
    assert integrity['all_image_hashes_verified'] == 12278
    # Preservar elementos y formato de la carátula aportada, no reconstruirla.
    source = DOCS / 'Informe_Fases_1_y_2_Con_Evidencias.docx'
    doc = Document(source)
    children = list(doc.element.body)
    boundary = next(i for i, element in enumerate(children)
                    if ''.join(element.xpath('.//w:t/text()')).strip() == 'Introducción y objetivos')
    cover_xml = [element.xml for element in children[:boundary]]
    for element in children[boundary:]:
        if element.tag != qn('w:sectPr'):
            doc.element.body.remove(element)
    for name in ['Normal', 'Heading 1', 'Heading 2', 'Caption']:
        style = doc.styles[name]
        style.font.name = 'Times New Roman'; style.font.size = Pt(12)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.line_spacing = 1.5 if name != 'Caption' else 1.0
    # Eliminar residuos decorativos de encabezados, conservando el contenido de portada.
    for name in ['Heading 1', 'Heading 2']:
        style = doc.styles[name]
        for border in list(style.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    markdown = ['# Informe de las fases 1 y 2', '', 'Carátula conservada del documento previo.']
    def p(text, style=None):
        par = doc.add_paragraph(text, style)
        par.paragraph_format.space_after = Pt(6)
        if style != 'Caption': par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in par.runs:
            run.font.name = 'Times New Roman'; run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0, 0, 0)
        markdown.append(text)
        return par
    def h(text, page=False):
        par = p(text, 'Heading 1'); par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        par.paragraph_format.page_break_before = page
        par.paragraph_format.keep_with_next = True
        markdown[-1] = '## ' + text
    def table(headers, rows, widths):
        tab = doc.add_table(rows=1, cols=len(headers)); tab.autofit = False
        for i, width in enumerate(widths): tab.columns[i].width = Cm(width)
        for i, text in enumerate(headers): tab.rows[0].cells[i].text = text
        for row in rows:
            cells = tab.add_row().cells
            for i, text in enumerate(row): cells[i].text = str(text)
        for ri, row in enumerate(tab.rows):
            tr = row._tr.get_or_add_trPr(); tr.append(OxmlElement('w:cantSplit'))
            if ri == 0: tr.append(OxmlElement('w:tblHeader'))
            for ci, cell in enumerate(row.cells):
                cell.width = Cm(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for side in ['top', 'bottom', 'left', 'right']:
                    edge = OxmlElement('w:' + side); edge.set(qn('w:val'), 'single')
                    edge.set(qn('w:sz'), '4'); edge.set(qn('w:color'), 'D9D9D9'); borders.append(edge)
                pr.append(borders)
                margins = OxmlElement('w:tcMar')
                for side in ['top', 'bottom', 'left', 'right']:
                    edge = OxmlElement('w:' + side); edge.set(qn('w:w'), '80'); edge.set(qn('w:type'), 'dxa'); margins.append(edge)
                pr.append(margins)
                if ri == 0:
                    shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E8E8E8'); pr.append(shade)
                for par in cell.paragraphs:
                    par.paragraph_format.line_spacing = 1.0
                    par.paragraph_format.space_before = Pt(2); par.paragraph_format.space_after = Pt(2)
                    par.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in par.runs:
                        run.font.name = 'Times New Roman'; run.font.size = Pt(12)
                        run.font.color.rgb = RGBColor(0, 0, 0); run.bold = ri == 0
        markdown.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' +
                        '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows))
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
    def figure(name, caption, width=15.8):
        par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(3); par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(ASSETS / name), width=Cm(width))
        p(caption, 'Caption')
        markdown.append(f'![{caption}](assets/{name})')

    # Los recortes se preparan desde capturas auténticas y PNG de la nueva entrega.
    # Los originales completos y sus hashes se conservan por separado.
    for name in ['F1-E1_output.png', 'transfer_integrity_output.png', 'F2-E2_output.png',
                 'training_samples_three.png', 'training_transformations_row.png']:
        assert (ASSETS / 'colab_20261021' / name).is_file(), name

    h('1 Preparación del entorno de trabajo', page=True)
    p('Este informe presenta el entorno y los datos preparados para desarrollar el sistema de apoyo a la detección de tuberculosis. Se revisaron las ejecuciones de Colab entregadas el 21 de octubre de 2026: nueve celdas de código del entorno y veinte de datos, todas ejecutadas en orden y sin errores. El código coincide con las versiones preparadas y los resultados guardados respaldan el cierre de ambas fases de preparación.')
    p('El repositorio organiza notebooks, datos, modelos, API y web. Colab utiliza Drive para conservar los archivos entre sesiones. El notebook 00 comprobó el entorno con GPU y la integridad de la copia de datos.')
    figure('colab_20261021/F1-E1_output.png', 'Figura 1. Comprobación del entorno en Colab. La salida identifica la GPU NVIDIA A100 y muestra la prueba CUDA como passed, es decir, aprobada. Esto confirma que el dispositivo pudo realizar una operación de cálculo pequeña. Entrenamiento: False indica que esta comprobación se hizo sin entrenar modelos.')
    figure('colab_20261021/transfer_integrity_output.png', 'Figura 2. Comprobación de la copia del dataset. Se compararon las huellas digitales SHA-256 de las 12.278 imágenes con el inventario previo. El resultado passed significa que todos los archivos comparados conservaron su contenido durante la transferencia.')
    p('Las evidencias de entorno se guardaron en Drive el 21 de octubre a las 18:49, hora de Lima. Las carpetas experiments y checkpoints están preparadas y vacías, como corresponde a una etapa previa al entrenamiento. La prueba CUDA básica permite comprobar el entorno; la compatibilidad de cada arquitectura se verificará al preparar los modelos.')

    h('2 Preparación del dataset y sus categorías', page=True)
    p('La copia disponible contiene 12.278 imágenes de 512 × 512 píxeles, incluidos TBX11K y datos adicionales. La base etiquetada reúne 3.800 imágenes saludables, 3.800 de enfermedad sin TB y 800 de TB. Se reservan 3.302 imágenes de test oficial sin etiquetas públicas y 576 adicionales. El inventario previo no registró errores de decodificación.')
    p('Se estudiaron las cuatro categorías descriptivas: saludable, enfermedad sin TB y los dos subtipos de TB. Para la clasificación inicial se adoptaron tres clases: health, sick y TB. La decisión conserva todas las imágenes TB y evita asignar arbitrariamente los casos que tienen ambos subtipos. Los subtipos originales se mantienen en las anotaciones de lesiones [1].')
    table(['Anotación de las imágenes TB', 'Imágenes'], [
        ['Solo ActiveTuberculosis', 630], ['Solo ObsoletePulmonaryTuberculosis', 139],
        ['Ambos subtipos', 30], ['TB sin cajas anotadas', 1], ['Total TB', 800],
    ], [12, 4])
    p('Un clasificador exclusivo de cuatro salidas requeriría definir el tratamiento de los casos mixtos. La agrupación inicial conserva estos casos y sus subtipos.')
    p('Se completó la descarga local de las fuentes externas NLM, con sus lecturas, documentos de uso y manifiestos separados [2].')
    table(['Fuente externa', 'Normal', 'Anormal compatible con TB', 'Total'], [
        ['Montgomery', 80, 58, 138], ['Shenzhen', 326, 336, 662], ['Total', 406, 394, 800],
    ], [5.1, 2.6, 5.5, 2.8])
    p('Las 800 imágenes y sus 800 lecturas fueron verificadas y permanecen en reserva externa. Sus etiquetas binarias proceden de la documentación de origen y no se equiparan a las tres clases internas. No se mezclaron con las particiones de TBX11K.')

    h('3 Particiones e integridad de los datos', page=True)
    p('Las listas oficiales contenían duplicados entre particiones. Se conservaron los originales y se retiraron de los manifiestos 126 copias exactas redundantes. Los contenidos cercanos conocidos se agruparon para mantenerlos en una misma partición. El resultado son 8.274 imágenes únicas distribuidas en 8.251 grupos, con seed 20261019 y proporción aproximada 70/15/15.')
    figure('colab_20261021/F2-E2_output.png', 'Figura 3. Particiones y control de coincidencias. Se reservaron 5.792 imágenes para entrenamiento, 1.241 para validación y 1.241 para prueba. Los ceros indican que las particiones no comparten rutas, archivos idénticos, contenidos RGB idénticos ni grupos de duplicados conocidos. La misma salida confirma la comparación completa de las 12.278 imágenes.')
    p('El entrenamiento usa sus imágenes para ajustar el modelo; la validación permitirá comparar alternativas; el test interno se conserva para la evaluación final. La clase TB tiene menos ejemplos que las demás y representa aproximadamente el 9,7 % del conjunto. Los pesos de clase se calcularon solo con las imágenes de entrenamiento.')
    p('Los siete CSV exportados desde Colab coinciden con los manifiestos locales congelados, y las pruebas locales reprodujeron las particiones sin cambios. El control reduce la fuga por duplicados conocidos. La búsqueda cercana no fue exhaustiva y faltan identificadores fiables de paciente; por ello, los ceros no garantizan independencia por paciente.')

    h('4 Anotaciones y preprocesamiento', page=True)
    p('Se conservaron y verificaron 1.211 cajas COCO sobre imágenes de 512 × 512. Los XML usan dimensiones originales distintas; se reutilizan las cajas COCO ya escaladas. La imagen tb1199 mantiene su clase TB y una marca de cajas ausentes, para impedir que se use automáticamente como negativo de detección.')
    figure('colab_20261021/training_samples_three.png', 'Figura 4. Muestras del conjunto de entrenamiento, tomadas de la salida del notebook 01. De izquierda a derecha se muestran una imagen saludable, una de enfermedad sin TB y una etiquetada como TB. En la tercera, los rectángulos delimitan lesiones anotadas en el dataset: rojo para lesiones activas y cian para antiguas. Son referencias originales para el trabajo posterior de detección.')
    p('El lector de clasificación entrega imágenes RGB de 3 × 512 × 512 en float32. La normalización usa media [0,485; 0,456; 0,406] y desviación [0,229; 0,224; 0,225]. Solo train aplica brillo y contraste de ±8 % y rotación de ±5°, con padding para conservar el campo. Validación y test son deterministas; CLAHE permanece desactivado.')
    figure('colab_20261021/training_transformations_row.png', 'Figura 5. Ejemplo de los aumentos aplicados únicamente al entrenamiento. La primera columna muestra la imagen de referencia y las tres siguientes presentan pequeñas variaciones de brillo, contraste y rotación. Estas variantes buscan aportar diversidad sin cambiar la clase de la imagen. Validación y prueba se procesan sin estos aumentos.')
    p('La estructura para detección conserva cajas, categorías y partición. El lector actual es de clasificación y no transforma cajas junto con imágenes aumentadas; esa adaptación se implementará al preparar el detector. Esta fase no incluye un detector entrenado.')

    h('5 Verificación y cierre de las fases', page=True)
    table(['Evidencia', 'Resultado'], [
        ['Pruebas locales del pipeline', '9 aprobadas; sin fallos'],
        ['Ejecuciones Colab del 21 de octubre', '9 celdas de entorno y 20 de datos sin errores'],
        ['Lote de datos en Colab CPU', '4 × 3 × 512 × 512'],
        ['Integridad en ambas ejecuciones', '12.278 imágenes comparadas en cada una'],
        ['Fuentes NLM completas', '800 imágenes y 800 lecturas verificadas'],
    ], [8.6, 7.4])
    p('La fase 1 cuenta con repositorio, notebooks, almacenamiento persistente y comprobación básica de GPU/CUDA. La fase 2 dispone de particiones reproducibles, control de duplicados conocidos, preprocesamiento y anotaciones preservadas. La adquisición de Montgomery y Shenzhen se completó en esta revisión; sus archivos permanecen separados de TBX11K.')
    p('Las fuentes externas están preparadas y reservadas, pero aún no habilitadas como evaluación independiente. TBX11K incorpora versiones redimensionadas de fuentes externas; se registraron candidatos de solapamiento que deberán revisarse antes de evaluar modelos. El test oficial sin etiquetas tampoco permite calcular métricas supervisadas con la información disponible.')
    p('Las capturas y los notebooks ejecutados respaldan estas comprobaciones. El notebook de datos guardó sus resultados en Drive el 21 de octubre a las 18:54, hora de Lima. También comprobó los hashes de los manifiestos externos y conservó el resumen de su preparación local; las 800 imágenes externas originales no se verificaron en esa sesión Colab. El entrenamiento de modelos y la integración web corresponden a etapas posteriores.')
    h('Referencias y evidencias')
    for text in [
        '[1] Liu, Y. et al. (2020). Rethinking Computer-Aided Tuberculosis Diagnosis. CVPR, 2646–2655. Fuente oficial del dataset: https://mmcheng.net/tb/',
        '[2] Jaeger, S. et al. (2014). Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. QIMS, 4(6), 475–477. Fuentes oficiales: https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html',
        'Los notebooks ejecutados, los registros de revisión, la guía de capturas y las condiciones de uso acompañan este informe en el proyecto. Repositorio del equipo: https://github.com/Incredible-06/Proyecto_Tuberculosis',
    ]:
        par = p(text); par.paragraph_format.line_spacing = 1.0
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # La carátula debe conservarse literalmente, incluidos sus párrafos vacíos.
    assert [element.xml for element in list(doc.element.body)[:boundary]] == cover_xml
    target = DOCS / 'Informe_Fases_1_y_2_Revisado.docx'
    doc.save(target)
    (DOCS / 'Informe_Fases_1_y_2.md').write_text('\n\n'.join(markdown) + '\n', encoding='utf-8', newline='\n')
    proof = {'cover_source': source.name, 'cover_elements': boundary, 'cover_preserved': True,
             'cover_xml_sha256': hashlib.sha256(''.join(cover_xml).encode()).hexdigest(),
             'report_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
             'figures': 5, 'external_images': 800,
             'colab_verification': 'reports/colab_20261021/verification.json',
             'latest_colab_code_cells': [9, 20]}
    (ROOT / 'reports/revision_20261021/report_verification.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()

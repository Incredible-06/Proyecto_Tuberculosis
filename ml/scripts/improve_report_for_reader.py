"""Mejora el Word editado por el usuario, conservando su estructura y evidencias."""
import hashlib
import io
import json
import re
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/report_reference/Informe_Con_Cambios_Del_Usuario.docx'
TARGET = ROOT / 'docs/Informe_Fases_1_y_2_Revisado.docx'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before_sha = sha(SOURCE)
    doc = Document(SOURCE)
    original = list(doc.paragraphs)
    assert len(original) == 61 and len(doc.inline_shapes) == 5
    assert original[25].text.startswith('1. FASE 1:')
    assert original[33].text.startswith('2. FASE 2:')
    assert original[41].text.startswith('3. Particiones')
    latest = json.loads((ROOT / 'reports/colab_20261021/verification.json').read_text(encoding='utf-8'))
    assert latest['status'] == 'passed'
    plot = ROOT / 'reports/colab_20261021/figures/class_distribution.png'
    assert sha(plot) == latest['figures']['class_distribution.png']['sha256']
    # Mantener imágenes, tablas, relaciones y portada de la versión del usuario.
    cover_before = [p._p.xml for p in original[:24]]
    original_media = {}
    for shape in doc.inline_shapes:
        rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        original_media[rid] = hashlib.sha256(doc.part.related_parts[rid].blob).hexdigest()
    replacements = {
        25: '1 Fase 1 Preparación del entorno de trabajo',
        26: 'El objetivo de estas dos fases fue dejar listo el entorno de trabajo y preparar los datos para los próximos experimentos. El entorno funciona en Colab, las imágenes están organizadas y las comprobaciones se ejecutaron sin errores. Este informe reúne las salidas más importantes y explica qué aporta cada una.',
        27: 'El repositorio separa el código, los notebooks, los datos, los modelos, la API y la web. Colab se usa para ejecutar el trabajo y Drive para guardar los archivos entre sesiones. El notebook de entorno comprobó que la GPU estuviera disponible y que pudiera realizar un cálculo con CUDA.',
        29: 'Figura 1. Prueba del entorno en Colab. La salida muestra una GPU NVIDIA A100 y el resultado passed, que significa que la prueba CUDA se completó correctamente. La GPU pudo realizar un cálculo pequeño. El campo Entrenamiento: False aclara que esta prueba no entrenó ningún modelo.',
        31: 'Figura 2. Revisión de la copia del dataset. Las huellas digitales SHA-256 de las 12.278 imágenes coincidieron con el inventario original. El resultado passed confirma que los archivos comparados conservaron su contenido al copiarse a Colab.',
        32: 'Los resultados quedaron guardados en Drive. Las carpetas experiments y checkpoints están listas para los experimentos y los pesos futuros, por eso todavía están vacías. El siguiente paso será comprobar la compatibilidad de los modelos que se van a entrenar.',
        33: '2 Fase 2 Preparación del dataset y sus categorías',
        34: 'El paquete reúne 12.278 imágenes de 512 × 512 píxeles. La base etiquetada incluye 3.800 imágenes saludables, 3.800 de enfermedad sin TB y 800 de TB. Se mantienen separadas 3.302 imágenes del test oficial sin etiquetas públicas y 576 adicionales. La revisión inicial no encontró errores de lectura.',
        35: 'Se revisaron las cuatro categorías previstas: saludable, enfermedad sin TB y los dos subtipos de TB. El clasificador inicial usará tres clases: health, sick y TB. Se agrupan los subtipos porque algunas imágenes tienen ambos; así se evita asignarles una categoría arbitraria y se conservan sus anotaciones.',
        37: 'Los 30 casos mixtos conservan sus dos subtipos dentro de la clase TB.',
        38: 'Montgomery y Shenzhen se descargaron desde NLM, con sus lecturas y documentos de uso, y quedaron separados de TBX11K.',
        40: 'Las 800 imágenes externas y sus lecturas se comprobaron localmente y quedaron reservadas. Sus etiquetas binarias no equivalen a las tres clases internas, por lo que no se añadieron a las particiones de TBX11K.',
        41: '3 Distribución de las imágenes',
        42: 'Después de controlar las copias exactas, quedaron 8.274 imágenes únicas para el trabajo supervisado: 3.800 health, 3.674 sick y 800 TB. Se repartieron en entrenamiento, validación y prueba, con una proporción aproximada de 70/15/15. El gráfico siguiente es la salida real que generó el notebook con esas particiones.',
        44: 'Figura 4. Salida real del control de particiones. La tabla confirma 5.792 imágenes para entrenamiento, 1.241 para validación y 1.241 para prueba. Los ceros muestran que las particiones no comparten rutas, archivos idénticos, contenidos RGB idénticos ni grupos de duplicados conocidos. También se completó la comparación de las 12.278 imágenes con el inventario.',
        46: 'Los siete archivos CSV exportados desde Colab coinciden con las listas locales que definen las particiones. Esto permite repetir el proceso con la misma distribución. El control evita que los duplicados conocidos aparezcan en conjuntos diferentes, aunque no garantiza independencia por paciente: faltan identificadores fiables y la búsqueda de imágenes parecidas no fue exhaustiva.',
        47: '5 Muestras y preprocesamiento',
        48: 'Las imágenes TB conservan 1.211 cajas COCO que señalan las zonas anotadas. Se usaron las cajas que ya corresponden al tamaño de 512 × 512, para evitar aplicar las coordenadas de los XML sobre una resolución distinta. El caso tb1199 sigue etiquetado como TB, pero está marcado porque no tiene cajas; no debe tratarse como un ejemplo negativo del detector.',
        50: 'Figura 5. Muestras reales del conjunto de entrenamiento. De izquierda a derecha: una imagen saludable, una de enfermedad sin TB y una etiquetada como TB. En la tercera, las cajas rojas marcan lesiones activas y las cian marcan lesiones antiguas, según las anotaciones del dataset. Estas cajas son referencias originales para preparar la detección.',
        51: 'Las imágenes se preparan en RGB, con tres canales y tamaño de 512 × 512. La normalización ajusta la escala de los píxeles con media [0,485; 0,456; 0,406] y desviación [0,229; 0,224; 0,225]. Solo entrenamiento recibe cambios de brillo y contraste de ±8 % y rotaciones de ±5°, con relleno para conservar el campo. Validación y prueba no reciben aumentos. CLAHE quedó desactivado.',
        53: 'Figura 6. Salida real de los aumentos de entrenamiento. La primera columna muestra la imagen de referencia y las tres siguientes muestran variaciones pequeñas de brillo, contraste y rotación. La idea es dar más variedad al entrenamiento sin cambiar la clase de la imagen. Estos cambios no se aplican a validación ni a prueba.',
        54: 'El lector actual está preparado para clasificación. Para detección ya se conservan las cajas, sus categorías y la partición de cada imagen. Cuando se prepare el detector, habrá que adaptar sus transformaciones para mover las cajas junto con la imagen.',
        55: '6 Comprobaciones y cierre de las fases',
        57: 'La fase 1 dejó listo el entorno con Colab, Drive, repositorio y una prueba básica de GPU/CUDA. La fase 2 dejó los datos organizados, las particiones reproducibles y el preprocesamiento comprobado. Las capturas muestran los resultados de esas comprobaciones y permiten revisar el avance desde el informe.',
        58: 'Montgomery y Shenzhen están descargados y reservados, pero todavía no deben presentarse como una prueba externa independiente. TBX11K incluye versiones de fuentes externas y hay candidatos de solapamiento que deben revisarse antes de evaluar modelos. El test oficial, al no tener etiquetas públicas, tampoco permite calcular métricas supervisadas con la información disponible.',
        59: 'Colab comprobó los hashes de las listas externas y guardó el resumen de su preparación local. Las 800 imágenes externas originales se revisaron localmente, no en esa sesión de Colab. El entrenamiento de modelos y la integración con la web quedan para las siguientes etapas.',
    }
    for index, text in replacements.items():
        original[index].text = text
    # Quitar el año visible de la portada por indicación del usuario.
    for run in original[24].runs: run.text = run.text.replace('-2026', '').replace('2026', '')
    assert original[24].text.strip() == 'Nuevo Chimbote'
    doc.tables[1].cell(0, 0).text = 'Validación externa'
    doc.tables[2].cell(2, 0).text = 'Ejecuciones en Colab'
    # Insertar el gráfico original entre la introducción de distribución y su explicación.
    def insert_after(anchor, text='', style=None):
        paragraph = doc.add_paragraph(text, style)
        anchor._p.addnext(paragraph._p)
        return paragraph
    graph = insert_after(original[42])
    graph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    graph.paragraph_format.keep_with_next = True
    graph.add_run().add_picture(str(plot), width=Cm(15.8))
    caption = insert_after(graph,
        'Figura 3. Distribución real por clase y partición, exportada por el notebook. Azul corresponde a entrenamiento, naranja a validación y verde a prueba. En las tres particiones hay menos imágenes de TB que de las otras clases. La tabla de la siguiente sección muestra los mismos conteos junto con sus controles de integridad.', 'Caption')
    purpose = insert_after(caption,
        'Entrenamiento servirá para ajustar los modelos; validación, para comparar alternativas; y prueba, para la evaluación final. La clase TB representa alrededor del 9,7 % de las imágenes utilizadas. Por eso se calcularon pesos de clase solo con entrenamiento, sin usar validación o prueba para ajustar ese cálculo.')
    # Dar a la evidencia de integridad una sección propia evita apretar ambas salidas.
    control = insert_after(purpose, '4 Control de duplicados e integridad', 'Heading 1')
    control.paragraph_format.page_break_before = True
    duplicate = insert_after(control,
        'Las listas originales tenían imágenes repetidas entre particiones. Para reducir esa fuga, se mantuvieron los archivos originales y se excluyeron de las listas de uso 126 copias exactas redundantes. Las imágenes parecidas ya identificadas se agruparon para que quedaran en una misma partición. Así se conservaron 8.274 imágenes únicas distribuidas en 8.251 grupos.')
    # La explicación del uso de las particiones ahora acompaña al gráfico.
    original[45]._p.getparent().remove(original[45]._p)
    # Normalizar solo los párrafos del cuerpo; la portada conserva su formato.
    for paragraph in list(doc.paragraphs)[25:]:
        pf = paragraph.paragraph_format
        pf.space_after = Pt(6)
        pf.line_spacing = 1.0 if paragraph.style.name == 'Caption' else 1.5
        if paragraph.style.name == 'Caption':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif paragraph.style.name.startswith('Heading'):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.keep_with_next = True
        elif paragraph.text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in paragraph.runs:
            run.font.name = 'Times New Roman'; run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0, 0, 0)
    for table in doc.tables:
        for ri, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.paragraph_format.space_before = Pt(2)
                    paragraph.paragraph_format.space_after = Pt(2)
                    for run in paragraph.runs:
                        run.font.name = 'Times New Roman'; run.font.size = Pt(12)
                        run.font.color.rgb = RGBColor(0, 0, 0)
                        if ri == 0: run.bold = True
    # Verificar que las cinco imágenes que eligió el usuario no se reemplazaron.
    for rid, expected in original_media.items():
        assert hashlib.sha256(doc.part.related_parts[rid].blob).hexdigest() == expected
    assert [p._p.xml for p in original[:24]] == cover_before
    all_text = '\n'.join(p.text for p in doc.paragraphs) + '\n' + '\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert not re.search(r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b|\b2026\b|\b\d{1,2}:\d{2}\b', all_text, re.I)
    assert len(doc.inline_shapes) == 6
    captions = [p.text for p in doc.paragraphs if p.style.name == 'Caption']
    assert all(text.startswith(f'Figura {i}.') for i, text in enumerate(captions, 1))
    doc.save(TARGET)
    assert sha(SOURCE) == before_sha
    # Mantener una versión de texto que respete el orden de párrafos y tablas.
    md = ['# Informe de las fases 1 y 2']
    for child in doc.element.body:
        if child.tag == qn('w:p'):
            texts = child.xpath('.//w:t/text()')
            if texts: md.append(''.join(texts).rstrip())
        elif child.tag == qn('w:tbl'):
            rows = [[''.join(c.xpath('.//w:t/text()')) for c in r.xpath('./w:tc')] for r in child.xpath('./w:tr')]
            md.append('| ' + ' | '.join(rows[0]) + ' |\n| ' + ' | '.join('---' for _ in rows[0]) + ' |\n' +
                      '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows[1:]))
    (ROOT / 'docs/Informe_Fases_1_y_2.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8', newline='\n')
    proof = {'source': str(SOURCE), 'source_sha256': before_sha, 'output_sha256': sha(TARGET),
             'user_source_unchanged': True, 'original_five_images_preserved': True,
             'added_bar_chart_sha256': sha(plot), 'chart_is_original_verified_notebook_output': True,
             'figures': 6, 'visible_dates_removed': True,
             'cover_change': 'Solo se quitó el año visible; el resto de la portada se conservó.',
             'user_edits_retained': ['Fases de entorno y datos explícitas', 'Fuentes externas separadas',
                                     'Sin reintroducir referencias o fechas eliminadas', 'Límites de la validación externa']}
    (ROOT / 'reports/colab_20261021/report_reader_improvement.json').write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    canonical = {'cover_source': str(SOURCE), 'source_user_sha256': before_sha,
                 'cover_preserved_except_year': True, 'cover_year_removed': True,
                 'report_sha256': proof['output_sha256'], 'figures': 6, 'external_images': 800,
                 'bar_chart_is_original_notebook_output': True,
                 'colab_verification': 'reports/colab_20261021/verification.json',
                 'latest_colab_code_cells': [9, 20]}
    (ROOT / 'reports/revision_20261021/report_verification.json').write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(TARGET)
    print('Figuras:', len(captions), '| Gráfico original verificado | Sin fechas visibles')


if __name__ == '__main__': main()

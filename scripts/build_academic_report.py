"""Informe de avance sustentado en evidencias existentes; ejecutar con Python documental bundled."""
import json
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'
TITLE='Desarrollo de un Sistema Web Basado en Deep Learning para el Apoyo a la Detección de Tuberculosis Pulmonar en Radiografías de Tórax'
AUTHORS=['Villar Bazan, Ronney Junior','Calvo Andrade, Sebastián Jefferson','Rodríguez Polo, Andy Javier','Rodas Aquino, Cesar Eduardo']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    evidence=ROOT/'reports/phase1/colab_review'
    with Image.open(evidence/'training_samples.png') as image:
        crops=[image.crop(box).convert('L') for box in [(5,5,335,335),(685,5,1015,335),(345,385,675,715)]]
    composite=Image.new('RGB',(1020,330),'white')
    for i,im in enumerate(crops):composite.paste(im,(i*345,0))
    composite.save(assets/'muestras_colab_sin_rotulos.png')
    with Image.open(evidence/'training_transformations.png') as image:
        before=image.crop((5,25,315,335)).convert('L')
        after=image.crop((325,25,635,335)).convert('L')
    composite=Image.new('RGB',(660,310),'white')
    composite.paste(before,(0,0));composite.paste(after,(350,0))
    composite.save(assets/'transformacion_colab_sin_rotulos.png')
    # Las figuras se recortan únicamente para retirar rótulos rasterizados; no se
    # modifican datos originales ni se crea evidencia de una ejecución distinta.
    (assets/'procedencia.json').write_text(json.dumps({'source':'reports/phase1/colab_review/',
        'samples_crops':[(5,5,335,335),(685,5,1015,335),(345,385,675,715)],
        'transform_crops':[(5,25,315,335),(325,25,635,335)],
        'presentation':'escala de grises; rotulos sustituidos por texto editable negro TNR12'},indent=2))
    git=r'C:\Users\andyr\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
    commit=subprocess.check_output([git,'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    review=json.loads((evidence/'verification.json').read_text(encoding='utf-8'))
    assert review['executed_code_cells']==14 and review['sample_hashes_verified']==27
    assert json.loads((ROOT/'reports/phase1/tests.json').read_text())['passed']
    assert json.loads((ROOT/'reports/environment/notebook_validation.json').read_text())['local_execution']
    doc=Document()
    section=doc.sections[0]
    section.page_width=Cm(21);section.page_height=Cm(29.7)
    section.top_margin=section.bottom_margin=section.left_margin=section.right_margin=Cm(2.5)
    section.header_distance=section.footer_distance=Cm(1.2)
    section.different_first_page_header_footer=True
    for style in doc.styles:
        if style.type in (1,2):
            style.font.name='Times New Roman';style.font.size=Pt(12);style.font.color.rgb=RGBColor(0,0,0)
            pr=style.element.get_or_add_rPr()
            fonts=pr.find(qn('w:rFonts'))
            if fonts is None:fonts=OxmlElement('w:rFonts');pr.insert(0,fonts)
            for k in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+k),'Times New Roman')
            for k in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:fonts.attrib.pop(qn('w:'+k),None)
    normal=doc.styles['Normal'].paragraph_format
    normal.line_spacing=1.5;normal.space_after=Pt(6)
    for name in ['Title','Subtitle','Heading 1','Heading 2','Caption']:
        pf=doc.styles[name].paragraph_format;pf.line_spacing=1.5
        pf.space_before=Pt(6);pf.space_after=Pt(6)
    for name in ['Heading 1','Heading 2']:doc.styles[name].font.bold=True
    doc.styles['Caption'].font.bold=False
    doc.styles['Caption'].font.italic=False
    # El template de Word incorpora un borde azul en Title; se elimina.
    for border in list(doc.styles.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    markdown=[]
    def run_format(run,bold=None):
        run.font.name='Times New Roman';run.font.size=Pt(12);run.font.color.rgb=RGBColor(0,0,0)
        if bold is not None:run.bold=bold
        fonts=run._element.get_or_add_rPr().get_or_add_rFonts()
        for k in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+k),'Times New Roman')
    def p(text,style=None,align=None):
        par=doc.add_paragraph(style=style)
        if align is not None:par.alignment=align
        elif style is None:par.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        run_format(par.add_run(text))
        markdown.append(text+'\n')
        return par
    def h(text,level=1):
        p(text,'Heading '+str(level));markdown[-1]='#'*(level+1)+' '+text+'\n'
    def page():doc.add_page_break();markdown.append('\n---\n')
    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for i,w in enumerate(widths):t.columns[i].width=Cm(w)
        for i,text in enumerate(headers):t.rows[0].cells[i].text=text
        for values in rows:
            cells=t.add_row().cells
            for i,text in enumerate(values):cells[i].text=str(text)
        for ri,row in enumerate(t.rows):
            prop=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');prop.append(cant)
            if ri==0:
                repeat=OxmlElement('w:tblHeader');prop.append(repeat)
            for ci,cell in enumerate(row.cells):
                cell.width=Cm(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr()
                borders=OxmlElement('w:tcBorders')
                for side in ['top','bottom','left','right']:
                    el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
                tcpr.append(borders)
                margins=OxmlElement('w:tcMar')
                for side in ['top','bottom','left','right']:
                    el=OxmlElement('w:'+side);el.set(qn('w:w'),'80');el.set(qn('w:type'),'dxa');margins.append(el)
                tcpr.append(margins)
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EEEEEE');tcpr.append(shade)
                for par in cell.paragraphs:
                    par.paragraph_format.line_spacing=1.0;par.paragraph_format.space_after=Pt(3);par.paragraph_format.space_before=Pt(3)
                    par.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for r in par.runs:run_format(r,bold=ri==0)
        markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |')
        markdown.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows);markdown.append('')
        doc.add_paragraph().paragraph_format.space_after=Pt(0)
    def figure(name,caption,width=15.7):
        par=doc.add_paragraph();par.alignment=WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next=True
        par.add_run().add_picture(str(assets/name),width=Cm(width))
        cap=p(caption,'Caption');cap.alignment=WD_ALIGN_PARAGRAPH.LEFT
        markdown.append('La figura está incluida en la versión DOCX editable; los recursos gráficos se conservan fuera de Git.\n')

    # Portada
    p('INGENIERÍA DE SOFTWARE II',align=WD_ALIGN_PARAGRAPH.CENTER)
    p('INFORME DE AVANCE DE LAS FASES 1 Y 2',align=WD_ALIGN_PARAGRAPH.CENTER)
    p(TITLE,'Title',WD_ALIGN_PARAGRAPH.CENTER)
    p('Preparación del entorno de trabajo y preparación del dataset',align=WD_ALIGN_PARAGRAPH.CENTER)
    p('Alumnos',align=WD_ALIGN_PARAGRAPH.CENTER)
    for name in AUTHORS:p(name,align=WD_ALIGN_PARAGRAPH.CENTER)
    p('Grupo A',align=WD_ALIGN_PARAGRAPH.CENTER)
    p('Docente',align=WD_ALIGN_PARAGRAPH.CENTER)
    p('Dr. Ing. Alfredo Daza Vergaray',align=WD_ALIGN_PARAGRAPH.CENTER)
    p('19 de octubre de 2026',align=WD_ALIGN_PARAGRAPH.CENTER)
    page()
    h('Introducción y objetivos')
    p('El proyecto propone un prototipo académico de sistema web para apoyar el análisis de radiografías de tórax mediante Deep Learning. Este avance desarrolla la base técnica y los datos necesarios para los experimentos posteriores. Su alcance comprende la preparación del entorno de trabajo y la preparación de TBX11K; no incluye entrenamiento de clasificadores, fusión de arquitecturas, detección de lesiones o despliegue web.')
    p('Se organizaron el proyecto, la configuración y los manifiestos reproducibles. La ejecución entregada de Google Colab comprobó lectura, transformaciones y un lote PyTorch en CPU. Permanecen pendientes la verificación real de GPU/CUDA y la comparación completa de hashes de la copia transferida; ambas tienen una prueba implementada para Colab.')
    h('Objetivo general',2)
    p('Preparar y documentar un entorno reproducible y un conjunto de datos trazable que permitan desarrollar posteriormente los modelos del proyecto sin mezclar los conjuntos de entrenamiento, validación y prueba por duplicados conocidos.')
    h('Objetivos específicos',2)
    for text in ['Organizar el código, dependencias, rutas y mecanismos de versionado para ejecutar el trabajo en Colab.',
                 'Caracterizar la copia disponible de TBX11K y conservar sus anotaciones y procedencia.',
                 'Construir particiones estratificadas y un lector con transformaciones moderadas y verificaciones reproducibles.',
                 'Documentar resultados observados y separar las comprobaciones pendientes de los logros verificados.']:
        p(text,'List Bullet')
    p('La numeración usada en este informe corresponde al plan del curso: Fase 1, entorno; Fase 2, dataset. Los nombres históricos phase1 de algunas carpetas identifican el primer desarrollo técnico de datos, sin cambiar esa correspondencia.')
    page()
    h('Fase 1 Preparación del entorno de trabajo')
    p('Se separaron notebooks, módulos reutilizables, configuración, manifiestos, resultados y documentación. Los directorios api, web y models reservan componentes futuros y explican su estado. Los originales de TBX11K se leen sin editarlos. Las reglas de Git excluyen el dataset, sus ZIP, pesos, resultados ejecutados y figuras generadas.')
    table(['Componente','Resultado comprobado'],[
        ['Python local','3.14.7'],['Python en Colab','3.13.16'],['NumPy y Pillow en Colab','2.1.3 y 11.3.0'],
        ['PyTorch en Colab','2.11.0+cpu'],['GPU y CUDA','No disponibles en la ejecución entregada'],
        ['Persistencia','Drive montado y resultados exportados']],[6.2,9.8])
    p('El notebook 01 permite montar Drive, extraer o copiar el dataset a almacenamiento temporal y registrar versiones y resultados. El almacenamiento temporal se reconstruye después de una desconexión; Drive conserva los archivos persistentes [2]. Los requisitos ligeros están separados de PyTorch/CUDA para evitar reinstalaciones innecesarias.')
    p('Se añadió 00_configuracion_entorno.ipynb para verificar por separado seeds, rutas de experimentos y checkpoints, dispositivo y una multiplicación CUDA de 4 × 4 si Colab asigna GPU. Sus siete celdas de código pasaron en modo local. La prueba CUDA está bloqueada fuera de Colab y no carga modelos.')
    p('El proyecto fue publicado en https://github.com/Incredible-06/Proyecto_Tuberculosis. El commit de código de referencia es '+commit[:12]+'. Se comprobó que los archivos preparados para Git conservan sus bytes exactos; esta regla evita que cambios automáticos de fin de línea invaliden los hashes del protocolo.')
    page()
    h('Fase 2 Caracterización y organización del dataset')
    p('TBX11K fue presentado por Liu et al. como un conjunto de 11.200 radiografías con anotaciones de regiones tuberculosas [1]. La descarga disponible incorpora también fuentes adicionales. Por ello, la cantidad local no debe interpretarse como pacientes únicos ni como imágenes exclusivamente del conjunto base.')
    table(['Contenido local','Cantidad'],[['Health','3.800'],['Sick sin TB','3.800'],['TB etiquetado','800'],
        ['Extra reservado','576'],['Test oficial sin etiquetas','3.302'],['Total bajo imgs','12.278']],[11.2,4.8])
    p('Las 12.278 imágenes son de 512 × 512 píxeles y están almacenadas en RGB: 12.174 PNG y 104 JPEG. La auditoría inicial no detectó errores de lectura o decodificación. Su inventario y hashes se reutilizaron en esta continuación; no se repitió el análisis completo.')
    p('Se adoptaron tres clases de imagen: health, sick y TB. Los subtipos pertenecen a las lesiones y pueden coexistir: hay 630 imágenes con solo lesiones activas, 139 con solo lesiones antiguas, 30 mixtas y una imagen TB sin objetos anotados. El clasificador inicial agrupa los subtipos de TB; esta decisión ajusta el alcance experimental respecto de las cuatro categorías descriptivas de la planificación.')
    p('Existen 800 XML y diez JSON COCO. Las 1.211 cajas COCO se conservaron con sus campos e identificadores y se comprobaron contra su fuente. Los XML usan dimensiones originales variables; las coordenadas COCO ya corresponden a 512². No se realizó una conversión nueva. tb1199 conserva clase TB y una bandera de cajas ausentes; no constituye un negativo para un futuro detector.')
    page()
    h('Control de duplicados y construcción de particiones')
    p('La auditoría encontró 216 pares exactos en todo el paquete. Las listas oficiales no comparten rutas, pero incluyen 38 pares exactos entre entrenamiento y validación, 64 entre entrenamiento y test disponible y nueve entre validación y test. Esto motivó un protocolo propio, preservando las listas originales como metadatos.')
    p('Dentro de las 8.400 imágenes base etiquetadas se identificaron 8.274 contenidos RGB exactos únicos. Se eligió un representante determinista por contenido y se conservaron los aliases en el índice de miembros. Las 126 copias redundantes se excluyeron de los manifiestos de uso, sin eliminar ningún archivo original.')
    p('Se refinaron los 13.783 pares candidatos de dHash igual mediante pHash de 256 bits, SSIM local, correlación y error de píxeles. Quedaron 44 pares, revisados visualmente por Codex junto con ocho controles rechazados. Se confirmaron como copias cercanas para el control de fuga, sin identificar pacientes. Veintitrés pares pertenecen al conjunto etiquetado. Los contenidos cercanos se mantienen juntos y producen 8.251 grupos indivisibles.')
    table(['Partición','Health','Sick','TB','Total'],[
        ['Train','2.660','2.572','560','5.792'],['Validación','570','551','120','1.241'],
        ['Test interno','570','551','120','1.241'],['Total','3.800','3.674','800','8.274']],[4.4,2.9,2.9,2.4,3.4])
    p('Se fijó seed 20261019 y una proporción aproximada 70/15/15 por clase, con redondeo por restos mayores y asignación de grupos indivisibles. Los manifiestos guardan rutas relativas, etiquetas, hashes, grupos, listas oficiales, aliases y anotaciones. El test interno se reserva para evaluación final; no se utiliza para seleccionar transformaciones o modelos.')
    page()
    h('Preprocesamiento y comprobaciones de funcionamiento')
    p('El lector entrega matrices float32 de forma 3 × 512 × 512 y etiquetas enteras. Convierte a RGB y conserva el campo mediante padding si cambia la proporción. Usa normalización configurable sobre RGB en [0,1], con media [0,485; 0,456; 0,406] y desviación [0,229; 0,224; 0,225], compatibles con la normalización documentada de ConvNeXt-Tiny [3]. Los requisitos de los pesos futuros deberán verificarse por arquitectura.')
    p('Solo train aplica variaciones de brillo y contraste de ±8 % y rotaciones de ±5°, con expansión antes de ajustar al lienzo. No se usan espejado, recortes agresivos o borrado de regiones. Validación y test son deterministas. CLAHE permanece desactivado para una posible ablación posterior. Los pesos de clase se calculan únicamente desde train.')
    table(['Comprobación','Resultado','Ámbito'],[
        ['Pruebas automáticas ligeras','9 aprobadas','Local'],
        ['Celdas del notebook de datos','14 sin errores','Colab CPU'],
        ['Celdas del notebook de entorno','7 aprobadas','Local'],
        ['Cruces internos conocidos','0 por ruta, hash o grupo','Local y Colab'],
        ['Lote de cuatro imágenes','4 × 3 × 512 × 512','Colab CPU'],
        ['Hashes de la transferencia','27 muestras coinciden','Colab'],
        ['Comparación completa de hashes','Implementada y pendiente','Colab']],[7.4,4.5,4.1])
    p('Los siete CSV entregados desde Colab coinciden byte a byte con los manifiestos locales congelados. Se conservaron la configuración y las decisiones de revisión. Las figuras se inspeccionaron técnicamente; esto no equivale a una validación clínica de los aumentos ni del sistema.')
    page()
    h('Dificultades soluciones y limitaciones')
    p('Las clases de imagen se confundían con subtipos de lesión en la planificación. La inspección permitió separar tres clases supervisadas y conservar los subtipos como metadatos, incluyendo los casos mixtos. Las cajas XML y las imágenes distribuidas tenían dimensiones distintas; se reutilizó el COCO verificado para evitar un escalado incorrecto.')
    p('Los duplicados entre listas oficiales podían introducir fuga de datos. Se construyeron particiones propias con representantes exactos y grupos de copias cercanas confirmadas. La búsqueda cercana solo refinó candidatos de dHash igual y no fue exhaustiva. No existen identificadores fiables de paciente, por lo que no se garantiza independencia por paciente o estudio.')
    p('El test oficial disponible carece de etiquetas públicas. Sus 3.302 imágenes se conservaron separadas, sin inferir clases por nombres o duplicados. Se documentaron 93 solapamientos de contenido conocidos con datos internos. No se calcularon métricas supervisadas sobre ese test.')
    p('Se reservaron las 576 imágenes de extra, incluidas 400 de la carpeta combinada Montgomery/Shenzhen y 176 DA/DB. Una evaluación externa futura requiere etiquetas y procedencia verificadas y un control ampliado de solapamientos. Esta reserva no reconstruye las fuentes externas completas.')
    h('Conclusiones y siguientes pasos',2)
    p('El avance dispone de una estructura versionada, un pipeline de datos reproducible y evidencia de funcionamiento en Colab CPU. La preparación del dataset está implementada y comprobada en el alcance descrito. Para cerrar todos los criterios de la preparación del entorno de la planificación, falta ejecutar la prueba GPU/CUDA del notebook 00 y registrar la comparación completa de hashes de la transferencia.')
    p('Después de incorporar esas evidencias, la etapa posterior será verificar implementaciones y pesos de ConvNeXt-Tiny y Vision Mamba y diseñar su comparación. El test interno permanecerá reservado. Los entrenamientos y comprobaciones de modelos completos se ejecutarán exclusivamente en Colab. Este informe no presenta métricas de clasificación, una web funcional o un detector entrenado.')
    page()
    h('Anexos de evidencia visual y archivos')
    figure('muestras_colab_sin_rotulos.png','Figura 1. Muestras de train de la ejecución Colab. De izquierda a derecha: health h0001, sick s0001 y TB tb0135. Las cajas del caso TB proceden del COCO original.')
    figure('transformacion_colab_sin_rotulos.png','Figura 2. Ejemplo health h0001 de la ejecución Colab. Izquierda: transformación determinista. Derecha: una variante de train con los aumentos moderados configurados. Se muestra antes de normalizar.')
    p('Evidencias principales: results/01_exploracion_preprocesamiento.ipynb y el ZIP de resultados de Colab; reports/phase1/colab_review/verification.json; reports/phase1/tests.json; reports/environment/notebook_validation.json; manifests/manifest_metadata.json. Los nombres y hashes permiten rastrear los resultados sin publicar las imágenes originales en Git.')
    page()
    h('Referencias y trazabilidad del proyecto')
    for reference in [
        '[1] Liu, Y., Wu, Y. H., Ban, Y., Wang, H. y Cheng, M. M. (2020). Rethinking Computer-Aided Tuberculosis Diagnosis. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2646–2655. https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html',
        '[2] Google. Google Colab Frequently Asked Questions. https://research.google.com/colaboratory/faq.html',
        '[3] PyTorch. Torchvision convnext_tiny. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html']:
        par=p(reference,align=WD_ALIGN_PARAGRAPH.LEFT)
        par.paragraph_format.line_spacing=1.0
        par.paragraph_format.space_after=Pt(10)
    p('Las cantidades, verificaciones y estados de este informe proceden de los artefactos generados y entregados por el proyecto. Se revisó la ejecución Colab identificada como phase1_20261020T042524626304Z, iniciada el 19 de octubre de 2026 a las 23:25 en America/Lima. La evidencia es el notebook ejecutado y sus salidas; no se observó una sesión en vivo.')
    table(['Directorio','Función'],[['notebooks','Configuración del entorno y preparación de datos'],['src/tbxdata','Lectura, preparación, validación y figuras'],['configs y manifests','Protocolo y particiones congeladas'],['reports y results','Evidencias y salidas de ejecución'],['scripts y tests','Reproducción y comprobaciones ligeras'],['api web models','Componentes futuros documentados']],[6.0,10.0])
    p('Para cerrar la evidencia restante se requiere el notebook 00 ejecutado en Colab y la carpeta results/entorno correspondiente, con environment_setup.json y transfer_integrity.json. Si no se asigna GPU, ese resultado debe conservarse como pendiente.')
    p('La revisión de fuentes y la discrepancia entre los términos publicados y los metadatos locales de licencia se documentan en docs/FUENTES_Y_LICENCIAS.md. Los originales no se modificaron.')
    # Garantía de presentación de todo el texto, incluidos runs heredados.
    for par in doc.paragraphs:
        for run in par.runs:run_format(run)
    for sec in doc.sections:
        for part in [sec.header,sec.footer,sec.first_page_header,sec.first_page_footer]:
            for par in part.paragraphs:
                for run in par.runs:run_format(run)
    doc.core_properties.title=TITLE
    doc.core_properties.author='; '.join(AUTHORS)
    doc.core_properties.subject='Ingeniería de Software II Grupo A Preparación del entorno y del dataset'
    target=OUT/'Informe_Fases_1_y_2_Final.docx'
    doc.save(target)
    (OUT/'Informe_Fases_1_y_2.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__':main()

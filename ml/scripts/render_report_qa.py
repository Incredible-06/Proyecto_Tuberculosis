"""Usa el rasterizador del skill con PDF exportado por Word en Windows.

El runtime Windows no incluye LibreOffice. Se sustituye solo el backend de
conversión por un PDF real de Word; la rasterización de render_docx.py se conserva.
"""
import importlib.util
import os
import shutil
import argparse
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem',default='Informe_Fases_1_y_2')
    parser.add_argument('--docx',default='Informe_Fases_1_y_2_Final.docx')
    parser.add_argument('--output',default='final')
    args=parser.parse_args()
    temporary=root/'docs/_tmp';temporary.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary)
    import tempfile
    tempfile.tempdir=str(temporary)
    dependencies=Path(r'C:\Users\andyr\.cache\codex-runtimes\codex-primary-runtime\dependencies')
    os.environ['PATH']=str(dependencies/'native/poppler/Library/bin')+os.pathsep+os.environ['PATH']
    source=Path(r'C:\Users\andyr\.codex\plugins\cache\openai-primary-runtime\documents\26.1020.11127\skills\documents\render_docx.py')
    spec=importlib.util.spec_from_file_location('packaged_docx_renderer',source)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    pdf=root/'docs/_qa'/f'{args.stem}.pdf'
    if not pdf.is_file():raise FileNotFoundError('Primero exporte el documento mediante Word.')
    def from_word(doc_path,user_profile,convert_tmp_dir,stem,verbose):
        target=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(pdf,target)
        return str(target),'PDF exportado por Word; LibreOffice no disponible en el runtime Windows.'
    renderer.convert_to_pdf=from_word
    pages=renderer.rasterize(str(root/'docs'/args.docx),str(root/'docs/_qa'/args.output),144,False,False)
    print('Rendered pages',len(pages))


if __name__=='__main__':main()

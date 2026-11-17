"""Rasteriza mediante render_docx.py usando el PDF real exportado por Word."""
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
temporary=ROOT/'tmp/phase5_docx_render'
temporary.mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary);tempfile.tempdir=str(temporary)
dependencies=Path(r'C:/Users/andyr/.cache/codex-runtimes/codex-primary-runtime/dependencies')
os.environ['PATH']=str(dependencies/'native/poppler/Library/bin')+os.pathsep+os.environ['PATH']
source=Path(r'C:/Users/andyr/.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('packaged_docx_renderer',source)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
pdf=ROOT/'reports/phase5_review/render/Informe_Avance_Fase_5.pdf'
if not pdf.is_file():raise FileNotFoundError('Primero exportar el DOCX usando Word.')
def from_word(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    target=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(pdf,target)
    return str(target),'PDF exportado por Microsoft Word. LibreOffice no está instalado en este runtime Windows.'
renderer.convert_to_pdf=from_word
pages=renderer.rasterize(str(ROOT/'docs/Informe_Avance_Fase_5.docx'),str(ROOT/'reports/phase5_review/render/pages'),144,False,False)
print('Páginas rasterizadas:',len(pages))

"""Valida estructura y ejecuta celdas ligeras en modo local, sin torch ni Drive."""
import argparse
import ast
import contextlib
import io
import json
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-local',action='store_true')
    p.add_argument('--notebook',default='01_exploracion_preprocesamiento.ipynb')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if Path(args.notebook).name!=args.notebook:raise ValueError('Nombre de notebook inválido.')
    nb=json.loads((root/'notebooks'/args.notebook).read_text(encoding='utf-8'))
    assert nb['nbformat']==4 and nb['nbformat_minor']==5
    assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
    executable=[]
    for i,c in enumerate(nb['cells']):
        assert c['cell_type'] in ['code','markdown'] and isinstance(c['source'],list)
        if c['cell_type']=='code':
            assert c['execution_count'] is None and c['outputs']==[]
            source=''.join(c['source'])
            ast.parse(source,filename=f'notebook_cell_{i}')
            executable.append((i,source))
    report={'notebook_format':'4.5','cells':len(nb['cells']),'code_cells_compiled':len(executable),
            'local_execution':args.execute_local,'executed_cells':[],'colab_execution':False}
    if args.execute_local:
        namespace={'__name__':'__main__'}
        for i,source in executable:
            start=time.time()
            buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(source,f'notebook_cell_{i}','exec'),namespace)
            assert not namespace.get('IN_COLAB',False),'No ejecutar este verificador local desde Colab.'
            report['executed_cells'].append({'index':i,'status':'passed','seconds':round(time.time()-start,3),'stdout':buffer.getvalue()})
            print('Cell',i,'OK',flush=True)
        assert 'torch' not in sys.modules,'El chequeo local no debe importar torch.'
        report['runtime_results']=str(namespace['RUN_DIR'].relative_to(root))
        report['training_executed']=False
        report['torch_imported']=False
    report['notebook']=args.notebook
    folder='environment' if args.notebook.startswith('00_') else 'phase1'
    output=root/'reports'/folder/'notebook_validation.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Notebook validado:',len(executable),'celdas Python; ejecución Colab pendiente.')


if __name__=='__main__':main()

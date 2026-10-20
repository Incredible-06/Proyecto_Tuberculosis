"""Ejecuta y registra las pruebas de datos locales; no importa torch."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    buffer=io.StringIO()
    start=time.time()
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'passed':result.wasSuccessful(),'seconds':round(time.time()-start,3),'python':platform.python_version(),
            'training_executed':False,'torch_imported':'torch' in sys.modules}
    output=root/'reports/phase1'
    output.mkdir(parents=True,exist_ok=True)
    (output/'tests.log').write_text(buffer.getvalue(),encoding='utf-8')
    (output/'tests.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(buffer.getvalue(),flush=True)
    if not result.wasSuccessful() or report['torch_imported']:sys.exit(1)


if __name__=='__main__':main()

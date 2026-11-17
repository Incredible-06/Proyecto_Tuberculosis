# Reemplazo de la celda F3-C1. Ejecutar en el notebook Colab ya preparado.
import json
import subprocess
import sys
from pathlib import Path
import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, '| CUDA de PyTorch:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
command = [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/setup_vmamba_colab.py'),
           '--project', str(PROJECT_ROOT), '--drive-base', str(DRIVE_BASE)]
log_path = DRIVE_BASE / 'results/vmamba_setup_diagnostic.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
print('Ejecutando instalador con salida en vivo...', flush=True)
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with log_path.open('w', encoding='utf-8') as log:
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
code = process.wait()
print('Código de salida:', code)
print('Registro completo guardado en:', log_path)
if code != 0:
    raise RuntimeError('El instalador falló. Comparte el error mostrado arriba; no continúes con los entrenamientos todavía.')

from tbxtrain.architectures import scan_backend
WEIGHTS = DRIVE_BASE / 'pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth'
print('Selective scan:', scan_backend())
if scan_backend() == 'torch_reference':
    raise RuntimeError('El instalador terminó, pero el kernel CUDA no está disponible en este proceso.')
CONFIG_SUITE = json.loads((PROJECT_ROOT / 'configs/phase3_suite.json').read_text())
display(CONFIG_SUITE)

"""Instala el kernel oficial de la revisión fijada, sin reemplazar torch/CUDA."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def patch_cuda13_cub(kernel):
    """Migra dos helpers CUB retirados, según sus reemplazos oficiales NVIDIA.

    No modifica el scan ni sus parámetros: LaneId -> registro PTX laneid;
    CTA_SYNC -> la misma __syncthreads que ejecutaba el helper anterior.
    """
    header = Path(kernel)/'csrc/selective_scan/reverse_scan.cuh'
    text = header.read_text()
    original = header.with_name(header.name+'.upstream.bak')
    if not original.exists(): original.write_text(text)
    marker = '// TBX CUDA13 CUB compatibility'
    if marker not in text:
        if text.count('cub::LaneId()') != 1 or text.count('cub::CTA_SYNC()') != 3:
            raise RuntimeError('reverse_scan.cuh difiere de la revisión fijada; no se aplica un parche ambiguo.')
        if '#pragma once' not in text: raise RuntimeError('Cabecera CUB inesperada.')
        text = text.replace('#pragma once', '#pragma once\n'+marker+'\n#include <cuda/ptx>', 1)
        text = text.replace('cub::LaneId()', '::cuda::ptx::get_sreg_laneid()')
        text = text.replace('cub::CTA_SYNC()', '__syncthreads()')
        header.write_text(text)
    return {'patch': 'cuda13_cub_helpers_v1',
            'upstream_reverse_scan_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'patched_reverse_scan_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
            'reference': 'https://github.com/NVIDIA/cccl/blob/v2.8.2/cub/cub/util_ptx.cuh'}


def patch_build_settings(kernel, torch_root):
    """Limita RAM de nvcc y usa el estándar exigido por los headers instalados."""
    setup_file = Path(kernel)/'setup.py'
    original = setup_file.read_text()
    backup = setup_file.with_name('setup.upstream.py.bak')
    if not backup.exists(): backup.write_text(original)
    header = Path(torch_root)/'include/torch/csrc/api/include/torch/all.h'
    standard = '20' if header.exists() and '202002L' in header.read_text() else '17'
    patched = original.replace('"--threads", "4"', '"--threads", "1"')
    patched = re.sub(r'-std=c\+\+(?:17|20)', '-std=c++'+standard, patched)
    setup_file.write_text(patched)
    return {'nvcc_threads': 1, 'cpp_standard': standard,
            'upstream_setup_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(),
            'patched_setup_sha256': hashlib.sha256(setup_file.read_bytes()).hexdigest()}


def build_environment(torch):
    from torch.utils.cpp_extension import CUDA_HOME
    expected_major = torch.version.cuda.split('.')[0]
    roots = [Path(CUDA_HOME)] if CUDA_HOME else []
    roots += sorted(Path('/usr/local').glob('cuda*'))
    nvcc_on_path = shutil.which('nvcc')
    if nvcc_on_path: roots.append(Path(nvcc_on_path).resolve().parents[1])
    inspected = []
    for root in dict.fromkeys(roots):
        nvcc = root/'bin/nvcc'
        if not nvcc.is_file(): continue
        p = subprocess.run([str(nvcc), '--version'], capture_output=True, text=True)
        match = re.search(r'release\s+(\d+)\.(\d+)', p.stdout+p.stderr)
        if not match: continue
        inspected.append((str(root), match.group(0)))
        if match.group(1) == expected_major:
            print('Compilador CUDA seleccionado:', nvcc, match.group(0), flush=True)
            env = dict(os.environ, MAX_JOBS='1', CMAKE_BUILD_PARALLEL_LEVEL='1',
                       PYTHONUNBUFFERED='1', CUDA_HOME=str(root))
            env['PATH'] = str(root/'bin')+os.pathsep+env.get('PATH','')
            props = torch.cuda.get_device_properties(0)
            env['TORCH_CUDA_ARCH_LIST'] = f'{props.major}.{props.minor}'
            return env
    raise RuntimeError(f'No se encontró nvcc compatible con CUDA {torch.version.cuda} de PyTorch. Toolkits inspeccionados: {inspected}. No se cambia torch/CUDA automáticamente.')


def setup(project, drive_base):
    import torch
    if not (importlib.util.find_spec('google') and importlib.util.find_spec('google.colab')) or not torch.cuda.is_available():
        raise RuntimeError('Este instalador requiere Google Colab con GPU CUDA.')
    project, drive_base = Path(project), Path(drive_base)
    info = json.loads((project/'vendor/vmamba/source.json').read_text())
    cache = Path('/content/tbx_vmamba_build')/info['revision']
    cache.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.find_spec('selective_scan_cuda_oflex')
    if spec is None:
        archive = cache/'source.tar.gz'
        if not archive.is_file():
            urllib.request.urlretrieve('https://github.com/MzeroMiko/VMamba/archive/'+info['revision']+'.tar.gz', archive)
        with tarfile.open(archive) as tar:
            for item in tar.getmembers():
                resolved = (cache/item.name).resolve()
                if cache.resolve() not in resolved.parents or item.issym() or item.islnk():
                    raise ValueError('Miembro no seguro en fuente VMamba.')
            tar.extractall(cache, filter='data')
        kernel = cache/('VMamba-'+info['revision'])/'kernels/selective_scan'
        env = build_environment(torch)
        changes = patch_build_settings(kernel, Path(torch.__file__).parent)
        if int(torch.version.cuda.split('.')[0]) >= 13:
            changes['cub_compatibility'] = patch_cuda13_cub(kernel)
        (cache/'build_settings.json').write_text(json.dumps(changes, indent=2)+'\n')
        command = [sys.executable, '-u', '-m', 'pip', 'install', '-v', '--no-build-isolation',
                   '--no-deps', '--no-cache-dir', str(kernel)]
        print('Compilando selective scan: una tarea, un hilo nvcc y salida detallada.', flush=True)
        print(json.dumps(changes, indent=2), flush=True)
        log = cache/'build.log'
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
            if process.wait(): raise RuntimeError('Falló compilación de selective scan; revise '+str(log)+'. No continúe al entrenamiento.')
        importlib.invalidate_caches()
    import selective_scan_cuda_oflex
    weights = drive_base/'pretrained'/Path(info['checkpoint_url']).name
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        temp = weights.with_suffix('.download')
        torch.hub.download_url_to_file(info['checkpoint_url'], str(temp), progress=True)
        # Se comprueba carga/arquitectura estricta en build_classifier antes de entrenar.
        temp.replace(weights)
    record = {'repository': info['repository'], 'revision': info['revision'],
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'kernel': str(selective_scan_cuda_oflex.__file__), 'weights': str(weights)}
    h = hashlib.sha256()
    with weights.open('rb') as f:
        for block in iter(lambda:f.read(1024**2), b''): h.update(block)
    record['weights_sha256'] = h.hexdigest()
    if record['weights_sha256'] != info['checkpoint_sha256']:
        raise ValueError('Checkpoint VMamba distinto del hash oficial verificado; no se carga ni entrena.')
    (drive_base/'pretrained/vmamba_setup.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return weights


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--drive-base', required=True)
    args = parser.parse_args()
    setup(args.project, args.drive_base)

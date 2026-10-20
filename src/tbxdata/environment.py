"""Registro del entorno y seeds. Las pruebas CUDA se permiten solo en Colab."""
import importlib.util
import os
import platform
import random
import sys
from pathlib import Path


def in_colab():
    return bool(importlib.util.find_spec('google') and importlib.util.find_spec('google.colab'))


def inspect_environment(seed, verify_cuda=False):
    import numpy as np
    import PIL
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED requiere establecerse antes del arranque para cambiar el
    # intérprete actual. La partición propia utiliza SHA256 y no depende de él.
    report={'python':platform.python_version(),'numpy':np.__version__,'pillow':PIL.__version__,
            'platform':platform.platform(),'seed':seed,'in_colab':in_colab(),
            'pythonhashseed_at_start':os.environ.get('PYTHONHASHSEED'),
            'training_executed':False,'cuda_probe':{'executed':False,'status':'not_requested'}}
    if verify_cuda and not report['in_colab']:
        raise RuntimeError('La prueba CUDA se ejecuta exclusivamente en Google Colab.')
    if report['in_colab']:
        import torch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        report.update(torch=torch.__version__,cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      deterministic_algorithms='enabled_with_warnings')
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            props=torch.cuda.get_device_properties(0)
            free,total=torch.cuda.mem_get_info()
            report['gpu']={'name':props.name,'compute_capability':[props.major,props.minor],
                           'memory_free_bytes':free,'memory_total_bytes':total}
            if verify_cuda:
                try:
                    a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
                    b=a@a.T
                    torch.cuda.synchronize()
                    if not torch.isfinite(b).all().item():raise RuntimeError('Resultado CUDA no finito.')
                    report['cuda_probe']={'executed':True,'status':'passed','operation':'4x4 float32 matrix multiplication; no model'}
                    del a,b
                except Exception as exc:
                    report['cuda_probe']={'executed':True,'status':'failed','error':repr(exc)}
        elif verify_cuda:
            report['cuda_probe']={'executed':False,'status':'gpu_not_available'}
    return report

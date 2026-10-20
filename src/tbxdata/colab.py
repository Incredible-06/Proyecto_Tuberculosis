"""Copias y extracción seguras para almacenamiento temporal de Colab.

Estas funciones no importan Google Drive ni modifican el origen.
"""
import json
import hashlib
import shutil
import stat
import zipfile
from pathlib import Path
from .common import safe_path


def safe_extract_zip(archive, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        needed=sum(i.file_size for i in members)
        if shutil.disk_usage(destination).free < needed+256*1024**2:
            raise OSError('Espacio temporal insuficiente para extraer el ZIP.')
        for info in members:
            safe_path(destination,info.filename)
            mode=info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError('No se aceptan enlaces simbólicos en archivos ZIP.')
        for info in members:
            target=safe_path(destination,info.filename)
            if info.is_dir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,target.open('wb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)


def locate_dataset(path):
    path=Path(path)
    candidates=([path] if (path/'imgs').is_dir() else [])
    if path.is_dir():
        candidates += [p for p in path.iterdir() if p.is_dir() and (p/'imgs').is_dir()]
    candidates=[p for p in candidates if (p/'lists/TBX11K_trainval.txt').is_file() and
                (p/'annotations/json/TBX11K_trainval.json').is_file()]
    if len(candidates)!=1:
        raise ValueError(f'No se encuentra una única raíz TBX11K en {path}. Debe contener imgs/, lists/ y annotations/.')
    return candidates[0]


def stage_dataset(source, cache, copy_to_local=True):
    """Copia carpeta o extrae ZIP. Un marcador permite reutilizar copia completa.

    Si cambió la fuente se exige otra carpeta cache. No elimina caches existentes.
    El verificador de manifiestos se ejecuta después, incluso al reutilizar cache.
    """
    source,cache=Path(source).resolve(),Path(cache).resolve()
    if not source.exists():raise FileNotFoundError(f'Fuente de dataset no encontrada: {source}')
    if not copy_to_local:
        if not source.is_dir():raise ValueError('Un ZIP requiere extracción temporal; active COPY_DATASET_TO_LOCAL.')
        return locate_dataset(source)
    if source==cache or source in cache.parents or cache in source.parents:
        raise ValueError('Origen y cache deben ser directorios separados.')
    cache.mkdir(parents=True,exist_ok=True)
    signature={'source':str(source),'bytes':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns}
    source_root=None
    files=[]
    if source.is_dir():
        source_root=locate_dataset(source)
        entries=sorted(source_root.rglob('*'))
        if any(p.is_symlink() for p in entries):
            raise ValueError('La fuente contiene enlaces simbólicos; prepare archivos regulares.')
        files=[p for p in entries if p.is_file()]
        metadata='\n'.join(f'{p.relative_to(source_root).as_posix()}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in files)
        signature['file_metadata_sha256']=hashlib.sha256(metadata.encode()).hexdigest()
    marker=cache/'stage_complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['source_signature']!=signature:
            raise ValueError('La fuente cambió. Use otra carpeta cache para conservar la copia anterior.')
        return locate_dataset(cache/'dataset')
    destination=cache/'dataset'
    if source.is_file():
        if source.suffix.lower()!='.zip':raise ValueError('Formato admitido: carpeta o ZIP.')
        safe_extract_zip(source,destination)
    else:
        needed=sum(p.stat().st_size for p in files)
        if shutil.disk_usage(cache).free < needed+256*1024**2:
            raise OSError('Espacio temporal insuficiente para copiar TBX11K.')
        for p in files:
            target=safe_path(destination,p.relative_to(source_root).as_posix())
            target.parent.mkdir(parents=True,exist_ok=True)
            # Reintento tras desconexión: solo reutiliza archivos de igual tamaño.
            # Los hashes de una muestra se comprueban después por separado.
            if not target.exists() or target.stat().st_size!=p.stat().st_size:
                shutil.copy2(p,target)
    result=locate_dataset(destination)
    marker.write_text(json.dumps({'source_signature':signature},indent=2))
    return result

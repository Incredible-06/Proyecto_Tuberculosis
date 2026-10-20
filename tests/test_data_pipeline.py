"""Pruebas locales ligeras. Sin torch, modelos, CUDA ni entrenamiento."""
import csv
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from tbxdata.colab import safe_extract_zip, stage_dataset
from tbxdata.common import safe_path, sha256
from tbxdata.dataset import TBXDataset, RadiographTransform, class_weights, load_config
from tbxdata.prepare import build_manifests
from tbxdata.validate import validate_manifests


@contextmanager
def workspace_temp():
    base=(PROJECT/'reports/phase1/test_tmp').resolve()
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temp:
        # Verifica el destino antes de que TemporaryDirectory limpie recursivamente.
        if base not in Path(temp).resolve().parents:
            raise ValueError('Directorio temporal fuera del workspace.')
        yield temp


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config(PROJECT/'configs/data.json')
        cls.root=PROJECT/'TBX11K'

    def test_manifest_integrity_and_no_group_leakage(self):
        result=validate_manifests(PROJECT)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(set(result['overlap_checks'].values()),{0})

    def test_rebuild_is_byte_identical(self):
        expected=json.loads((PROJECT/'manifests/manifest_metadata.json').read_text())
        with workspace_temp() as temp:
            rebuilt=build_manifests(PROJECT,self.root,Path(temp)/'manifests')
            self.assertEqual(expected,rebuilt)

    def test_samples_shape_finite_and_repeatable(self):
        ds=TBXDataset(self.root,PROJECT/'manifests/train.csv',self.config)
        indices=[next(i for i,r in enumerate(ds.rows) if r['class_name']==c) for c in self.config['classes']]
        for idx in indices:
            a,label=ds[idx]
            self.assertEqual(a.shape,(3,512,512));self.assertEqual(a.dtype,np.float32)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a,ds[idx][0])
            ds.set_epoch(1)
            self.assertFalse(np.array_equal(a,ds[idx][0]))
            ds.set_epoch(0)
        self.assertEqual(np.stack([ds[i][0] for i in indices]).shape,(3,3,512,512))

    def test_evaluation_is_deterministic_and_denormalizes(self):
        ds=TBXDataset(self.root,PROJECT/'manifests/validation.csv',self.config)
        a,_=ds[0];ds.set_epoch(10)
        np.testing.assert_array_equal(a,ds[0][0])
        with Image.open(self.root/ds.rows[0]['relative_path']) as im:
            np.testing.assert_array_equal(ds.transform.denormalize(a),np.asarray(im.convert('RGB')))

    def test_rejects_augmented_test_and_unlabeled_sources(self):
        for name in ['validation','test']:
            with self.assertRaises(ValueError):TBXDataset(self.root,PROJECT/f'manifests/{name}.csv',self.config,training=True)
        for name in ['official_test_unlabeled','reserved_external']:
            with self.assertRaises(ValueError):TBXDataset(self.root,PROJECT/f'manifests/{name}.csv',self.config)

    def test_class_weights_train_only(self):
        weights=class_weights(PROJECT/'manifests/train.csv',self.config)
        counts=np.array([2660,2572,560])
        self.assertAlmostEqual(float(np.sum(weights*counts)),5792,places=3)
        self.assertGreater(weights[2],weights[0])
        with self.assertRaises(ValueError):class_weights(PROJECT/'manifests/test.csv',self.config)

    def test_rejects_path_traversal_and_zip_symlinks(self):
        with workspace_temp() as temp:
            for relative in ['../outside.png','/absolute.png','C:/outside.png','imgs/../../outside']:
                with self.assertRaises(ValueError):safe_path(temp,relative)
            archive=Path(temp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','test')
            with self.assertRaises(ValueError):safe_extract_zip(archive,Path(temp)/'out')
            with zipfile.ZipFile(archive,'w') as z:
                info=zipfile.ZipInfo('link');info.external_attr=(0o120777<<16)
                z.writestr(info,'outside')
            with self.assertRaises(ValueError):safe_extract_zip(archive,Path(temp)/'out')

    def test_staging_folder_zip_and_reuse_on_tiny_fixture(self):
        with workspace_temp() as temp:
            base=Path(temp);source=base/'source'
            for name in ['imgs/test.png','lists/TBX11K_trainval.txt','annotations/json/TBX11K_trainval.json']:
                target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture')
            staged=stage_dataset(source,base/'cache')
            self.assertEqual((staged/'imgs/test.png').read_text(),'fixture')
            self.assertEqual(stage_dataset(source,base/'cache'),staged)
            archive=base/'data.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for p in source.rglob('*'):
                    if p.is_file():z.write(p,'TBX11K/'+p.relative_to(source).as_posix())
            staged_zip=stage_dataset(archive,base/'cache_zip')
            self.assertEqual((staged_zip/'imgs/test.png').read_text(),'fixture')
            (source/'imgs/test.png').write_text('changed fixture')
            with self.assertRaises(ValueError):stage_dataset(source,base/'cache')

    def test_boxes_are_original_and_missing_tb_is_flagged(self):
        ds=TBXDataset(self.root,PROJECT/'manifests/train.csv',self.config)
        idx=next(i for i,r in enumerate(ds.rows) if r['image_id']=='tb1199')
        meta=ds.metadata(idx)
        self.assertEqual(meta['class_name'],'TB');self.assertEqual(meta['boxes'],[])
        self.assertEqual(meta['annotation_status'],'missing_tb_boxes')
        self.assertEqual(meta['box_coordinate_space'],'original_512_image_xywh')


if __name__=='__main__':unittest.main(verbosity=2)

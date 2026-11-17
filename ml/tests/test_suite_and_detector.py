"""Controles de cierre, gradientes entre ramas y conversión de cajas; fixtures."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ['torch','torchvision','timm','fvcore'])
if AVAILABLE:
    import torch
    from tbxtrain.architectures import build_classifier, CrossFusion
from tbxtrain.suite import close_phase3, validate_suite, epoch_train, run_suite_model
from tbxtrain.protocol import atomic_json
from tbxtrain.detection import yolo_box, prepare_detection
from tbxdata.common import sha256


@unittest.skipUnless(AVAILABLE,'Fixture neuronal opcional: torch/torchvision/timm/fvcore requeridos.')
class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def setUp(self): self.cfg=json.loads((ROOT/'configs/phase3_suite.json').read_text())
    def test_real_architectures_shapes_and_trainable_gradients(self):
        for name in self.cfg['architectures']:
            with self.subTest(name=name):
                torch.manual_seed(53)
                model=build_classifier(name,pretrained=False)
                model.configure_phase(False)
                self.assertFalse(any(p.requires_grad for m in model.backbones() for p in m.parameters()))
                model.configure_phase(True)
                logits=model(torch.randn(2,3,32,32))
                self.assertEqual(tuple(logits.shape),(2,3))
                torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward()
                trainable=[p for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
                model.parameter_groups(self.cfg['head_lr'],self.cfg['backbone_lr'])
                del model
    def test_cross_attention_uses_both_maps_and_is_trainable(self):
        fusion=CrossFusion(channels=16,dim=8,heads=2,grid=2).eval()
        a=torch.randn(2,16,4,4,requires_grad=True)
        b=torch.randn(2,16,4,4,requires_grad=True)
        logits=fusion(a,b); logits.square().sum().backward()
        self.assertGreater(a.grad.abs().sum().item(),0)
        self.assertGreater(b.grad.abs().sum().item(),0)
        self.assertFalse(torch.allclose(logits,fusion(a.detach(),torch.zeros_like(b))))
    def test_config_and_public_runtime_guard(self):
        validate_suite(self.cfg)
        with self.assertRaises(ValueError): validate_suite(dict(self.cfg,precision='fp16'))
        with self.assertRaises(ValueError): validate_suite(dict(self.cfg,protocol_id='../escape'))
        with self.assertRaises(RuntimeError): run_suite_model(ROOT,ROOT/'TBX11K',ROOT/'tmp','vmamba_tiny',self.cfg)
    def test_close_requires_three_trained_compatible_intact_models(self):
        base=(ROOT/'tmp').resolve()
        with tempfile.TemporaryDirectory(dir=base) as temp:
            t=Path(temp); self.assertIn(base,t.resolve().parents)
            ids=[]
            for name in self.cfg['architectures']:
                run='synthetic_'+name; ids.append(run)
                ckpt=t/'checkpoints'/run/'best.pt'; ckpt.parent.mkdir(parents=True); ckpt.write_bytes(b'synthetic fixture')
                atomic_json(t/'experiments'/run/'summary.json',{
                    'architecture':name,'run_id':run,'status':'completed_controlled','training_executed':True,
                    'optimizer_updates':1,'checkpoint_sha256':sha256(ckpt),'completed_epochs':2,'best_epoch':1,
                    'protocol':self.cfg,'provenance':{'fixture':'same'},'test_evaluated':False,'external_evaluated':False,
                    'validation':{'n':1241,'macro_f1':.5,'accuracy':.5,'per_class':{'TB':{'recall':.5,'precision':.5,'specificity':.5}},'confusion_matrix':[[1,0,0],[0,1,0],[0,1,1]]}})
            with self.assertRaises(ValueError): close_phase3(t,ids[:2])
            (t/'checkpoints'/ids[-1]/'best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError): close_phase3(t,ids)
            (t/'checkpoints'/ids[-1]/'best.pt').write_bytes(b'synthetic fixture')
            summary,path=close_phase3(t,ids)
            self.assertTrue(summary['phase3_complete']); self.assertFalse(summary['winner_finalized'])
            self.assertTrue((path/'comparison.csv').exists())
    def test_weighted_accumulation_preserves_partial_window(self):
        class Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.linear=torch.nn.Linear(4,3)
            def forward(self,x): return self.linear(x)
            def configure_phase(self,fine): self.train()
        torch.manual_seed(61)
        m=Tiny(); expected=copy.deepcopy(m)
        x=torch.randn(7,4); y=torch.tensor([0,1,2,2,0,1,2]); w=torch.tensor([.5,1.,4.])
        cfg=dict(self.cfg,accumulation_steps=3,gradient_clip=1e6)
        opt=torch.optim.SGD(m.parameters(),lr=.02); ref=torch.optim.SGD(expected.parameters(),lr=.02)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=2)
        result=epoch_train(torch,m,loader,opt,w,cfg,0)
        for a,b in [(0,6),(6,7)]:
            ref.zero_grad(); torch.nn.functional.cross_entropy(expected(x[a:b]),y[a:b],weight=w).backward();ref.step()
        for actual,target in zip(m.parameters(),expected.parameters()): torch.testing.assert_close(actual,target,atol=1e-7,rtol=1e-5)
        self.assertEqual(result['optimizer_updates'],2)


class DetectionTests(unittest.TestCase):
    def test_xywh_normalization_and_invalid_boxes(self):
        self.assertEqual(yolo_box({'bbox':[128,64,256,128]},512,512),(.5,.25,.5,.25))
        for box in [[0,0,0,2],[-1,0,1,1],[500,0,30,10],[float('nan'),0,1,1]]:
            with self.assertRaises(ValueError): yolo_box({'bbox':box},512,512)
    def test_missing_tb_is_excluded_and_non_tb_has_empty_labels(self):
        base=(ROOT/'tmp').resolve()
        with tempfile.TemporaryDirectory(dir=base) as temp:
            t=Path(temp); self.assertIn(base,t.resolve().parents)
            rows=[]
            for index,cls in enumerate(['health','sick','TB','TB']):
                rel=f'imgs/{index}.png'; (t/'imgs').mkdir(exist_ok=True); Image.new('RGB',(32,32)).save(t/rel)
                rows.append({'relative_path':rel,'class_name':cls,'width':'32','height':'32',
                    'boxes_json':json.dumps([{'bbox':[0,0,16,16]}] if index==2 else [])})
            with mock.patch('tbxdata.validate.validate_manifests'),mock.patch('tbxtrain.detection.read_csv',return_value=rows):
                report=prepare_detection(t,t,t/'converted')
            self.assertEqual(report['counts']['train']['excluded_TB_missing_boxes'],1)
            self.assertEqual((t/'converted/labels/train/imgs__0.txt').read_text(),'')
            self.assertEqual((t/'converted/labels/train/imgs__2.txt').read_text().strip(),'0 0.250000000 0.250000000 0.500000000 0.500000000')
            self.assertFalse((t/'converted/images/train/imgs__3.png').exists())
            self.assertNotIn('test:',(t/'converted/dataset.yaml').read_text())


if __name__=='__main__': unittest.main()

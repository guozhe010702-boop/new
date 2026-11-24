"""Mocked orchestration tests only; these do not claim NPU or HDMI validation."""
import importlib.util,json,shutil,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class DeliveryTests(unittest.TestCase):
 def fixture(self,folder,stage):
  root=Path(folder);(root/'tools').mkdir();(root/'docs').mkdir();(root/'results/qualification-20261124').mkdir(parents=True)
  for n in ('deploy.py','report.py'):shutil.copyfile(ROOT/'tools'/n,root/'tools'/n)
  (root/'README.md').write_text('Unit fixture only')
  for rel in ['runtime/lib/librknnrt.so','yolo11_bytetrack/models/int8.rknn','real_esrgan/models/fp16.rknn','depth_anything_v2/models/fp16.rknn','common/display.cpp']:
   p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'unit fixture, not deployable')
  job={'stage':stage,'cases':[],'current':'unit-fixture','observed_process_running':stage=='running','observed_boot_id':'unit','boot_id':'unit'};(root/'results/qualification-20261124/job.json').write_text(json.dumps(job))
  spec=importlib.util.spec_from_file_location('delivery_under_test',ROOT/'tools/finalize_async.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=root;m.STATE=root/'results/async-delivery.json'
  def run(args,**kw):
   if Path(args[1]).name=='sync_status.py':return None
   return subprocess.run(args,check=True,**kw)
  def remote(code):return json.dumps({'stage':'complete'}) if "print(p.read_text()" in code else ''
  def get(remote_path,path):
   path.parent.mkdir(parents=True,exist_ok=True)
   if remote_path.endswith('advanced-final-results.tgz'):
    p=root/'results/final-audit.json';p.write_text(json.dumps({'passed':True,'scope':'mocked unit fixture only'}))
    with tarfile.open(path,'w:gz') as a:a.add(p,arcname='results/final-audit.json')
   else:path.write_bytes(b'unit fixture, not deployable')
  m.run=run;m.remote=remote;m.get=get;m.send=lambda *args:None
  return root,m
 def test_completed_delivery_checks_package_hashes(self):
  with tempfile.TemporaryDirectory(prefix='delivery-unit-',dir=ROOT/'results') as tmp:
   assert Path(tmp).resolve().parent==(ROOT/'results').resolve()
   root,m=self.fixture(tmp,'complete');m.main();state=json.loads(m.STATE.read_text());self.assertEqual(state['stage'],'complete');self.assertTrue(state['audit_passed']);self.assertTrue(Path(state['package']).is_file())
 def test_failed_qualification_never_packages(self):
  with tempfile.TemporaryDirectory(prefix='delivery-unit-',dir=ROOT/'results') as tmp:
   assert Path(tmp).resolve().parent==(ROOT/'results').resolve()
   root,m=self.fixture(tmp,'failed')
   with self.assertRaises(RuntimeError):m.main()
   self.assertFalse((root/'releases').exists())
if __name__=='__main__':unittest.main()

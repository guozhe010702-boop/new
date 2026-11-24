"""Isolated quantization experiment; never replaces the selected model."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'real_esrgan/experiments/mmse32';out.mkdir(parents=True,exist_ok=False)
images=(ROOT/'data/calibration-real_esrgan.txt').read_text().splitlines();indices=np.linspace(0,len(images)-1,32,dtype=int);dataset=out/'calibration.txt';dataset.write_text(''.join(images[i]+'\n' for i in indices))
report={'algorithm':'mmse','calibration_count':32,'selection_indices':indices.tolist(),'selected_as_default':False,'started':time.time()};rk=RKNN(verbose=False);start=time.monotonic()
try:
 assert rk.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[255,255,255]],quantized_algorithm='mmse') in (0,None)
 assert rk.load_onnx(model=str(ROOT/'real_esrgan/models/model.onnx'))==0
 assert rk.build(do_quantization=True,dataset=str(dataset))==0
 path=out/'model.rknn';assert rk.export_rknn(str(path))==0;report.update(stage='converted_not_validated',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
except BaseException as e:report.update(stage='failed',error=repr(e));raise
finally:
 rk.release();report['seconds']=time.monotonic()-start;(out/'conversion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

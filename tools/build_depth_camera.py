"""Build an explicit 16:9 camera profile from original, unchanged learned weights."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np,onnx,onnxruntime as ort,torch
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-proof',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'depth_anything_v2/models';prior=json.loads((source/'conversion-fp16.json').read_text());proof=json.loads(a.teacher_proof.read_text());assert proof['summary']['294']['teacher_agreement_pass'];assert digest(source/'depth_anything_v2_vits.pth')==prior['source_sha256']==proof['source_sha256'];sys.path.insert(0,str(ROOT/'vendor/depth_anything'));from depth_anything_v2.dpt import DepthAnythingV2
 torch.set_num_threads(4);model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(source/'depth_anything_v2_vits.pth',map_location='cpu'),strict=True);model.eval();torch.manual_seed(20261125);x=torch.rand(1,3,294,518);artifact=out/'candidate.onnx';torch.onnx.export(model,x,str(artifact),opset_version=17,input_names=['images'],output_names=['output_0'],do_constant_folding=True);onnx.checker.check_model(onnx.load(artifact));options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(artifact),sess_options=options,providers=['CPUExecutionProvider'])
 with torch.inference_mode():reference=model(x).numpy()
 got=session.run(None,{'images':x.numpy()})[0];np.testing.assert_allclose(reference,got,atol=.005,rtol=.005)
 report={'stage':'preparing','target':'rk3588','precision':'fp16','input_width':518,'input_height':294,'source_sha256':prior['source_sha256'],'source_onnx_sha256':prior['onnx_sha256'],'candidate_onnx_sha256':digest(artifact),'baseline_rknn_sha256':digest(source/'fp16.rknn'),'teacher_agreement':proof['summary']['294'],'torch_onnx_max_abs_difference':float(np.max(np.abs(reference-got))),'selected_as_default':False,'board_validation_passed':False,'mean':prior['mean'],'std':prior['std'],'validation':[]};(out/'prepared').mkdir();(out/'reference').mkdir();records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation']
 for record in records:
  raw=cv2.imread(str(ROOT/record['source']));h,w=raw.shape[:2];camera=abs(w/h-16/9)<.002
  if camera:
   old=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(old)==record['prepared_sha256'];prepared=cv2.imread(str(old))[112:406].copy()
  else:
   scale=min(518/w,294/h);ww,hh=round(w*scale),round(h*scale);prepared=np.full((294,518,3),114,np.uint8);yy,xx=(294-hh)//2,(518-ww)//2;prepared[yy:yy+hh,xx:xx+ww]=cv2.resize(raw,(ww,hh))
  dest=out/'prepared'/record['file'];assert cv2.imwrite(str(dest),prepared);rgb=cv2.cvtColor(prepared,cv2.COLOR_BGR2RGB);value=(rgb.astype(np.float32)-np.asarray(prior['mean'],np.float32))/np.asarray(prior['std'],np.float32);ref=session.run(None,{'images':value.transpose(2,0,1)[None]})[0];ref.astype(np.float32).tofile(out/'reference'/(dest.stem+'-0.f32'));report['validation'].append({'file':record['file'],'prepared_sha256':digest(dest),'source':record['source'],'source_sha256':record['source_sha256'],'camera_16_9':camera})
 rk=RKNN(verbose=False,verbose_file=str(out/'compiler.log'))
 try:
  assert rk.config(target_platform='rk3588',mean_values=[prior['mean']],std_values=[prior['std']],float_dtype='float16',optimization_level=3) in (0,None);assert rk.load_onnx(model=str(artifact))==0;assert rk.build(do_quantization=False)==0;output=out/'candidate.rknn';assert rk.export_rknn(str(output))==0;report.update(stage='compiled_board_pending',candidate_rknn_sha256=digest(output),candidate_bytes=output.stat().st_size)
 finally:rk.release();(out/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='validation'}),flush=True)
if __name__=='__main__':main()

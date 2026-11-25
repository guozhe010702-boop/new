"""Evaluate removal of camera letterbox padding before building any candidate."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np,onnxruntime as ort,torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'depth_anything_v2/models';prior=json.loads((source/'conversion-fp16.json').read_text());assert digest(source/'model.onnx')==prior['onnx_sha256'];assert digest(source/'depth_anything_v2_vits.pth')==prior['source_sha256'];sys.path.insert(0,str(ROOT/'vendor/depth_anything'));from depth_anything_v2.dpt import DepthAnythingV2
 torch.set_num_threads(4);model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(source/'depth_anything_v2_vits.pth',map_location='cpu'),strict=True);model.eval();options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(source/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider']);records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation'];rows=[];report={'stage':'checking_camera_quality','input_width':518,'original_height':518,'source_sha256':prior['source_sha256'],'source_onnx_sha256':prior['onnx_sha256'],'baseline_rknn_sha256':digest(source/'fp16.rknn'),'camera_rows':rows,'selected_as_default':False}
 for record in records:
  raw=cv2.imread(str(ROOT/record['source']));h,w=raw.shape[:2]
  if abs(w/h-16/9)>.002:continue
  path=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(path)==record['prepared_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.asarray(prior['mean'],np.float32))/np.asarray(prior['std'],np.float32);x=x.transpose(2,0,1)[None];ref=session.run(None,{'images':x})[0].reshape(518,518);entry={'file':record['file'],'source':record['source'],'source_sha256':record['source_sha256'],'variants':{}}
  for height in [406,350,294]:
   top=(518-height)//2;view=np.ascontiguousarray(x[:,:,top:top+height,:])
   with torch.inference_mode():got=model(torch.from_numpy(view)).numpy().reshape(height,518)
   # Only the real 16:9 image area; padding agreement must not hide depth errors.
   y0=113-top;y1=y0+291;teacher=ref[113:404,:];student=got[y0:y1,:];entry['variants'][str(height)]={'spearman':float(spearmanr(teacher.ravel(),student.ravel()).statistic),'max_abs_difference':float(np.max(np.abs(teacher-student)))}
  rows.append(entry);print(json.dumps(entry),flush=True);(out/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
 assert len(rows)>=10;report['summary']={}
 for height in [406,350,294]:
  values=[v['variants'][str(height)]['spearman'] for v in rows];report['summary'][str(height)]={'images':len(values),'median_spearman':float(np.median(values)),'minimum_spearman':min(values),'teacher_agreement_pass':float(np.median(values))>=.98 and min(values)>=.95}
 report['stage']='camera_teacher_agreement_complete';(out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()

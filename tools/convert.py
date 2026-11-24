"""Run in the pinned x86_64 RKNN conversion container, not on the camera board."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx,onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class SRVGG(nn.Module):
    # Same architecture / parameter names as upstream SRVGGNetCompact general-x4v3.
    def __init__(self):
        super().__init__(); self.body=nn.ModuleList([nn.Conv2d(3,64,3,1,1),nn.PReLU(64)])
        for _ in range(32):self.body.extend([nn.Conv2d(64,64,3,1,1),nn.PReLU(64)])
        self.body.append(nn.Conv2d(64,48,3,1,1));self.upsampler=nn.PixelShuffle(4)
    def forward(self,x):
        y=x
        for layer in self.body:y=layer(y)
        return self.upsampler(y)+torch.nn.functional.interpolate(x,scale_factor=4,mode='nearest')
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',choices=['yolo11_bytetrack','real_esrgan','depth_anything_v2'],required=True);p.add_argument('--precision',choices=['fp16','int8'],default='fp16');p.add_argument('--onnx-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4); folder=ROOT/a.project/'models';folder.mkdir(exist_ok=True)
    report={'project':a.project,'precision':a.precision,'board_validated':False,'started':time.time()};start=time.monotonic();log=folder/('conversion-'+a.precision+'.json')
    try:
        mean=[0,0,0];std=[255,255,255]
        if a.project=='yolo11_bytetrack':
            from ultralytics import YOLO
            from ultralytics.nn.modules.head import Detect
            weight=folder/'yolo11n.pt';wrapper=YOLO(str(weight),task='detect');model=wrapper.model.cpu().float().eval();model.fuse();size=640
            for layer in model.modules():
                if type(layer) is Detect:layer.export=True;layer.format='rknn';layer.dynamic=False
            (folder/'classes.json').write_text(json.dumps(wrapper.names,ensure_ascii=False))
        elif a.project=='real_esrgan':
            size=256;weight=folder/'realesr-general-x4v3.pth';model=SRVGG();checkpoint=torch.load(weight,map_location='cpu');model.load_state_dict(checkpoint.get('params_ema',checkpoint.get('params',checkpoint)),strict=True);model.eval()
        else:
            sys.path.insert(0,str(ROOT/'vendor/depth_anything'))
            from depth_anything_v2.dpt import DepthAnythingV2
            size=518;weight=folder/'depth_anything_v2_vits.pth';model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(weight,map_location='cpu'),strict=True);model.eval()
            mean=[123.675,116.28,103.53];std=[58.395,57.12,57.375]
        path=folder/'model.onnx';x=torch.rand(1,3,size,size)
        if path.exists():
            prior=folder/'conversion-fp16.json'
            if not prior.exists():raise ValueError('Cached ONNX needs its original conversion receipt; use a fresh model folder')
            cached=json.loads(prior.read_text())
            if cached.get('source_sha256')!=digest(weight) or cached.get('onnx_sha256')!=digest(path):raise ValueError('Cached ONNX/checkpoint changed; preserve old artifacts and export to a fresh model folder')
        if not path.exists():
            with torch.inference_mode():reference=model(x)
            if isinstance(reference,torch.Tensor):reference=[reference]
            torch.onnx.export(model,x,str(path),opset_version=17 if a.project=='depth_anything_v2' else 12,input_names=['images'],output_names=['output_'+str(i) for i in range(len(reference))],do_constant_folding=True)
            onnx.checker.check_model(onnx.load(path))
            opts=ort.SessionOptions();opts.intra_op_num_threads=4
            session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
            got=session.run(None,{'images':x.numpy()});err=max(float(np.max(np.abs(u.detach().numpy()-v))) for u,v in zip(reference,got))
            for u,v in zip(reference,got):np.testing.assert_allclose(u.detach().numpy(),v,atol=.005,rtol=.005)
            report['onnx_random_max_abs_error']=err
        graph=onnx.load(path);report.update(source_sha256=digest(weight),onnx_sha256=digest(path),input_size=size,mean=mean,std=std,outputs=[{'name':v.name,'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in graph.graph.output])
        if a.onnx_only:report['stage']='onnx_validated';return
        from rknn.api import RKNN
        rk=RKNN(verbose=False)
        try:
            assert rk.config(mean_values=[mean],std_values=[std],target_platform='rk3588') in (None,0)
            assert rk.load_onnx(model=str(path))==0
            dataset=ROOT/'data'/('calibration-'+a.project+'.txt')
            if a.precision=='int8' and not dataset.is_file():raise ValueError('Missing representative calibration set; no random-image INT8 substitute')
            assert rk.build(do_quantization=a.precision=='int8',dataset=str(dataset) if a.precision=='int8' else None)==0
            artifact=folder/(a.precision+'.rknn');temporary=folder/(a.precision+'.pending.rknn');assert rk.export_rknn(str(temporary))==0;temporary.replace(artifact)
            report.update(stage='converted_not_board_validated',rknn_sha256=digest(artifact))
        finally:rk.release()
    except BaseException as e:
        report.update(stage='failed',error=str(e),traceback=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-start
        destination=folder/('conversion-'+a.precision+'-failed-'+str(time.time_ns())+'.json') if report.get('stage')=='failed' and log.exists() else log
        destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

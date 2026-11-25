"""Build isolated FP16 placement candidates; never change the default model."""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np,onnxruntime as ort,onnx
from onnx import helper
from rknn.api.custom_op import get_node_attr
from scipy.stats import spearmanr
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
class cstAlignCornersResize:
 op_type='cstAlignCornersResize'
 def shape_infer(self,node,in_shapes,in_dtypes):return [[in_shapes[0][0],in_shapes[0][1],get_node_attr(node,'height'),get_node_attr(node,'width')]], [in_dtypes[0]]
 def compute(self,node,inputs):
  value=inputs[0]
  for axis,size in [(3,get_node_attr(node,'width')),(2,get_node_attr(node,'height'))]:
   position=(np.arange(size,dtype=np.float32)*np.float32(value.shape[axis]-1))/np.float32(size-1);lo=np.floor(position).astype(np.int64);hi=np.minimum(lo+1,value.shape[axis]-1);shape=[1]*4;shape[axis]=size;f=(position-lo).astype(np.float32).reshape(shape);value=np.take(value,lo,axis=axis)*(1-f)+np.take(value,hi,axis=axis)*f
  return [value]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['unfused','gpu-resize','gpu-resize-output','gpu-attention','custom-resize','custom-resize-gpu','custom-resize-cpu'],required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'depth_anything_v2/models';prior=json.loads((source/'conversion-fp16.json').read_text());original=source/'model.onnx';assert digest(original)==prior['onnx_sha256'];report={'stage':'building','target':'rk3588','precision':'fp16','input_size':518,'variant':a.mode,'source_onnx_sha256':digest(original),'baseline_rknn_sha256':digest(source/'fp16.rknn'),'selected_as_default':False,'board_validation_passed':False,'performance_validated':False};rk=None;start=time.monotonic()
 try:
  options={} if a.mode=='custom-resize' else {'disable_rules':['fuse_matmul_softmax_matmul_to_sdpa']} if a.mode=='unfused' else {'op_target':{name+('_output_0' if a.mode=='gpu-resize-output' else ''):'gpu' for name in ['/depth_head/refinenet3/Resize','/depth_head/refinenet2/Resize','/depth_head/refinenet1/Resize','/depth_head/Resize']}}
  if a.mode=='gpu-attention':options={'op_target':{f'/blocks.{i}/attn/{op}_output_0':'gpu' for i in range(12) for op in ['MatMul','Softmax','MatMul_1']}}
  model=original
  if a.mode in ['custom-resize-gpu','custom-resize-cpu']:
   graph=onnx.load(original);sizes={'/depth_head/refinenet3/Resize':74,'/depth_head/refinenet2/Resize':148,'/depth_head/refinenet1/Resize':296,'/depth_head/Resize':518};names=[]
   for node in graph.graph.node:
    if node.name not in sizes:continue
    attrs={v.name:helper.get_attribute_value(v) for v in node.attribute};assert node.op_type=='Resize' and attrs['coordinate_transformation_mode']==b'align_corners' and attrs['mode']==b'linear';node.op_type=cstAlignCornersResize.op_type;node.domain='aa';del node.input[1:];del node.attribute[:];node.attribute.extend([helper.make_attribute('height',sizes[node.name]),helper.make_attribute('width',sizes[node.name])]);names.append(node.output[0])
   assert len(names)==4;graph.opset_import.append(helper.make_opsetid('aa',1));model=out/'candidate.onnx';onnx.save(graph,model);onnx.checker.check_model(graph);options={'op_target':{name:'gpu' if a.mode.endswith('gpu') else 'cpu' for name in names}};report['candidate_onnx_sha256']=digest(model)
  report['config_overrides']=options;rk=RKNN(verbose=True,verbose_file=str(out/'compiler.log'));assert rk.config(target_platform='rk3588',mean_values=[prior['mean']],std_values=[prior['std']],float_dtype='float16',optimization_level=3,**options) in (0,None)
  if a.mode=='custom-resize':
   class Resize:op_type='Resize'
   assert rk.reg_custom_op(Resize)==0
  if a.mode in ['custom-resize-gpu','custom-resize-cpu']:assert rk.reg_custom_op(cstAlignCornersResize)==0
  assert rk.load_onnx(model=str(model))==0;assert rk.build(do_quantization=False)==0;artifact=out/'candidate.rknn';assert rk.export_rknn(str(artifact))==0;report.update(candidate_rknn_sha256=digest(artifact),candidate_bytes=artifact.stat().st_size)
  assert rk.init_runtime(target=None)==0;opts=ort.SessionOptions();opts.intra_op_num_threads=4;session=ort.InferenceSession(str(original),sess_options=opts,providers=['CPUExecutionProvider']);records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation'];rows=[]
  for index in [0,32]:
   record=records[index];path=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(path)==record['prepared_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.asarray(prior['mean'],np.float32))/np.asarray(prior['std'],np.float32);ref=session.run(None,{'images':x.transpose(2,0,1)[None]})[0];got=np.asarray(rk.inference(inputs=[rgb[None]],data_format=['nhwc'])[0]);assert got.shape==ref.shape and np.isfinite(got).all();rows.append({'file':record['file'],'spearman':float(spearmanr(ref.ravel(),got.ravel()).statistic),'max_abs_difference':float(np.max(np.abs(ref-got)))})
  report.update(stage='simulator_checked_board_pending',simulator_sample_gate_passed=all(row['spearman']>=.99 for row in rows),simulator_rows=rows)
 except BaseException as error:report.update(stage='failed',error=repr(error));raise
 finally:
  if rk is not None:rk.release()
  report['seconds']=time.monotonic()-start;(out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

"""Equivalent FP32 graph candidates; NPU accuracy is checked separately."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import cv2,numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper
from scipy.stats import spearmanr
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def interpolation(n,m):
    # Exact align_corners geometry: each destination uses two source samples.
    # Match ONNX Runtime ALIGN_CORNERS: float32 multiply, then divide.
    position=(np.arange(m,dtype=np.float32)*np.float32(n-1))/np.float32(m-1);low=np.floor(position).astype(int);high=np.minimum(low+1,n-1);fraction=position-low;matrix=np.zeros((n,m),np.float32)
    for j in range(m):matrix[low[j],j]+=1-fraction[j];matrix[high[j],j]+=fraction[j]
    return matrix
def rewrite_resize(graph,gather=False):
    sizes={'/depth_head/refinenet3/Resize':(37,74),'/depth_head/refinenet2/Resize':(74,148),'/depth_head/refinenet1/Resize':(148,296),'/depth_head/Resize':(296,518)};nodes=[];found=[]
    for node in graph.graph.node:
        if node.name not in sizes:nodes.append(node);continue
        attrs={v.name:helper.get_attribute_value(v) for v in node.attribute};assert node.op_type=='Resize' and attrs['mode']==b'linear' and attrs['coordinate_transformation_mode']==b'align_corners';n,m=sizes[node.name];prefix='__r4_resize_'+str(n)+'_'+str(m);weights=prefix+'_weights';graph.graph.initializer.append(numpy_helper.from_array(interpolation(n,m),weights))
        if gather:
            position=(np.arange(m,dtype=np.float32)*np.float32(n-1))/np.float32(m-1);low=np.floor(position).astype(np.int64);high=np.minimum(low+1,n-1);fraction=(position-low).astype(np.float32)
            for label,values in [('low',low),('high',high)]:graph.graph.initializer.append(numpy_helper.from_array(values,prefix+'_'+label))
            source=node.input[0]
            for axis in [3,2]:
                tag=prefix+'_axis'+str(axis);shape=[1,1,1,1];shape[axis]=m
                for label,values in [('left',1-fraction),('right',fraction)]:graph.graph.initializer.append(numpy_helper.from_array(values.reshape(shape),tag+'_'+label))
                target=node.output[0] if axis==2 else tag+'_out';nodes.extend([helper.make_node('Gather',[source,prefix+'_low'],[tag+'_lo'],name=tag+'_gather_lo',axis=axis),helper.make_node('Gather',[source,prefix+'_high'],[tag+'_hi'],name=tag+'_gather_hi',axis=axis),helper.make_node('Mul',[tag+'_lo',tag+'_left'],[tag+'_weighted_lo'],name=tag+'_weight_lo'),helper.make_node('Mul',[tag+'_hi',tag+'_right'],[tag+'_weighted_hi'],name=tag+'_weight_hi'),helper.make_node('Add',[tag+'_weighted_lo',tag+'_weighted_hi'],[target],name=tag+'_sum')]);source=target
        else:
            nodes.extend([helper.make_node('MatMul',[node.input[0],weights],[prefix+'_x'],name=prefix+'_width'),helper.make_node('Transpose',[prefix+'_x'],[prefix+'_xt'],name=prefix+'_transpose_in',perm=[0,1,3,2]),helper.make_node('MatMul',[prefix+'_xt',weights],[prefix+'_y'],name=prefix+'_height'),helper.make_node('Transpose',[prefix+'_y'],[node.output[0]],name=prefix+'_transpose_out',perm=[0,1,3,2])])
        found.append(node.name)
    assert len(found)==4;del graph.graph.node[:];graph.graph.node.extend(nodes);return found
def rewrite_heads(graph,group_size=2):
    lookup={n.name:n for n in graph.graph.node};skip=set();replacement={};axes='__r4_head_axis';step='__r4_head_step';graph.graph.initializer.extend([numpy_helper.from_array(np.array([1],np.int64),axes),numpy_helper.from_array(np.array([1],np.int64),step)])
    groups=6//group_size
    for group in range(groups):
        graph.graph.initializer.extend([numpy_helper.from_array(np.array([group*group_size],np.int64),'__r4_head_start_'+str(group)),numpy_helper.from_array(np.array([group*group_size+group_size],np.int64),'__r4_head_end_'+str(group))])
    for block in range(12):
        base='/blocks.'+str(block)+'/attn/';qk=lookup[base+'MatMul'];soft=lookup[base+'Softmax'];av=lookup[base+'MatMul_1'];assert soft.input[0]==qk.output[0] and av.input[0]==soft.output[0];skip.update([qk.name,soft.name,av.name]);nodes=[];outs=[]
        for group in range(groups):
            prefix='__r4_head_'+str(block)+'_'+str(group)
            for role,source in [('q',qk.input[0]),('k',qk.input[1]),('v',av.input[1])]:nodes.append(helper.make_node('Slice',[source,'__r4_head_start_'+str(group),'__r4_head_end_'+str(group),axes,step],[prefix+'_'+role],name=prefix+'_slice_'+role))
            nodes.extend([helper.make_node('MatMul',[prefix+'_q',prefix+'_k'],[prefix+'_scores'],name=prefix+'_qk'),helper.make_node('Softmax',[prefix+'_scores'],[prefix+'_prob'],name=prefix+'_softmax',axis=-1),helper.make_node('MatMul',[prefix+'_prob',prefix+'_v'],[prefix+'_out'],name=prefix+'_av')]);outs.append(prefix+'_out')
        nodes.append(helper.make_node('Concat',outs,[av.output[0]],name='__r4_head_join_'+str(block),axis=1));replacement[av.name]=nodes
    nodes=[]
    for node in graph.graph.node:
        if node.name in replacement:nodes.extend(replacement[node.name])
        elif node.name not in skip:nodes.append(node)
    del graph.graph.node[:];graph.graph.node.extend(nodes);return list(replacement)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['resize-matmul','resize-gather','head-groups'],required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--head-group-size',type=int,choices=[1,2],default=2);parser.add_argument('--flash',action='store_true');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'depth_anything_v2/models';prior=json.loads((source/'conversion-fp16.json').read_text());original=source/'model.onnx';assert digest(original)==prior['onnx_sha256'];graph=onnx.load(original);weights={v.name:v.SerializeToString() for v in graph.graph.initializer};modified=rewrite_resize(graph,args.variant=='resize-gather') if args.variant.startswith('resize-') else rewrite_heads(graph,args.head_group_size);onnx.checker.check_model(graph);intermediate=out/'unpruned.onnx';onnx.save(graph,intermediate);candidate=out/'candidate.onnx';onnx.utils.extract_model(str(intermediate),str(candidate),[v.name for v in graph.graph.input],[v.name for v in graph.graph.output]);graph=onnx.load(candidate);current={v.name:v.SerializeToString() for v in graph.graph.initializer};assert all(current[k]==v for k,v in weights.items());intermediate.unlink()
    report={'stage':'onnx_checking','variant':args.variant,'head_group_size':args.head_group_size,'enable_flash_attention':args.flash,'target':'rk3588','precision':'fp16','input_size':518,'source_onnx_sha256':digest(original),'baseline_rknn_sha256':digest(source/'fp16.rknn'),'candidate_onnx_sha256':digest(candidate),'learned_weights_unchanged':True,'rewritten_nodes':modified,'selected_as_default':False,'board_validation_passed':False,'performance_validated':False,'started':time.time()};rk=None;start=time.monotonic()
    try:
        options=ort.SessionOptions();options.intra_op_num_threads=4;a=ort.InferenceSession(str(original),sess_options=options,providers=['CPUExecutionProvider']);b=ort.InferenceSession(str(candidate),sess_options=options,providers=['CPUExecutionProvider']);records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation'];rows=[]
        for index in [0,32]:
            record=records[index];path=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(path)==record['prepared_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.asarray(prior['mean'],np.float32))/np.asarray(prior['std'],np.float32);x=x.transpose(2,0,1)[None];ref=a.run(None,{'images':x})[0];got=b.run(None,{'images':x})[0];assert got.shape==ref.shape and np.isfinite(got).all();np.testing.assert_allclose(ref,got,atol=1e-4,rtol=1e-4);rows.append({'file':record['file'],'max_abs_difference':float(np.max(np.abs(ref-got))),'spearman':float(spearmanr(ref.ravel(),got.ravel()).statistic)})
        del a,b;report['onnx_equivalence_rows']=rows;rk=RKNN(verbose=True,verbose_file=str(out/'compiler.log'));assert rk.config(target_platform='rk3588',mean_values=[prior['mean']],std_values=[prior['std']],float_dtype='float16',optimization_level=3,enable_flash_attention=args.flash) in (0,None);assert rk.load_onnx(model=str(candidate))==0;assert rk.build(do_quantization=False)==0;artifact=out/'candidate.rknn';assert rk.export_rknn(str(artifact))==0;report.update(stage='converted_not_board_validated',candidate_rknn_sha256=digest(artifact),candidate_bytes=artifact.stat().st_size,onnx_equivalence_passed=True)
    except BaseException as error:report.update(stage='failed',error=repr(error));raise
    finally:
        if rk is not None:rk.release()
        report['seconds']=time.monotonic()-start;(out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

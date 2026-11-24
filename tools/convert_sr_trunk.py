"""Extract the unchanged learned SR convolution trunk; validate the moved tail algebra."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import onnx,onnxruntime as ort
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    folder=ROOT/'real_esrgan/experiments/cpu-tail-v1';folder.mkdir(parents=True,exist_ok=False);source=ROOT/'real_esrgan/models/model.onnx';graph=onnx.load(source);nodes=[n for n in graph.graph.node if n.op_type=='DepthToSpace'];assert len(nodes)==1
    shuffle=nodes[0];attrs={a.name:onnx.helper.get_attribute_value(a) for a in shuffle.attribute};assert attrs.get('blocksize')==4 and attrs.get('mode')==b'CRD',attrs
    target=folder/'trunk.onnx';onnx.utils.extract_model(str(source),str(target),[graph.graph.input[0].name],[shuffle.input[0]]);trunk=onnx.load(target);onnx.checker.check_model(trunk)
    original={v.name:v.SerializeToString() for v in graph.graph.initializer};assert all(original[v.name]==v.SerializeToString() for v in trunk.graph.initializer)
    options=ort.SessionOptions();options.intra_op_num_threads=4;full=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider']);part=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider']);checks=[]
    for seed in [17,23]:
        x=np.random.default_rng(seed).random((1,3,256,256),dtype=np.float32);a=full.run(None,{'images':x})[0];v=part.run(None,{'images':x})[0];assert v.shape==(1,48,256,256)
        b=v.reshape(1,3,4,4,256,256).transpose(0,1,4,2,5,3).reshape(1,3,1024,1024)+x.repeat(4,2).repeat(4,3);difference=float(np.max(np.abs(a-b)));np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6);checks.append({'seed':seed,'onnx_max_abs_difference':difference})
    rk=RKNN(verbose=False);start=time.monotonic()
    try:
        assert rk.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[255,255,255]]) in (0,None)
        assert rk.load_onnx(model=str(target))==0;assert rk.build(do_quantization=False)==0;artifact=folder/'fp16-trunk.rknn';assert rk.export_rknn(str(artifact))==0
        receipt={'stage':'converted_not_board_validated','source_onnx_sha256':digest(source),'trunk_onnx_sha256':digest(target),'rknn_sha256':digest(artifact),'learned_initializers_unchanged':True,'onnx_equivalence':checks,'precision':'fp16','input':[1,3,256,256],'raw_output':[1,48,256,256],'reconstructed_output':[1,3,1024,1024],'seconds':time.monotonic()-start,'selected_as_default':False};(folder/'conversion.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
    finally:rk.release()
if __name__=='__main__':main()

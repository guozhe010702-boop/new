"""Quality-gated automatic mixed-precision candidate; default deployment stays FP16."""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np,onnxruntime as ort
from scipy.stats import spearmanr
from rknn.api import RKNN
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'depth_anything_v2/models';prior=json.loads((source/'conversion-fp16.json').read_text());onnx=source/'model.onnx';assert digest(onnx)==prior['onnx_sha256'];dataset=ROOT/'data/calibration-depth_anything_v2.txt';records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2'];paths=dataset.read_text().splitlines();assert len(paths)==len(records['calibration'])==194
    for name,record in zip(paths,records['calibration']):assert digest(Path(name))==record['prepared_sha256']
    assert not ({record['source_sha256'] for record in records['calibration']} & {record['source_sha256'] for record in records['validation']})
    report={'stage':'building','target':'rk3588','precision':'mixed-int8-fp16','input_size':518,'source_onnx_sha256':digest(onnx),'baseline_rknn_sha256':digest(source/'fp16.rknn'),'calibration_images':194,'calibration_list_sha256':digest(dataset),'auto_hybrid_cos_thresh':.995,'selected_as_default':False,'board_validation_passed':False,'performance_validated':False,'started':time.time()};start=time.monotonic();rk=None;receipt=out/'probe.json';receipt.write_text(json.dumps(report,indent=2)+'\n')
    try:
        rk=RKNN(verbose=False,verbose_file=str(out/'compiler.log'));assert rk.config(target_platform='rk3588',mean_values=[prior['mean']],std_values=[prior['std']],quantized_dtype='w8a8',quantized_algorithm='normal',auto_hybrid_cos_thresh=.995) in (0,None);assert rk.load_onnx(model=str(onnx))==0;assert rk.build(do_quantization=True,dataset=str(dataset),auto_hybrid=True)==0;artifact=out/'candidate.rknn';assert rk.export_rknn(str(artifact))==0;report.update(stage='converted_not_board_validated',candidate_rknn_sha256=digest(artifact),candidate_bytes=artifact.stat().st_size,build_seconds=time.monotonic()-start);receipt.write_text(json.dumps(report,indent=2)+'\n')
        assert rk.init_runtime(target=None)==0;options=ort.SessionOptions();options.intra_op_num_threads=4;reference=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider']);rows=[]
        for index in [0,32]:
            record=records['validation'][index];path=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(path)==record['prepared_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.array(prior['mean'],np.float32))/np.array(prior['std'],np.float32);ref=reference.run(None,{'images':x.transpose(2,0,1)[None]})[0];outputs=rk.inference(inputs=[rgb[None]],data_format=['nhwc']);assert outputs is not None and len(outputs)==1;got=np.asarray(outputs[0]);assert got.shape==ref.shape and np.isfinite(got).all();rows.append({'file':record['file'],'onnx_spearman':float(spearmanr(ref.ravel(),got.ravel()).statistic),'max_abs_difference':float(np.max(np.abs(ref-got)))});report['simulator_rows']=rows
        report.update(stage='simulator_checked_board_pending',simulator_sample_gate_passed=all(row['onnx_spearman']>=.99 for row in rows))
    except BaseException as error:report.update(stage='failed',error=repr(error));raise
    finally:
        if rk is not None:rk.release()
        report['seconds']=time.monotonic()-start;receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

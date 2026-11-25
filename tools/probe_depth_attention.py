"""Isolated FP16 Flash Attention candidate; never changes the deployed model.

Run in the pinned RKNN Toolkit2 2.3.2 conversion image. Simulator checks are
preliminary numerical checks, not evidence of RK3588 performance or accuracy.
"""
from pathlib import Path
import argparse,hashlib,json,time,inspect
import cv2,numpy as np,onnxruntime as ort
from scipy.stats import spearmanr
from rknn.api import RKNN

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--simulator-samples',type=int,choices=[0,2,5],default=2);args=parser.parse_args();folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    source=ROOT/'depth_anything_v2/models';original=json.loads((source/'conversion-fp16.json').read_text());onnx=source/'model.onnx';baseline=source/'fp16.rknn'
    assert digest(onnx)==original['onnx_sha256'],'Original ONNX changed';assert digest(baseline)==original['rknn_sha256'],'Original deployed RKNN changed';assert 'enable_flash_attention' in inspect.signature(RKNN.config).parameters
    report={'stage':'building','target':'rk3588','precision':'fp16','input_size':518,'changed_setting':{'enable_flash_attention':True},'source_onnx_sha256':digest(onnx),'baseline_rknn_sha256':digest(baseline),'mean':original['mean'],'std':original['std'],'selected_as_default':False,'board_validation_passed':False,'performance_validated':False,'started':time.time()};receipt=folder/'probe.json';receipt.write_text(json.dumps(report,indent=2)+'\n');rk=None;start=time.monotonic()
    try:
        rk=RKNN(verbose=True,verbose_file=str(folder/'compiler.log'));assert rk.config(target_platform='rk3588',mean_values=[original['mean']],std_values=[original['std']],float_dtype='float16',optimization_level=3,enable_flash_attention=True) in (0,None)
        assert rk.load_onnx(model=str(onnx))==0;assert rk.build(do_quantization=False)==0
        artifact=folder/'fp16-flash.rknn';assert rk.export_rknn(str(artifact))==0;report.update(stage='converted_not_board_validated',candidate_rknn_sha256=digest(artifact),candidate_bytes=artifact.stat().st_size,baseline_bytes=baseline.stat().st_size,byte_identical_to_baseline=digest(artifact)==digest(baseline),build_seconds=time.monotonic()-start);receipt.write_text(json.dumps(report,indent=2)+'\n')
        if report['byte_identical_to_baseline']:
            report['stage']='byte_identical_no_new_candidate';return
        if artifact.stat().st_size>512*1024**2:
            report.update(stage='oversize_candidate_not_evaluated',note='Artifact exceeds the 512MiB experiment budget; no runtime or board test started.');return
        if not args.simulator_samples:return
        assert rk.init_runtime(target=None)==0
        options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider']);records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation'];indices=[0,32] if args.simulator_samples==2 else [0,16,32,40,49];rows=[]
        for index in indices:
            record=records[index];path=ROOT/'data/prepared/validation/depth_anything_v2'/record['file'];assert digest(path)==record['prepared_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.asarray(original['mean'],np.float32))/np.asarray(original['std'],np.float32);reference=session.run(None,{'images':x.transpose(2,0,1)[None]})[0]
            before=time.monotonic();outputs=rk.inference(inputs=[rgb[None]],data_format=['nhwc']);assert outputs is not None and len(outputs)==1;candidate=np.asarray(outputs[0]);assert candidate.shape==reference.shape and np.isfinite(candidate).all();np.save(folder/(path.stem+'-simulator.npy'),candidate)
            row={'file':record['file'],'prepared_sha256':record['prepared_sha256'],'simulator_seconds':time.monotonic()-before,'onnx_spearman':float(spearmanr(reference.ravel(),candidate.ravel()).statistic),'onnx_max_abs_difference':float(np.max(np.abs(reference-candidate)))}
            previous=ROOT/'results/parity-depth-fp16'/(path.stem+'-0.f32')
            if previous.exists():
                old=np.fromfile(previous,np.float32).reshape(reference.shape);row['prior_board_fp16_spearman']=float(spearmanr(old.ravel(),candidate.ravel()).statistic)
            rows.append(row);report['simulator_rows']=rows;receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
        report.update(stage='simulator_checked_board_pending',simulator_samples=len(rows),simulator_sample_gate_passed=all(row['onnx_spearman']>=.99 and row.get('prior_board_fp16_spearman',1)>=.99 for row in rows),scope='Simulator sample parity only; must evaluate 50 images and benchmark actual RK3588 hardware before adoption.')
    except BaseException as error:report.update(stage='failed',error=repr(error));raise
    finally:
        if rk is not None:rk.release()
        report['seconds']=time.monotonic()-start;receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

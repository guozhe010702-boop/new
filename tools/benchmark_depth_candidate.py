"""Bounded RK3588 candidate A/B profiling without camera or display changes.

Uses existing npu_profile/model_batch programs. It never selects a default
model. A promising candidate still requires full quality and live validation.
"""
from pathlib import Path
import argparse,array,csv,fcntl,hashlib,json,math,os,platform,statistics,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    result={}
    for name in ['/sys/class/devfreq/fdab0000.npu/cur_freq','/sys/class/devfreq/dmc/cur_freq','/sys/class/thermal/thermal_zone0/temp']:
        path=Path(name)
        if path.exists():result[name]=path.read_text().strip()
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--runs',type=int,default=12);parser.add_argument('--skip-quality-export',action='store_true');parser.add_argument('--allow-mixed-experiment',action='store_true');args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine()!='aarch64' or os.geteuid()!=0:parser.error('Run as root on the RK3588 ARM64 board')
    if not 4<=args.runs<=30:parser.error('runs must be 4..30')
    candidate=args.candidate.resolve();proof=json.loads(candidate.with_name('probe.json').read_text());baseline=ROOT/'depth_anything_v2/models/fp16.rknn';assert proof['target']=='rk3588' and proof['input_size']==518
    assert proof['precision']=='fp16' or (args.allow_mixed_experiment and proof['precision']=='mixed-int8-fp16'), 'Mixed precision requires explicit experimental opt-in'
    assert proof.get('simulator_sample_gate_passed') is True or (proof.get('onnx_equivalence_passed') is True and proof.get('learned_weights_unchanged') is True),'Candidate lacks preliminary numerical checks';assert digest(candidate)==proof['candidate_rknn_sha256'] and digest(baseline)==proof['baseline_rknn_sha256']
    records=json.loads((ROOT/'data/dataset.json').read_text())['depth_anything_v2']['validation'];prepared=ROOT/'data/prepared/validation/depth_anything_v2';image=prepared/records[0]['file'];assert digest(image)==records[0]['prepared_sha256'];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();out=args.output.resolve()
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib'));env.pop('LD_PRELOAD',None)
    with open('/tmp/imx415-live-mpp-bench.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True,exist_ok=False);report={'stage':'running','started':time.time(),'boot_id':boot,'baseline_sha256':digest(baseline),'candidate_sha256':digest(candidate),'core_mask':7,'selected_as_default':False,'board_quality_validated':False,'live_pipeline_validated':False,'cases':[]}
        try:
            for name,model in [('a1',baseline),('b1',candidate),('b2',candidate),('a2',baseline)]:
                report['current']=name;(out/'job.json').write_text(json.dumps(report,indent=2));before=snapshot();command=[str(ROOT/'build/npu_profile'),str(model),str(image),str(out/name),'7','0',str(args.runs),'standard','3']
                with (out/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100,env=env)
                rows=[row for row in csv.DictReader((out/name/'timings.csv').open()) if row['warmup']=='0'];metrics={key:statistics.median(float(row[key]) for row in rows) for key in rows[0] if key not in ['iteration','warmup']};after=snapshot();assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot;assert int(after['/sys/class/thermal/thermal_zone0/temp'])<65000,'Conservative thermal budget exceeded'
                data=array.array('f');data.frombytes((out/name/'output-0.f32').read_bytes());assert data.itemsize==4 and len(data)==518*518 and all(math.isfinite(v) for v in data),'Invalid depth output';case={'name':name,'before':before,'after':after,'medians':metrics};report['cases'].append(case);print(json.dumps(case),flush=True)
            a=statistics.mean(case['medians']['total_ms'] for case in report['cases'] if case['name'].startswith('a'));b=statistics.mean(case['medians']['total_ms'] for case in report['cases'] if case['name'].startswith('b'));report.update(baseline_total_ms=a,candidate_total_ms=b,call_reduction_percent=(1-b/a)*100,worth_full_quality_check=b<a*.95)
            for label,model in [('baseline-layers',baseline),('candidate-layers',candidate)]:
                with (out/(label+'.log')).open('w') as log:subprocess.run([str(ROOT/'build/npu_profile'),str(model),str(image),str(out/label),'7','8','1','standard','1'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30,env=env)
            if report['worth_full_quality_check'] and not args.skip_quality_export:
                assert len(records)==50
                for record in records:assert digest(prepared/record['file'])==record['prepared_sha256']
                with (out/'quality-export.log').open('w') as log:subprocess.run([str(ROOT/'build/model_batch'),str(candidate),str(prepared),str(out/'quality-tensors')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,env=env)
                report.update(stage='quality_tensors_ready_for_host_check',quality_tensor_files=len(list((out/'quality-tensors').glob('*.f32'))))
                assert report['quality_tensor_files']==50
            else:report['stage']='profiled_not_adopted'
        except BaseException as error:report.update(stage='failed',error=repr(error));raise
        finally:report['finished']=time.time();(out/'job.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
if __name__=='__main__':main()

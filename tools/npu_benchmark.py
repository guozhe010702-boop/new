"""Bounded, camera-free NPU stage/graph profiling on the RK3588 board."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,statistics,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def snapshot():
    result={}
    for name in ['/sys/class/devfreq/fdab0000.npu/cur_freq','/sys/class/devfreq/dmc/cur_freq','/sys/class/thermal/thermal_zone0/temp','/sys/kernel/debug/rknpu/load']:
        path=Path(name)
        if path.exists():result[name]=path.read_text().strip()
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',choices=['real_esrgan','depth_anything_v2'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=12);p.add_argument('--image',type=Path,help='already prepared 256x256 SR or 518x518 depth input');a=p.parse_args()
    if not 1<=a.iterations<=100:p.error('iterations must be 1..100')
    a.output=a.output.resolve()
    with open('/tmp/imx415-live-mpp-bench.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();state={'stage':'running','boot_id':boot,'project':a.project,'started':time.time(),'cases':[]}
        cases=[('baseline-a',7,0,'standard'),('prealloc',7,0,'prealloc'),('sram',7,2048,'prealloc'),('baseline-b',7,0,'standard'),('single-core',1,0,'standard'),('layers',7,8,'standard')]
        if a.project=='real_esrgan' and (ROOT/a.project/'models/fp16-trunk.rknn').is_file():cases.append(('cpu-tail-native',7,0,'cpu-tail-native'))
        try:
            for name,mask,flags,mode in cases:
                state['current']=name;(a.output/'job.json').write_text(json.dumps(state,indent=2));folder=a.output/name;model=ROOT/a.project/'models'/('fp16-trunk.rknn' if mode=='cpu-tail-native' else 'fp16.rknn');image=a.image or ROOT/'data/prepared/validation'/a.project/'0000.png';before=snapshot();command=[str(ROOT/'build/npu_profile'),str(model),str(image),str(folder),str(mask),str(flags),str(1 if flags==8 else a.iterations),mode,str(1 if flags==8 else 3)]
                with (a.output/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100)
                rows=[x for x in csv.DictReader((folder/'timings.csv').open()) if x['warmup']=='0'];record={'name':name,'before':before,'after':snapshot(),'median':{key:statistics.median(float(v[key]) for v in rows) for key in rows[0] if key not in ['iteration','warmup']},'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()};state['cases'].append(record);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'board reboot';assert int(record['after']['/sys/class/thermal/thermal_zone0/temp'])<65000,'thermal budget exceeded';print(json.dumps(record),flush=True)
            state['stage']='complete'
        except BaseException as error:state.update(stage='failed',error=repr(error));raise
        finally:state['finished']=time.time();(a.output/'job.json').write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()

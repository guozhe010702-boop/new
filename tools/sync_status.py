"""Read-only board status download to the Windows checkout. No camera operations."""
import argparse,base64,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='root@192.168.2.221');p.add_argument('--identity',default=str(Path.home()/'.ssh/evm_rk3588'));a=p.parse_args()
    code="""from pathlib import Path
import json
r=Path('/userdata/av_dev/advanced algorithms')
paths=['results/qualification-20261124/job.json','results/lifecycle/result.json','results/4k-x4.tif.json']+[n+'/recommended.json' for n in ['yolo11_bytetrack','real_esrgan','depth_anything_v2']]
data={}
for name in paths:
 p=r/name
 if p.exists():data[name]=json.loads(p.read_text())
job=data.get('results/qualification-20261124/job.json',{})
if job:
 job['observed_boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 job['observed_process_running']=Path('/proc/'+str(job.get('pid'))).exists()
if job.get('current'):
 p=r/'results/qualification-20261124'/job['current']/'status.json'
 if p.exists():data['_live']=json.loads(p.read_text())
print(json.dumps(data))
"""
    command='python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+base64.b64encode(code.encode()).decode()+"'))")
    raw=subprocess.check_output(['ssh','-i',a.identity,'-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,command],text=True);data=json.loads(raw)
    for rel,value in data.items():
        if rel.startswith('_'):continue
        path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    job=data.get('results/qualification-20261124/job.json',{});print(json.dumps({'stage':job.get('stage'),'current':job.get('current'),'finished_cases':len(job.get('cases',[])),'same_boot':job.get('observed_boot_id')==job.get('boot_id'),'suite_process_running':job.get('observed_process_running'),'live':data.get('_live')},indent=2))
if __name__=='__main__':main()

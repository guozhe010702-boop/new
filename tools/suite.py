#!/usr/bin/env python3
"""Finite, serial hardware qualification. Every duration and failure is recorded."""
import argparse,csv,json,os,signal,subprocess,sys,time
from pathlib import Path
from control import ROOT,atomic,sample
def analysis(folder,warmup=5):
    summary=json.loads((folder/'summary.json').read_text());rows=list(csv.DictReader((folder/'frames.csv').open()))
    rows=[r for r in rows if r['outcome']=='processed'];result=dict(summary)
    if len(rows)>=2:
        base=int(rows[0]['captured_us']);rows=[r for r in rows if int(r['captured_us'])-base>=warmup*1000000]
        if len(rows)>=2:result['steady_processed_fps']=(len(rows)-1)/((int(rows[-1]['captured_us'])-int(rows[0]['captured_us']))/1000000)
    resource=[json.loads(r) for r in (folder/'telemetry.jsonl').read_text().splitlines()]
    if resource:
        result['max_temperature_c']=max((r.get('temperature_c') or 0) for r in resource);rss=[r.get('VmRSS',0)/1024 for r in resource];result['rss_max_mib']=max(rss);result['fds_min']=min(r.get('fd_count',0) for r in resource);result['fds_max']=max(r.get('fd_count',0) for r in resource)
        good=[r for r in resource if 'cpu_ticks' in r]
        if len(good)>1:result['native_cpu_one_core_percent']=100*(good[-1]['cpu_ticks']-good[0]['cpu_ticks'])/os.sysconf('SC_CLK_TCK')/(good[-1]['monotonic']-good[0]['monotonic'])
        if len(rss)>=600:result['last_10min_rss_delta_mib']=rss[-1]-min(rss[-600:])
    result['restoration_pass']=json.loads((folder/'restoration.json').read_text()).get('restored',False);atomic(folder/'analysis.json',result);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--round-seconds',type=int,default=300);p.add_argument('--endurance-seconds',type=int,default=1800);p.add_argument('--scan-seconds',type=int,default=30);p.add_argument('--project',action='append');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();state={'stage':'running','pid':os.getpid(),'boot_id':boot,'cases':[],'started_at':time.time(),'round_seconds':a.round_seconds,'endurance_seconds':a.endurance_seconds};atomic(a.output/'job.json',state)
    child=None;interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(project,name,seconds,precision,workers,mask,extra=None):
        nonlocal child
        if interrupted:raise KeyboardInterrupt
        path=a.output/(project+'-'+name);command=[sys.executable,str(ROOT/'tools/control.py'),'run','--project',project,'--precision',precision,'--workers',str(workers),'--mask',str(mask),'--hdmi','1','--seconds',str(seconds),'--output',str(path),'--save-every','0']+(extra or [])
        state.update(current=path.name,current_started=time.time());atomic(a.output/'job.json',state)
        with (a.output/(path.name+'.log')).open('w') as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,AA_QUALIFICATION='1'));rc=child.wait()
        result={'name':path.name,'returncode':rc,'seconds_requested':seconds,'precision':precision,'workers':workers,'mask':mask}
        if (path/'summary.json').exists():result['analysis']=analysis(path)
        result['pass']=rc==0 and not result.get('analysis',{}).get('error',True) and result.get('analysis',{}).get('restoration_pass',False) and result.get('analysis',{}).get('seconds',0)>=seconds*.98 and result.get('analysis',{}).get('max_temperature_c',99)<72 and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
        state['cases'].append(result);atomic(a.output/'job.json',state)
        if not result['pass']:raise RuntimeError('Failed hardware case '+path.name)
        return result
    try:
        for project in a.project or ['yolo11_bytetrack','real_esrgan','depth_anything_v2']:
            quality=json.loads((ROOT/'results'/(project+'-quality.json')).read_text());assert quality.get('fp16_reference_pass'),'FP16 baseline not validated';precision='int8' if quality.get('int8_quality_pass') else 'fp16';scan=[]
            for workers,mask,label in [(1,1,'single-core'),(1,7,'three-core'),(3,7,'three-context')]:scan.append(run(project,'scan-'+label,a.scan_seconds,precision,workers,mask))
            best=max(r['analysis'].get('steady_processed_fps',0) for r in scan)
            eligible=[r for r in scan if r['analysis'].get('steady_processed_fps',0)>=best*.95]
            chosen=min(eligible,key=lambda r:(r['analysis']['age_ms']['p95'],r['workers']));workers,mask=chosen['workers'],chosen['mask']
            for i in range(3):run(project,'repeat-'+str(i+1),a.round_seconds,precision,workers,mask)
            run(project,'stall',30,precision,workers,mask,['--inject-stall-ms','300'])
            final=run(project,'endurance',a.endurance_seconds,precision,workers,mask);profile={'precision':precision,'workers':workers,'mask':mask,'qualification':str(a.output),'endurance_seconds':a.endurance_seconds,'stability_validated':a.endurance_seconds>=1800 and a.round_seconds>=300,'target_fps':20 if project=='depth_anything_v2' else 25 if project=='real_esrgan' else 30,'measured_fps':final['analysis'].get('steady_processed_fps',0)}
            profile['performance_target_met']=profile['measured_fps'] >= (29 if project=='yolo11_bytetrack' else profile['target_fps'])
            atomic(ROOT/project/'recommended.json',profile)
        state.update(stage='complete',finished_at=time.time())
    except BaseException as e:
        state.update(stage='interrupted' if interrupted else 'failed',error=str(e),finished_at=time.time());raise
    finally:atomic(a.output/'job.json',state)
if __name__=='__main__':main()

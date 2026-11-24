#!/usr/bin/env python3
"""Independent manual supervision. No boot service and no global runtime replacement."""
import argparse,datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys,time,shlex,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'state/active.json'
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def cmd(args,check=False):return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=check)
def alive(pid):
    try:return Path('/proc/'+str(pid)+'/exe').resolve()==ROOT/'build/advanced_vision'
    except OSError:return False
def restore():
    if not STATE.exists():return
    s=json.loads(STATE.read_text())
    if alive(s.get('native_pid')):raise RuntimeError('Native process still active; stop first')
    if s.get('restored'):return
    output=Path(s['output']);ok=True
    output.mkdir(parents=True,exist_ok=True)
    telemetry=Path(s.get('telemetry_file','/nonexistent'))
    if telemetry.parent==ROOT/'state' and telemetry.name.startswith('telemetry-') and telemetry.suffix=='.jsonl' and telemetry.exists():
        if not (output/'telemetry.jsonl').exists():shutil.copyfile(telemetry,output/'telemetry.jsonl')
        telemetry.unlink()
    if not (output/'native.log').exists() and (ROOT/'state/native-startup.log').exists():shutil.copyfile(ROOT/'state/native-startup.log',output/'native.log')
    if s.get('format'):
        result=cmd(['v4l2-ctl','-d','/dev/video44','--set-fmt-video='+s['format']]);(output/'restore-camera.log').write_text(result.stdout);ok&=result.returncode==0
    if s.get('desktop_active'):
        result=cmd(['systemctl','start',s['desktop']]);(output/'restore-desktop.log').write_text(result.stdout);ok&=result.returncode==0
        env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/run/user/1000/gdm/Xauthority')
        for _ in range(20):
            result=subprocess.run(['xrandr']+s.get('layout',[]),env=env,capture_output=True,text=True)
            if result.returncode==0:break
            time.sleep(.5)
        ok&=result.returncode==0
    s.update(restored=bool(ok),restored_at=time.time());atomic(STATE,s);atomic(output/'restoration.json',s)
def sample(pid):
    result={'time':time.time(),'monotonic':time.monotonic()}
    try:
        status=Path('/proc/'+str(pid)+'/status').read_text()
        for key in ('VmRSS','Threads'):
            m=re.search(r'^'+key+r':\s*(\d+)',status,re.M)
            if m:result[key]=int(m[1])
        stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split();result['cpu_ticks']=int(stat[11])+int(stat[12]);result['fd_count']=len(list(Path('/proc/'+str(pid)+'/fd').iterdir()))
    except OSError:pass
    values=[]
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:values.append(float(p.read_text())/1000)
        except (OSError,ValueError):pass
    result['temperature_c']=max(values) if values else None
    p=Path('/sys/kernel/debug/rknpu/load');result['npu_load']=p.read_text().strip() if p.exists() else None
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','start','stop','status','restore']);p.add_argument('--project',choices=['capture','yolo11_bytetrack','depth_anything_v2','real_esrgan'],default='yolo11_bytetrack');p.add_argument('--precision',choices=['fp16','int8'],default='fp16');p.add_argument('--seconds',type=int,default=0);p.add_argument('--hdmi',type=int,choices=[0,1],default=1);p.add_argument('--output',type=Path);p.add_argument('--image');p.add_argument('--video');p.add_argument('--workers',type=int,default=1);p.add_argument('--mask',type=int,default=7);p.add_argument('--save-every',type=int,default=300);p.add_argument('--capture-memory',choices=['mmap','dmabuf'],default='mmap');p.add_argument('--inject-stall-ms',type=int,default=0);p.add_argument('--line',type=float,default=.5);p.add_argument('--class',dest='target_class',type=int,default=0);p.add_argument('--inject-display-error-after',type=int,default=0);p.add_argument('--render',choices=['scalar','vector'],default='vector');p.add_argument('--sr-tail',choices=['npu','cpu']);p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080);p.add_argument('--roi-x',type=int,default=-1);p.add_argument('--roi-y',type=int,default=-1);p.set_defaults(precision=None,workers=None,mask=None);a=p.parse_args();profile_file=ROOT/a.project/'recommended.json';profile=json.loads(profile_file.read_text()) if profile_file.exists() else {};a.precision=a.precision or profile.get('precision','fp16');a.workers=a.workers or profile.get('workers',1);a.mask=a.mask or profile.get('mask',7);a.sr_tail=a.sr_tail or profile.get('sr_tail','npu')
    if a.sr_tail=='cpu' and (a.project!='real_esrgan' or a.precision!='fp16'):p.error('--sr-tail cpu requires real_esrgan fp16')
    if a.action=='restore':restore();return 0
    if a.action=='status':
        s=json.loads(STATE.read_text()) if STATE.exists() else {};s['native_running']=alive(s.get('native_pid'))
        if s.get('output'):
            status=Path(s['output'])/'status.json'
            if status.exists():s['metrics']=json.loads(status.read_text());s['metrics_stale']=not s['native_running'] or time.time()-status.stat().st_mtime>3
        latest=ROOT/'state/telemetry-latest.json'
        if latest.exists():
            t=json.loads(latest.read_text())
            if t.get('native_pid')==s.get('native_pid'):s['hardware']=t
        print(json.dumps(s,indent=2));return 0
    if a.action=='stop':
        if os.environ.get('AA_FINAL_CHECK')!='1' and cmd(['systemctl','is-active','--quiet','advanced-final-check.service']).returncode==0:
            cmd(['systemctl','stop','advanced-final-check.service'],check=True);return 0
        if os.environ.get('AA_QUALIFICATION')!='1' and cmd(['systemctl','is-active','--quiet','advanced-qualification.service']).returncode==0:
            cmd(['systemctl','stop','advanced-qualification.service'],check=True);return 0
        if cmd(['systemctl','is-active','--quiet','advanced-algorithms.service']).returncode==0:
            cmd(['systemctl','stop','advanced-algorithms.service'],check=True);return 0
        if not STATE.exists():return 0
        s=json.loads(STATE.read_text());pid=s.get('supervisor_pid');command=Path('/proc/'+str(pid)+'/cmdline')
        if s.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and command.exists() and str(Path(__file__).resolve()).encode() in command.read_bytes():
            os.kill(pid,signal.SIGTERM)
            for _ in range(100):
                if not alive(s.get('native_pid')) and json.loads(STATE.read_text()).get('restored'):return 0
                time.sleep(.2)
            raise RuntimeError('Stop timed out; inspect logs')
        if alive(s.get('native_pid')):raise RuntimeError('Supervisor missing; inspect native ownership before termination')
        restore();return 0
    if a.action=='start':
        if os.environ.get('AA_FINAL_CHECK')!='1' and cmd(['systemctl','is-active','--quiet','advanced-final-check.service']).returncode==0:raise RuntimeError('Final qualification check owns the camera')
        if cmd(['systemctl','is-active','--quiet','advanced-qualification.service']).returncode==0:raise RuntimeError('Qualification suite owns the camera')
        if cmd(['systemctl','is-active','--quiet','advanced-algorithms.service']).returncode==0:raise RuntimeError('Algorithm already active')
        post='/usr/bin/python3 '+shlex.quote(str(Path(__file__).resolve()))+' restore'
        command=['systemd-run','--unit=advanced-algorithms.service','--collect','--property=KillMode=mixed','--property=TimeoutStopSec=20','--property=WorkingDirectory='+str(ROOT),'--property=ExecStopPost='+post,sys.executable,str(Path(__file__).resolve()),'run']+sys.argv[2:]
        print(cmd(command,check=True).stdout);return 0
    if os.geteuid()!=0:raise RuntimeError('Run as root')
    if os.environ.get('AA_FINAL_CHECK')!='1' and cmd(['systemctl','is-active','--quiet','advanced-final-check.service']).returncode==0:raise RuntimeError('Final qualification check owns the camera')
    if os.environ.get('AA_QUALIFICATION')!='1' and cmd(['systemctl','is-active','--quiet','advanced-qualification.service']).returncode==0:raise RuntimeError('Qualification suite owns the camera')
    lock=Path('/tmp/imx415-live-mpp-bench.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        if alive(previous.get('native_pid')):raise RuntimeError('Previous native process still active')
        if not previous.get('restored'):restore()
    if not (a.image or a.video):
        owners=cmd(['lsof','-t','/dev/video44'])
        if owners.stdout.strip():raise RuntimeError('Camera occupied: '+owners.stdout.strip())
    output=a.output or ROOT/'results'/(a.project+'-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output=output.resolve()
    if output.exists():raise RuntimeError('Output must be new')
    # Native owns mkdir; supervisor logs wait until its output directory exists.
    f=cmd(['v4l2-ctl','-d','/dev/video44','--get-fmt-video'],check=True).stdout;size=re.search(r'Width/Height\s*:\s*(\d+)/(\d+)',f);pixel=re.search(r"Pixel Format\s*:\s*'([^']+)'",f)
    if not size or not pixel:raise RuntimeError('Cannot save camera format')
    desktop=cmd(['systemctl','show','display-manager','-p','Id','--value']).stdout.strip();active=bool(desktop) and cmd(['systemctl','is-active','--quiet',desktop]).returncode==0
    env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/run/user/1000/gdm/Xauthority');xr=subprocess.run(['xrandr','--query'],env=env,capture_output=True,text=True);layout=[]
    for line in xr.stdout.splitlines():
        m=re.match(r'(\S+) connected (primary )?(\d+)x(\d+)\+(\d+)\+(\d+)',line)
        if m:layout+=['--output',m[1],'--mode',m[3]+'x'+m[4],'--pos',m[5]+'x'+m[6]]+(['--primary'] if m[2] else [])
    s={'output':str(output),'format':'width='+size[1]+',height='+size[2]+',pixelformat='+pixel[1],'desktop':desktop,'desktop_active':active and bool(a.hdmi),'layout':layout,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'supervisor_pid':os.getpid(),'restored':False};atomic(STATE,s)
    command=[str(ROOT/'build/advanced_vision'),'--project',a.project,'--output',str(output),'--model',str(ROOT/a.project/'models'/(a.precision+('-trunk.rknn' if a.sr_tail=='cpu' else '.rknn'))),'--seconds',str(a.seconds),'--hdmi',str(a.hdmi),'--workers',str(a.workers),'--mask',str(a.mask),'--save-every',str(a.save_every),'--capture-memory',a.capture_memory,'--inject-stall-ms',str(a.inject_stall_ms),'--line',str(a.line),'--class',str(a.target_class)]
    command+=['--inject-display-error-after',str(a.inject_display_error_after)]
    if os.environ.get('AA_RENDER_OPTION_OMIT')!='1':command+=['--render',a.render,'--sr-tail',a.sr_tail]
    command+=['--width',str(a.width),'--height',str(a.height),'--roi-x',str(a.roi_x),'--roi-y',str(a.roi_y)]
    if a.image:command+=['--image',str(Path(a.image).resolve())]
    if a.video:command+=['--video',str(Path(a.video).resolve())]
    child=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if s['desktop_active']:cmd(['systemctl','stop',desktop],check=True)
        native_env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib'));native_env.pop('LD_PRELOAD',None)
        with (ROOT/'state/native-startup.log').open('w') as native:
            child=subprocess.Popen(command,env=native_env,stdout=native,stderr=subprocess.STDOUT);s['native_pid']=child.pid;atomic(STATE,s)
            terminate_at=None;previous=None;previous_sync=0
            with tempfile.NamedTemporaryFile(mode='w',prefix='telemetry-',suffix='.jsonl',dir=str(ROOT/'state'),delete=False) as telemetry:
                telemetry_path=Path(telemetry.name);s['telemetry_file']=str(telemetry_path);atomic(STATE,s)
                while child.poll() is None:
                    t=sample(child.pid)
                    if previous and 'cpu_ticks' in t and 'cpu_ticks' in previous:t['cpu_one_core_percent']=100*(t['cpu_ticks']-previous['cpu_ticks'])/os.sysconf('SC_CLK_TCK')/(t['monotonic']-previous['monotonic'])
                    previous=t;telemetry.write(json.dumps(t)+'\n');telemetry.flush();
                    if not previous_sync or t['monotonic']-previous_sync>=5:os.fsync(telemetry.fileno());previous_sync=t['monotonic']
                    atomic(ROOT/'state/telemetry-latest.json',dict(t,native_pid=child.pid))
                    if t.get('temperature_c',0) and t['temperature_c']>=72:stop();terminate_at=terminate_at or time.monotonic()
                    if stopping:terminate_at=terminate_at or time.monotonic()
                    if terminate_at and time.monotonic()-terminate_at>10:child.kill()
                    time.sleep(1)
        output.mkdir(parents=True,exist_ok=True);(output/'native.log').write_text((ROOT/'state/native-startup.log').read_text());shutil.copyfile(telemetry_path,output/'telemetry.jsonl');telemetry_path.unlink();atomic(output/'command.json',command)
        return child.returncode
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        output.mkdir(parents=True,exist_ok=True);restore()
if __name__=='__main__':sys.exit(main())

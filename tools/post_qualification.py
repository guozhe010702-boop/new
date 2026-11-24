"""Final finite checks after the long suite, supervised by its own transient unit."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,hashlib
from control import ROOT,atomic
state_file=ROOT/'results/post-qualification.json';state={'stage':'running','pid':os.getpid(),'started':time.time()};atomic(state_file,state)
child=None
def stop(*_):
    if child and child.poll() is None:child.terminate()
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def run(args):
    global child
    child=subprocess.Popen(args,env=dict(os.environ,AA_FINAL_CHECK='1'))
    if child.wait()!=0:raise RuntimeError('Post-check failed: '+str(args))
try:
    job=json.loads((ROOT/'results/qualification-20261124/job.json').read_text());assert job['stage']=='complete'
    aspect=json.loads((ROOT/'results/display-aspect-fix/change.json').read_text())
    assert hashlib.sha256((ROOT/'build/advanced_vision').read_bytes()).hexdigest()==aspect['after_binary_sha256'],'Qualified display binary changed'
    run([sys.executable,str(ROOT/'tools/audit_qualification.py')])
    old=ROOT/'results/lifecycle';backup=ROOT/'results/lifecycle-initial'
    if old.exists() and not backup.exists():old.rename(backup)
    if not old.exists():run([sys.executable,str(ROOT/'tools/lifecycle.py')])
    life=json.loads((old/'result.json').read_text());assert len(life)>=7 and all(r['passed'] for r in life)
    final=ROOT/'results/yolo-final-display-smoke'
    if not final.exists():run([sys.executable,str(ROOT/'tools/control.py'),'run','--project','yolo11_bytetrack','--seconds','60','--save-every','0','--output',str(final)])
    summary=json.loads((final/'summary.json').read_text());assert not summary['error'] and summary['seconds']>=60
    run([sys.executable,str(ROOT/'tools/final_cleanup.py')])
    owners=subprocess.run(['lsof','-t','/dev/video44'],capture_output=True,text=True);assert owners.returncode in (0,1) and not owners.stdout.strip(),'Camera is still occupied'
    restoration=json.loads((final/'restoration.json').read_text());assert restoration['restored']
    idle={'camera_owners':[],'desktop_active':subprocess.run(['systemctl','is-active','--quiet','gdm.service']).returncode==0,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'at':time.time()};assert idle['boot_id']==job['boot_id'];atomic(ROOT/'results/final-idle.json',idle)
    state.update(stage='complete',finished=time.time())
except BaseException as e:
    state.update(stage='failed',error=repr(e),finished=time.time())
    for project in ('yolo11_bytetrack','real_esrgan','depth_anything_v2'):
        p=ROOT/project/'recommended.json'
        if p.exists():
            profile=json.loads(p.read_text());profile.update(stability_validated=False,final_check_error=repr(e));atomic(p,profile)
    raise
finally:
    if child and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=25)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    atomic(state_file,state)

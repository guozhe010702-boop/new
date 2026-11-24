"""Windows-only finite watcher: mirror status, then audit, collect and package.

Does not start new benchmarks, publish to GitHub or register a startup task.
"""
import base64,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];KEY=str(Path.home()/'.ssh/evm_rk3588');HOST='root@192.168.2.221';BOARD='/userdata/av_dev/advanced algorithms'
STATE=ROOT/'results/async-delivery.json'
def atomic(value):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)
def run(args,**kw):return subprocess.run(args,check=True,text=True,**kw)
def remote(code):
    command='python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+base64.b64encode(code.encode()).decode()+"'))")
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,command],text=True)
def send(path,rel):
    run(['scp','-i',KEY,str(path),HOST+':'+BOARD+'/'+rel])
def get(remote_path,local_path):
    local_path.parent.mkdir(parents=True,exist_ok=True);run(['scp','-i',KEY,HOST+':'+remote_path,str(local_path)])
def main():
    if os.name!='nt':raise SystemExit('Run this delivery watcher on the Windows checkout')
    state={'stage':'waiting_for_qualification','pid':os.getpid(),'started':time.time()};atomic(state);deadline=time.monotonic()+6*3600;errors=0
    while time.monotonic()<deadline:
        try:
            run([sys.executable,str(ROOT/'tools/sync_status.py')]);run([sys.executable,str(ROOT/'tools/report.py')]);job=json.loads((ROOT/'results/qualification-20261124/job.json').read_text());errors=0
            state.update(stage='waiting_for_qualification',current=job.get('current'),finished_cases=len(job.get('cases',[])),updated=time.time());atomic(state)
            if job['stage']=='complete':break
            if job['stage']!='running' or not job.get('observed_process_running') or job.get('observed_boot_id')!=job.get('boot_id'):raise RuntimeError('Qualification stopped or board restarted: '+str(job.get('stage')))
        except subprocess.SubprocessError as e:
            errors+=1;state.update(network_errors=errors,last_error=str(e));atomic(state)
            if errors>=20:raise
        time.sleep(45)
    else:raise RuntimeError('Qualification did not complete within the delivery window')
    state.update(stage='post_checks');atomic(state)
    # Update documentation/tools only. Never overwrite the running native binary.
    for path in (ROOT/'tools').glob('*'):
        if path.is_file():send(path,'tools/'+path.name)
    for path in (ROOT/'docs').glob('*'):
        if path.is_file():send(path,'docs/'+path.name)
    send(ROOT/'README.md','README.md')
    remote("""from pathlib import Path
import subprocess,shlex,json
r=Path('/userdata/av_dev/advanced algorithms')
unit='advanced-final-check.service'
p=r/'results/post-qualification.json'
if not (p.exists() and json.loads(p.read_text()).get('stage')=='complete'):
 if subprocess.run(['systemctl','is-active','--quiet',unit]).returncode!=0:
  post='/usr/bin/python3 '+shlex.quote(str(r/'tools/final_cleanup.py'))
  subprocess.run(['systemd-run','--unit='+unit,'--collect','--property=KillMode=mixed','--property=TimeoutStopSec=45','--property=WorkingDirectory='+str(r),'--property=ExecStopPost='+post,'--setenv=AA_FINAL_CHECK=1','/usr/bin/python3',str(r/'tools/post_qualification.py')],check=True)
""")
    for _ in range(60):
        raw=remote("from pathlib import Path; p=Path('/userdata/av_dev/advanced algorithms/results/post-qualification.json'); print(p.read_text() if p.exists() else '{}')")
        post=json.loads(raw)
        if post.get('stage')=='complete':break
        if post.get('stage')=='failed':raise RuntimeError('Post-check failed: '+str(post))
        time.sleep(10)
    else:raise RuntimeError('Final post-check timed out')
    state.update(stage='collecting');atomic(state)
    remote("""from pathlib import Path
import tarfile
r=Path('/userdata/av_dev/advanced algorithms')
with tarfile.open('/tmp/advanced-final-results.tgz','w:gz',compresslevel=1) as a:
 for p in (r/'results').rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(r)
  if p.suffix in ('.f32','.tif','.nv12') or p.name=='advanced_vision-before' or p.name.startswith('frame-'):continue
  a.add(p,arcname=rel.as_posix(),recursive=False)
""")
    archive=ROOT/'results/final-board-results.tgz';get('/tmp/advanced-final-results.tgz',archive)
    with tarfile.open(archive) as a:
        assert all('..' not in Path(m.name).parts and not Path(m.name).is_absolute() for m in a)
        a.extractall(ROOT)
    for rel in ['common/display.cpp']+['build/'+n for n in ('advanced_vision','model_batch','sr_tiled','core_tests','hardware_tests')]:get(BOARD+'/'+rel,ROOT/rel)
    # The actual 16K output remains on the board; also deliver its Windows copy.
    get(BOARD+'/results/4k-x4.tif',ROOT/'results/4k-x4.tif')
    run([sys.executable,str(ROOT/'tools/sync_status.py')]);run([sys.executable,str(ROOT/'tools/report.py')])
    audit=json.loads((ROOT/'results/final-audit.json').read_text());assert audit['passed']
    manifest={}
    for dirname in ('common','runtime','vendor','tools'):
        for p in (ROOT/dirname).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'docs/source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    environment=ROOT/'docs/environment.json'
    if environment.exists():
        data=json.loads(environment.read_text());data['core_source_sha256']={k:v for k,v in manifest.items() if k.startswith('common/')};data['display_revision_receipt']='results/display-aspect-fix/change.json';environment.write_text(json.dumps(data,indent=2)+'\n')
    destination=ROOT/'releases/advanced-algorithms-rk3588-20261124.tar.gz';destination.parent.mkdir(exist_ok=True)
    if not destination.exists():run([sys.executable,str(ROOT/'tools/deploy.py'),'--output',str(destination)])
    expected=json.loads((ROOT/'package-manifest.json').read_text())
    with tarfile.open(destination) as a:
        for rel,sha in expected.items():
            content=a.extractfile('advanced algorithms/'+rel).read();assert hashlib.sha256(content).hexdigest()==sha
    for rel in ('docs/实测报告.md','docs/source-manifest.json','README.md'):send(ROOT/rel,rel)
    state.update(stage='complete',finished=time.time(),package=str(destination),audit_passed=True,preview_stopped=True);atomic(state)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        atomic({'stage':'failed','error':repr(e),'traceback':traceback.format_exc(),'at':time.time()})
        try:run([sys.executable,str(ROOT/'tools/sync_status.py')]);run([sys.executable,str(ROOT/'tools/report.py')])
        except Exception:pass
        raise

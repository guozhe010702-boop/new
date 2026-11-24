#!/usr/bin/env python3
"""Run only while all camera applications are stopped. Uses real model/DRM."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from control import ROOT,STATE,atomic,alive
out=ROOT/'results/lifecycle';out.mkdir(exist_ok=False);records=[]
def control(*args):return subprocess.run([sys.executable,str(ROOT/'tools/control.py')]+list(args),capture_output=True,text=True)
def record(name,ok,**kw):
    records.append(dict(name=name,passed=bool(ok),**kw));atomic(out/'result.json',records)
    if not ok:raise RuntimeError(name+' failed')
def wait_native():
    for _ in range(30):
        if STATE.exists():
            s=json.loads(STATE.read_text())
            if alive(s.get('native_pid')):return s
        time.sleep(.5)
    raise RuntimeError('No native process')
r=control('run','--project','yolo11_bytetrack','--precision','int8','--seconds','10','--inject-display-error-after','5','--output',str(out/'display-error'))
restore=json.loads((out/'display-error/restoration.json').read_text());record('injected-display-failure',r.returncode==1 and restore['restored'],returncode=r.returncode)
for i in range(2):
    r=control('start','--project','yolo11_bytetrack','--precision','int8','--save-every','0');record('manual-start-'+str(i),r.returncode==0,stdout=r.stdout,stderr=r.stderr);s=wait_native();time.sleep(5)
    if i==0:
        duplicate=control('start','--project','yolo11_bytetrack');record('duplicate-start-rejected',duplicate.returncode!=0)
    stopped=control('stop');state=json.loads(STATE.read_text());record('manual-stop-'+str(i),stopped.returncode==0 and state.get('restored') and not alive(s['native_pid']))
r=control('start','--project','yolo11_bytetrack','--precision','int8','--save-every','0');s=wait_native();time.sleep(5);os.kill(s['supervisor_pid'],signal.SIGKILL)
for _ in range(80):
    state=json.loads(STATE.read_text())
    if state.get('restored') and not alive(s['native_pid']):break
    time.sleep(.5)
record('supervisor-sigkill-restored',state.get('restored') and not alive(s['native_pid']))
print(json.dumps(records))

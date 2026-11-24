#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,os,platform,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-package',action='store_true');a=parser.parse_args()
    if a.verify_package:
        m=json.loads((ROOT/'package-manifest.json').read_text())
        for rel,expected in m.items():
            p=ROOT/rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('Package checksum mismatch: '+rel)
        print('Package checksums passed')
    if platform.system()!='Linux' or platform.machine()!='aarch64':raise SystemExit('Runtime requires RK3588 ARM64 Linux; Windows holds sources only')
    if os.geteuid()!=0:raise SystemExit('Run as root')
    for p in ('/dev/video44','/dev/v4l-subdev2','/dev/dri/card0','/dev/dma_heap/system-dma32','/dev/rknpu'):
        if not Path(p).exists():
            if p=='/dev/rknpu':print('NPU device name varies by BSP; actual RKNN init still required')
            else:raise SystemExit('Missing hardware node: '+p)
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib'))
    p=subprocess.run(['ldd',str(ROOT/'build/advanced_vision')],env=env,capture_output=True,text=True);print(p.stdout)
    if p.returncode or 'not found' in p.stdout:raise SystemExit('Missing shared libraries')
    for library in ('librknnrt.so','librga.so.2'):
        if not any(library in line and str(ROOT/'runtime/lib') in line for line in p.stdout.splitlines()):raise SystemExit('Runtime resolved outside the tested private library directory: '+library)
    subprocess.run(['gst-inspect-1.0','v4l2src'],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['gst-inspect-1.0','videorate'],stdout=subprocess.DEVNULL,check=True)
    print('Preflight passed; camera ownership and HDMI are checked at startup')
if __name__=='__main__':main()

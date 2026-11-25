"""Create a reproducible ARM64 runtime archive from a completed local checkout."""
from pathlib import Path
import argparse,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Archive already exists'
    included=[]
    for f in ROOT.rglob('*'):
        if not f.is_file():continue
        rel=f.relative_to(ROOT);parts=rel.parts
        if parts[0] in ('.git','state','results','data','build-from-stage','prebuilt','releases') or 'experiments' in parts or '__pycache__' in parts or f.suffix in ('.onnx','.pth','.pt','.pyc','.tgz','.zip') or f.name.endswith('.tar.gz'):continue
        if parts[0]=='build' and rel.as_posix() not in ('build/advanced_vision','build/model_batch','build/sr_tiled','build/hardware_tests','build/core_tests','build/render_tests','build/npu_profile','build/sr_tail_check','build/async_writer_tests'):continue
        if f.name=='package-manifest.json':continue
        included.append(f)
    required=[ROOT/'build/advanced_vision',ROOT/'runtime/lib/librknnrt.so']+[ROOT/n/'models'/('int8.rknn' if n=='yolo11_bytetrack' else 'fp16.rknn') for n in ('yolo11_bytetrack','real_esrgan','depth_anything_v2')]
    profile=ROOT/'real_esrgan/recommended.json'
    if profile.exists() and json.loads(profile.read_text()).get('sr_tail')=='cpu':required.append(ROOT/'real_esrgan/models/fp16-trunk.rknn')
    depth_profile=ROOT/'depth_anything_v2/recommended.json'
    if depth_profile.exists() and json.loads(depth_profile.read_text()).get('depth_profile')=='camera-rect':required.append(ROOT/'depth_anything_v2/models/fp16-camera-518x294.rknn')
    assert all(p in included for p in required),'Missing runtime files'
    manifest={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in included};(ROOT/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'w:gz',compresslevel=6) as archive:
        for f in included+[ROOT/'package-manifest.json']:
            rel=f.relative_to(ROOT);info=archive.gettarinfo(str(f),arcname='advanced algorithms/'+rel.as_posix());info.uid=info.gid=0;info.uname=info.gname='root';info.mode=0o755 if f.suffix=='.sh' or rel.parts[0]=='build' else 0o644
            with f.open('rb') as stream:archive.addfile(info,stream)
    checksum=hashlib.sha256(a.output.read_bytes()).hexdigest();a.output.with_suffix(a.output.suffix+'.sha256').write_text(checksum+'  '+a.output.name+'\n');print(checksum)
if __name__=='__main__':main()

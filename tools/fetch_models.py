"""Fetch the three original official checkpoints with fixed SHA256 checks."""
from pathlib import Path
import hashlib,urllib.request,json
ROOT=Path(__file__).resolve().parents[1]
ASSETS=[
 ('yolo11_bytetrack/models/yolo11n.pt','https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt','0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'),
 ('real_esrgan/models/realesr-general-x4v3.pth','https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth','8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'),
 ('depth_anything_v2/models/depth_anything_v2_vits.pth','https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth','715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378')]
def main():
    records=[]
    for name,url,expected in ASSETS:
        p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'RK3588-advanced-algorithms'})
            with urllib.request.urlopen(request,timeout=120) as response:p.with_suffix(p.suffix+'.part').write_bytes(response.read())
            temp=p.with_suffix(p.suffix+'.part')
            if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected:raise RuntimeError('Checkpoint checksum mismatch: '+name)
            temp.replace(p)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('Existing checkpoint mismatch: '+name)
        records.append(dict(path=name,source=url,sha256=expected))
    (ROOT/'docs/model-sources.json').write_text(json.dumps(records,indent=2)+'\n');print('Official checkpoint checksums passed')
if __name__=='__main__':main()

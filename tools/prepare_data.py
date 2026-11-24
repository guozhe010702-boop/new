"""Deterministic, disjoint calibration/validation preprocessing in conversion container."""
from pathlib import Path
import cv2,numpy as np,json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def letterbox(im,n):
    h,w=im.shape[:2];s=min(n/w,n/h);nw,nh=round(w*s),round(h*s);x,y=(n-nw)//2,(n-nh)//2;o=np.full((n,n,3),114,np.uint8);o[y:y+nh,x:x+nw]=cv2.resize(im,(nw,nh));return o
def main():
    coco=sorted((R/'data/coco128/images/train2017').glob('*.jpg'));camera=sorted((R/'data/camera').glob('*.png'))
    assert len(coco)>=128 and camera,'Need real representative images'
    split={'calibration':coco[:96]+camera[::2],'validation':coco[96:]+camera[1::2][:18]}
    # Exact image hashes are used to prohibit repeated images across partitions.
    assert not ({sha(p) for p in split['calibration']} & {sha(p) for p in split['validation']})
    receipt={}
    for project,size in [('yolo11_bytetrack',640),('depth_anything_v2',518),('real_esrgan',256)]:
        receipt[project]={}
        for kind,images in split.items():
            folder=R/'data/prepared'/kind/project;folder.mkdir(parents=True,exist_ok=True);records=[]
            for i,p in enumerate(images):
                image=cv2.imread(str(p));assert image is not None
                if project=='real_esrgan':
                    h,w=image.shape[:2]
                    if min(h,w)<256:image=cv2.resize(image,(max(w,256),max(h,256)));h,w=image.shape[:2]
                    image=image[(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]
                else:image=letterbox(image,size)
                target=folder/(f'{i:04d}.png');cv2.imwrite(str(target),image);records.append({'file':target.name,'source':str(p.relative_to(R)),'source_sha256':sha(p),'prepared_sha256':sha(target)})
            receipt[project][kind]=records
            if kind=='calibration':(R/'data'/('calibration-'+project+'.txt')).write_text(''.join(str(folder/r['file'])+'\n' for r in records))
    (R/'data/dataset.json').write_text(json.dumps(receipt,indent=2));print({k:len(v) for k,v in split.items()})
if __name__=='__main__':main()

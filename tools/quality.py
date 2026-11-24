"""Compare actual board tensors with ONNX and each other. Never mark missing tests passed."""
from pathlib import Path
import argparse,json,cv2,numpy as np,onnxruntime as ort
from scipy.stats import spearmanr
def structural_similarity(a,b,data_range=1,channel_axis=2):
    a=a.astype(np.float64);b=b.astype(np.float64);u=cv2.GaussianBlur(a,(11,11),1.5);v=cv2.GaussianBlur(b,(11,11),1.5);aa=cv2.GaussianBlur(a*a,(11,11),1.5)-u*u;bb=cv2.GaussianBlur(b*b,(11,11),1.5)-v*v;ab=cv2.GaussianBlur(a*b,(11,11),1.5)-u*v
    value=((2*u*v+.01**2)*(2*ab+.03**2))/((u*u+v*v+.01**2)*(aa+bb+.03**2));return float(value[5:-5,5:-5].mean())
R=Path(__file__).resolve().parents[1]
def outputs(folder,stem,shapes):
    return [np.fromfile(folder/(stem+'-'+str(i)+'.f32'),np.float32).reshape(shape) for i,shape in enumerate(shapes)]
def detection(heads):
    boxes=[]
    for i in range(3):
        b=heads[i*3][0];scores=heads[i*3+1][0];h,w=scores.shape[1:];c=scores.argmax(0);conf=scores.max(0);ys,xs=np.where(conf>=.1)
        for y,x in zip(ys,xs):
            a=b[:,y,x].reshape(4,16);a=np.exp(a-a.max(1,keepdims=True));d=(a*np.arange(16)).sum(1)/a.sum(1);step=640/w;xx=(x+.5)*step;yy=(y+.5)*step;boxes.append([xx-d[0]*step,yy-d[1]*step,xx+d[2]*step,yy+d[3]*step,float(conf[y,x]),int(c[y,x])])
    boxes=sorted(boxes,key=lambda x:-x[4]);keep=[]
    for b in boxes:
        if any(k[5]==b[5] and iou(b,k)>.45 for k in keep):continue
        keep.append(b)
        if len(keep)>=300:break
    return keep
def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h;return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def map_score(predictions,records):
    gt={};classes=set()
    for record in records:
        source=R/record['source']
        if 'coco128' not in str(source):continue
        h,w=cv2.imread(str(source)).shape[:2];scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale);px,py=(640-nw)//2,(640-nh)//2
        label=R/'data/coco128/labels/train2017'/(source.stem+'.txt');items=[]
        if label.exists():
            for line in label.read_text().splitlines():
                cls,x,y,bw,bh=map(float,line.split());classes.add(int(cls));items.append([(x-bw/2)*nw+px,(y-bh/2)*nh+py,(x+bw/2)*nw+px,(y+bh/2)*nh+py,int(cls)])
        gt[Path(record['file']).stem]=items
    aps=[]
    for cls in sorted(classes):
        total=sum(sum(b[4]==cls for b in v) for v in gt.values())
        if not total:continue
        detections=sorted([(b[4],key,b) for key,boxes in predictions.items() if key in gt for b in boxes if b[5]==cls],reverse=True,key=lambda x:x[0])
        for threshold in np.arange(.5,.951,.05):
            matched=set();tp=[];fp=[]
            for _,key,b in detections:
                candidates=[(iou(b,g),j) for j,g in enumerate(gt[key]) if g[4]==cls and (key,j) not in matched]
                best=max(candidates,default=(0,-1));yes=best[0]>=threshold
                if yes:matched.add((key,best[1]))
                tp.append(int(yes));fp.append(int(not yes))
            tp=np.cumsum(tp);fp=np.cumsum(fp);rec=tp/total;prec=tp/np.maximum(1,tp+fp)
            aps.append(np.mean([max(prec[rec>=r],default=0) for r in np.linspace(0,1,101)]))
    return float(np.mean(aps)) if aps else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--fp16',type=Path,required=True);p.add_argument('--int8',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((R/a.project/'models/conversion-fp16.json').read_text());shapes=[v['shape'] for v in manifest['outputs']];records=json.loads((R/'data/dataset.json').read_text())[a.project]['validation'];opts=ort.SessionOptions();opts.intra_op_num_threads=4;session=ort.InferenceSession(str(R/a.project/'models/model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    rows=[];pred_f={};pred_i={};pred_ref={}
    for record in records:
        path=R/'data/prepared/validation'/a.project/record['file'];stem=path.stem
        if not (a.fp16/(stem+'-0.f32')).exists():continue
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);x=(rgb.astype(np.float32)-np.array(manifest['mean'],np.float32))/np.array(manifest['std'],np.float32);reference=session.run(None,{'images':x.transpose(2,0,1)[None]});fp=outputs(a.fp16,stem,shapes);row={'image':record['file'],'fp16_vs_onnx_max_absolute':max(float(np.max(np.abs(f-r))) for f,r in zip(fp,reference))}
        if a.project=='yolo11_bytetrack':pred_f[stem]=detection(fp);pred_ref[stem]=detection(reference)
        elif a.project=='depth_anything_v2':row['fp16_vs_onnx_spearman']=float(spearmanr(fp[0].ravel(),reference[0].ravel()).statistic)
        else:
            u=np.clip(fp[0][0].transpose(1,2,0),0,1);v=np.clip(reference[0][0].transpose(1,2,0),0,1);row['fp16_vs_onnx_psnr_db']=float(-10*np.log10(max(np.mean((u-v)**2),1e-20)))
        if a.int8 and (a.int8/(stem+'-0.f32')).exists():
            quant=outputs(a.int8,stem,shapes)
            if a.project=='depth_anything_v2':row['spearman']=float(spearmanr(fp[0].ravel(),quant[0].ravel()).statistic)
            elif a.project=='real_esrgan':
                u=np.clip(fp[0][0].transpose(1,2,0),0,1);v=np.clip(quant[0][0].transpose(1,2,0),0,1);mse=np.mean((u-v)**2);row['psnr_db']=float(-10*np.log10(max(mse,1e-20)));row['ssim']=float(structural_similarity(u,v,data_range=1,channel_axis=2))
            else:pred_i[stem]=detection(quant)
        rows.append(row)
    assert rows,'No real board tensors'
    report={'project':a.project,'validation_images':len(rows),'expected_images':len(records),'rows':rows,'int8_quality_pass':False,'scope':'quantization parity; not metric depth accuracy or scientific detail validation'}
    if a.project=='depth_anything_v2' and all('spearman' in r for r in rows):report['median_spearman']=float(np.median([r['spearman'] for r in rows]));report['int8_quality_pass']=len(rows)==len(records) and report['median_spearman']>=.98
    if a.project=='real_esrgan' and all('psnr_db' in r for r in rows):report['mean_psnr_db']=float(np.mean([r['psnr_db'] for r in rows]));report['mean_ssim']=float(np.mean([r['ssim'] for r in rows]));report['int8_quality_pass']=len(rows)==len(records) and report['mean_psnr_db']>=35 and report['mean_ssim']>=.98
    if a.project=='yolo11_bytetrack':
        report['fp16_subset_map']=map_score(pred_f,records);report['int8_subset_map']=map_score(pred_i,records) if pred_i else None;report['int8_quality_pass']=len(pred_i)==len(records) and report['int8_subset_map'] is not None and report['fp16_subset_map']-report['int8_subset_map']<=.02;report['map_scope']='COCO128 held-out subset only; not full COCO val2017'
        report['onnx_subset_map']=map_score(pred_ref,records);report['fp16_reference_pass']=len(rows)==len(records) and report['onnx_subset_map']-report['fp16_subset_map']<=.02
    elif a.project=='depth_anything_v2':report['fp16_onnx_median_spearman']=float(np.median([r['fp16_vs_onnx_spearman'] for r in rows]));report['fp16_reference_pass']=len(rows)==len(records) and report['fp16_onnx_median_spearman']>=.99
    else:report['fp16_onnx_mean_psnr_db']=float(np.mean([r['fp16_vs_onnx_psnr_db'] for r in rows]));report['fp16_reference_pass']=len(rows)==len(records) and report['fp16_onnx_mean_psnr_db']>=40
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'})
if __name__=='__main__':main()

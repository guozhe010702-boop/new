from pathlib import Path
import cv2,numpy as np,json
R=Path(__file__).resolve().parents[1];folder=R/'results/seam-test'
a=cv2.imread(str(folder/'reference.png')).astype(np.float32);b=cv2.imread(str(folder/'tiled.png')).astype(np.float32);assert a.shape==b.shape==(1536,1536,3)
error=np.abs(a-b);mask=np.zeros(a.shape[:2],bool)
for v in (176*4,352*4):mask[max(0,v-4):v+4,:]=True;mask[:,max(0,v-4):v+4]=True
mse=float(np.mean((a-b)**2));report={'psnr_db':float(10*np.log10(255**2/max(1e-12,mse))),'seam_band_mae_8bit':float(error[mask].mean()),'all_pixel_mae_8bit':float(error.mean()),'max_error_8bit':float(error.max()),'reference':'full 384x384 PyTorch FP32 vs tiled board FP16','passed':bool(mse<255**2/10**4 and error[mask].mean()<2)}
(folder/'comparison.json').write_text(json.dumps(report,indent=2));print(report)
assert report['passed'],'Tile seam parity failed'

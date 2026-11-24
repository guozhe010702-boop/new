from pathlib import Path
import cv2,torch,numpy as np,json
from convert import SRVGG,ROOT
torch.set_num_threads(4)
folder=ROOT/'results/seam-test';folder.mkdir(exist_ok=True)
image=cv2.imread(str(next((ROOT/'data/coco128/images/train2017').glob('*.jpg'))));image=cv2.resize(image,(384,384));cv2.imwrite(str(folder/'input.png'),image)
model=SRVGG();checkpoint=torch.load(ROOT/'real_esrgan/models/realesr-general-x4v3.pth',map_location='cpu');model.load_state_dict(checkpoint.get('params_ema',checkpoint.get('params',checkpoint)));model.eval()
with torch.inference_mode():out=model(torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255))[0].numpy().transpose(1,2,0)
cv2.imwrite(str(folder/'reference.png'),cv2.cvtColor(np.clip(np.rint(out*255),0,255).astype(np.uint8),cv2.COLOR_RGB2BGR))

# inference.py
import cv2, numpy as np, torch
from ultralytics import YOLO
from models import SiameseUNet
from PIL import Image
import torchvision.transforms as T

yolo = YOLO('yolov8n.pt')  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
change_model = SiameseUNet(in_ch=3, base_ch=32).to(device)
try:
    change_model.load_state_dict(torch.load('change_detector.pth', map_location=device))
    change_model.eval()
except Exception as e:
    print("Warning: could not load change detector weights:", e)
    change_model.eval()

def preprocess_pair(im1_bgr, im2_bgr, out_size=(512,512)):
    im1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2RGB)
    im1 = cv2.resize(im1, out_size)
    im2 = cv2.resize(im2, out_size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    t1 = transform(Image.fromarray(im1)).unsqueeze(0)
    t2 = transform(Image.fromarray(im2)).unsqueeze(0)
    return t1, t2

def run_change_detector(img_old_bgr, img_new_bgr, thresh=0.4):
    t1, t2 = preprocess_pair(img_old_bgr, img_new_bgr)
    t1 = t1.to(device); t2 = t2.to(device)
    with torch.no_grad():
        out = change_model(t1, t2)[0,0].cpu().numpy()
    h,w = img_new_bgr.shape[:2]
    out_resized = cv2.resize((out*255).astype('uint8'), (w,h))
    _, mask = cv2.threshold(out_resized, int(thresh*255), 255, cv2.THRESH_BINARY)
    return mask

def run_yolo(image_bgr, conf=0.25):
    results = yolo(image_bgr[..., ::-1], imgsz=640, conf=conf, verbose=False)
    r = results[0]
    objs = []

    has_masks = hasattr(r, 'masks') and r.masks is not None and len(r.masks.data)>0
    for i, box in enumerate(r.boxes.data.tolist()):
        # box: [x1,y1,x2,y2,score,class]
        x1,y1,x2,y2,score,cls = box[:6]
        cls = int(cls)
        bbox = [int(x1),int(y1),int(x2),int(y2)]
        mask = None
        if has_masks:
        
            try:
            
                m = r.masks.data[i].cpu().numpy()  
                mask = (m * 255).astype('uint8')
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))
            except Exception:
                mask = None
        objs.append({'class': int(cls), 'score': float(score), 'bbox': bbox, 'mask': mask})
    return objs

def compute_damage_for_objects(objects, change_mask):
    """
    objects: list with 'bbox' and 'mask' (mask may be None)
    change_mask: binary uint8 mask (255 for change)
    Returns: objects with damage_pct field
    """
    out = []
    for obj in objects:
        if obj['mask'] is not None:
            obj_mask = (obj['mask'] > 127).astype('uint8')
        else:
            x1,y1,x2,y2 = obj['bbox']
            h = change_mask.shape[0]; w = change_mask.shape[1]
            x1,x2 = max(0,x1), min(w-1,x2)
            y1,y2 = max(0,y1), min(h-1,y2)
            obj_mask = np.zeros_like(change_mask, dtype='uint8')
            obj_mask[y1:y2, x1:x2] = 1
        changed = ((change_mask>127) & (obj_mask>0)).sum()
        area = (obj_mask>0).sum()
        damage_pct = 0.0 if area==0 else 100.0 * changed / area
        out.append({**obj, 'damage_pct': float(damage_pct), 'changed_pixels': int(changed), 'object_pixels': int(area)})
    return out

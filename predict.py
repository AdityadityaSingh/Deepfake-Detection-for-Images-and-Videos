"""Classify a single face image or a video.
Usage:
  python3 predict.py path/to/face.jpg            # image (face crop or full photo)
  python3 predict.py path/to/video.mp4           # video
  python3 predict.py x.jpg --model EfficientNet-B0
"""
import argparse, os, cv2, numpy as np, torch
from dfd import RES, to_tensor
from video_infer import load, score_video, CASCADE

ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--model', default='SFCA-Net'); ap.add_argument('--ckpt', default=None)
a = ap.parse_args(); torch.set_num_threads(2)
model = load(a.model, a.ckpt or f'results/ckpt/{a.model}__all.pt')
if a.path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
    r = score_video(model, a.path)
    print(f"{r['video']}: {r['verdict']}  P(fake)={r.get('p_fake_mean_logit', float('nan')):.3f}  frames scored={r['frames_scored']}")
else:
    img = cv2.imread(a.path); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(img, 1.3, 5)
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3]); img = img[y + 2:y + h - 2, x + 2:x + w - 2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_AREA)[None]
    with torch.no_grad(): p = torch.sigmoid(model(to_tensor(x))['logit']).item()
    print(f"{os.path.basename(a.path)}: {'FAKE' if p >= 0.5 else 'REAL'}  P(fake)={p:.3f}  face {'detected' if len(faces) else 'not detected; used full image'}")

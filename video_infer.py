"""Video deepfake detection with a trained frame model.
Pipeline: sample frames -> Haar-cascade face detection -> 96x96 crop -> per-frame P(fake)
-> temporal aggregation (mean of logits and top-k mean) -> video verdict.

Usage: python3 video_infer.py --model SFCA-Net --videos a.mp4 b.mp4 [--labels fake real] [--plot out.png]
"""
import argparse, contextlib, io, json, os
import cv2, numpy as np, torch
from dfd import build, to_tensor, RES

CASCADE = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_alt.xml'))  # same detector used to build the training crops

def face_crops(path, every=3, max_frames=120):
    cap = cv2.VideoCapture(path); crops, idx, fi = [], [], 0
    while len(crops) < max_frames:
        ok, fr = cap.read()
        if not ok: break
        if fi % every == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(fr, 1.3, 5)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                crop = fr[y + 2:y + h - 2, x + 2:x + w - 2]  # same cropping rule as the training data extraction
                crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_AREA)
                crops.append(crop); idx.append(fi)
        fi += 1
    cap.release(); return np.array(crops), np.array(idx), fi

def load(name, ckpt):
    with contextlib.redirect_stdout(io.StringIO()): m = build(name)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')); m.eval(); return m

@torch.no_grad()
def score_video(model, path, thr=0.5):
    X, idx, n = face_crops(path)
    if len(X) == 0: return dict(video=os.path.basename(path), frames_scored=0, verdict='no face found')
    logit = model(to_tensor(X))['logit'].numpy(); p = 1 / (1 + np.exp(-logit))
    k = max(1, len(p) // 4); topk = float(np.sort(p)[-k:].mean())
    p_video = float(1 / (1 + np.exp(-logit.mean())))
    return dict(video=os.path.basename(path), frames_total=int(n), frames_scored=int(len(p)), frame_idx=idx.tolist(),
                frame_prob=p.tolist(), p_fake_mean_logit=p_video, p_fake_topk=topk, frac_frames_fake=float((p >= thr).mean()),
                verdict='FAKE' if p_video >= thr else 'REAL')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='SFCA-Net'); ap.add_argument('--ckpt', default=None)
    ap.add_argument('--videos', nargs='+', required=True); ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--plot', default='figures/18_video_inference.png'); ap.add_argument('--json', default='results/video_inference.json')
    a = ap.parse_args(); torch.set_num_threads(2)
    model = load(a.model, a.ckpt or f'results/ckpt/{a.model}__all.pt')
    out = []
    for i, v in enumerate(a.videos):
        r = score_video(model, v); r['label'] = a.labels[i] if a.labels else 'unknown'; out.append(r)
        print(f"{r['video']}: label={r['label']} frames={r.get('frames_scored')} P(fake)={r.get('p_fake_mean_logit', float('nan')):.3f} -> {r['verdict']}")
    os.makedirs(os.path.dirname(a.json), exist_ok=True); json.dump(out, open(a.json, 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7, 2.8)); cols = ['#e34948', '#2a78d6', '#1baf7a', '#eda100']
    for i, r in enumerate(out):
        if not r.get('frames_scored'): continue
        ax.plot(r['frame_idx'], r['frame_prob'], marker='o', ms=2.5, lw=1, color=cols[i % 4],
                label=f"{r['video']} ({r['label']}): video P(fake) = {r['p_fake_mean_logit']:.2f} -> {r['verdict']}")
    ax.axhline(0.5, color='#52514e', ls=':', lw=0.8); ax.set_ylim(0, 1); ax.set_xlabel('Frame index'); ax.set_ylabel('P(fake) per frame')
    ax.set_title(f'Video-level inference with {a.model} (temporal aggregation of frame scores)'); ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    fig.savefig(a.plot, bbox_inches='tight', dpi=200); fig.savefig(a.plot.replace('.png', '.pdf'), bbox_inches='tight')

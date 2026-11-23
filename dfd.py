"""Deepfake detection: data, degradations, models, training, evaluation.

Proposed model: SFCA-Net (Spatial-Frequency Cross-Attention Network).
  * Spatial stream: ImageNet-pretrained EfficientNet-B0 feature map -> tokens.
  * Frequency stream: per-image log-amplitude FFT spectrum + 3 fixed SRM
    high-pass residuals -> light CNN -> tokens.
  * Bidirectional multi-head cross-attention between the two token sets,
    gated fusion, and auxiliary per-stream heads (deep supervision).
  * Degradation-aware training augmentation (blur / noise / JPEG / light).
"""
import glob, os, random, io
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

ROOT = '/home/claude/ext/tranthuhoai3786_FaceForensics/image'
MANIPS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
RES = 96
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------- data
def load_split(split, cache='/home/claude/dfd/cache'):
    os.makedirs(cache, exist_ok=True)
    f = f'{cache}/{split}_{RES}.npz'
    if os.path.exists(f):
        z = np.load(f); return z['x'], z['y'], z['m']
    X, Y, M = [], [], []
    for mi, cls in enumerate(['youtube'] + MANIPS):
        for p in sorted(glob.glob(f'{ROOT}/{split}/{cls}/*.jpg')):
            im = cv2.imread(p)
            if im is None: continue
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_AREA)
            X.append(im); Y.append(0 if cls == 'youtube' else 1); M.append(mi)  # m: 0 real, 1..4 manip
    X, Y, M = np.stack(X), np.array(Y), np.array(M)
    np.savez(f, x=X, y=Y, m=M); return X, Y, M


# ---------------------------------------------------------------- degradations (RECAF protocol)
def jpeg(im, q):
    ok, b = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)

def gnoise(im, s, rng=np.random):
    return np.clip(im.astype(np.float32) + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)

def motion(im, k):
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(im, -1, ker)

def illum(im, d):
    return np.clip(im.astype(np.int16) + d, 0, 255).astype(np.uint8)

DEGRADATIONS = {
    'Blur k3': lambda im: cv2.GaussianBlur(im, (3, 3), 0),
    'Blur k7': lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    'Noise s5': lambda im: gnoise(im, 5, np.random.RandomState(1)),
    'Noise s10': lambda im: gnoise(im, 10, np.random.RandomState(2)),
    'JPEG q23': lambda im: jpeg(im, 23),
    'JPEG q40': lambda im: jpeg(im, 40),
    'Motion k9': lambda im: motion(im, 9),
    'Illum +30': lambda im: illum(im, 30),
    'Illum -30': lambda im: illum(im, -30),
}

def degrade_set(X, name):
    f = DEGRADATIONS[name]; return np.stack([f(x) for x in X])

def train_aug(im, strong):
    """Random flip; with `strong`, one random degradation drawn from ranges
    that deliberately differ from the fixed test settings above."""
    if random.random() < 0.5: im = im[:, ::-1].copy()
    if strong and random.random() < 0.6:
        c = random.randrange(4)
        if c == 0: s = random.uniform(0.3, 1.2); im = cv2.GaussianBlur(im, (0, 0), s)
        elif c == 1: im = gnoise(im, random.uniform(1, 8))
        elif c == 2: im = jpeg(im, random.randint(30, 95))
        else: im = illum(im, random.randint(-25, 25))
    return im

def to_tensor(X):
    x = (X.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


# ---------------------------------------------------------------- models
class Meso4(nn.Module):
    """MesoNet-4 (Afchar et al., 2018), trained from scratch."""
    def __init__(s):
        super().__init__()
        def blk(i, o, k, p): return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.ReLU(), nn.BatchNorm2d(o), nn.MaxPool2d(p))
        s.f = nn.Sequential(blk(3, 8, 3, 2), blk(8, 8, 5, 2), blk(8, 16, 5, 2), blk(16, 16, 5, 4))
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(16 * 3 * 3, 16), nn.LeakyReLU(0.1), nn.Dropout(0.5), nn.Linear(16, 1))
    def forward(s, x): return {'logit': s.h(s.f(x)).squeeze(1)}


def effnet():
    from efficientnet_pytorch import EfficientNet
    return EfficientNet.from_pretrained('efficientnet-b0')

class EffB0(nn.Module):
    """ImageNet-pretrained EfficientNet-B0, fine-tuned (spatial-only baseline)."""
    def __init__(s):
        super().__init__(); s.b = effnet(); s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1))
    def embed(s, x): return F.adaptive_avg_pool2d(s.b.extract_features(x), 1).flatten(1)
    def forward(s, x):
        e = s.embed(x); return {'logit': s.h(e).squeeze(1), 'emb': e}


SRM = np.array([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], np.float32)
SRM = SRM / np.array([4, 12, 2], np.float32)[:, None, None]

class FreqInput(nn.Module):
    """Builds a 4-channel frequency/noise representation: log|FFT| + 3 SRM residuals."""
    def __init__(s):
        super().__init__(); s.register_buffer('srm', torch.from_numpy(SRM)[:, None])
        s.register_buffer('mean', torch.tensor(MEAN)[None, :, None, None]); s.register_buffer('std', torch.tensor(STD)[None, :, None, None])
    def forward(s, x):
        rgb = x * s.std + s.mean
        g = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])[:, None]
        spec = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))))
        spec = (spec - spec.mean((2, 3), keepdim=True)) / (spec.std((2, 3), keepdim=True) + 1e-6)
        res = F.conv2d(g * 255., s.srm, padding=2) / 10.
        return torch.cat([spec, res.clamp(-3, 3)], 1)

class FreqCNN(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.inp = FreqInput(); s.f = nn.Sequential(blk(4, 32), blk(32, 64), blk(64, 128), blk(128, d))  # 96 -> 6x6
    def forward(s, x): return s.f(s.inp(x))

class FreqOnly(nn.Module):
    """Frequency/noise-residual stream alone (scratch)."""
    def __init__(s):
        super().__init__(); s.f = FreqCNN(); s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(s, x): return {'logit': s.h(F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)).squeeze(1)}


class CrossBlock(nn.Module):
    def __init__(s, d, h=4):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True)
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(s, q, kv):
        q = q + s.a(s.n1(q), kv, kv, need_weights=False)[0]
        return q + s.ff(s.n2(q))

class SFCANet(nn.Module):
    """Proposed Spatial-Frequency Cross-Attention Network."""
    def __init__(s, d=256, fusion='xattn'):
        super().__init__(); s.fusion = fusion
        s.b = effnet(); s.sp = nn.Conv2d(1280, d, 1); s.fq = FreqCNN(d)
        s.pos_s = nn.Parameter(torch.zeros(1, 9, d)); s.pos_f = nn.Parameter(torch.zeros(1, 36, d))
        nn.init.trunc_normal_(s.pos_s, std=0.02); nn.init.trunc_normal_(s.pos_f, std=0.02)
        if fusion == 'xattn':
            s.s2f = CrossBlock(d); s.f2s = CrossBlock(d)
        s.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        s.h = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 1))
        s.aux_s = nn.Linear(d, 1); s.aux_f = nn.Linear(d, 1)

    def forward(s, x):
        ts = s.sp(s.b.extract_features(x)).flatten(2).transpose(1, 2)  # B,9,d
        tf = s.fq(x).flatten(2).transpose(1, 2)                       # B,36,d
        ps, pf = ts.mean(1), tf.mean(1)
        if s.fusion == 'xattn':
            ts2 = s.s2f(ts + s.pos_s, tf + s.pos_f); tf2 = s.f2s(tf + s.pos_f, ts + s.pos_s)
            zs, zf = ts2.mean(1), tf2.mean(1)
        else:
            zs, zf = ps, pf
        g = s.gate(torch.cat([zs, zf], 1)); z = g * zs + (1 - g) * zf
        return {'logit': s.h(z).squeeze(1), 'aux_s': s.aux_s(ps).squeeze(1), 'aux_f': s.aux_f(pf).squeeze(1), 'emb': z, 'gate': g.mean(1)}


def build(name):
    return {'MesoNet-4': Meso4, 'EfficientNet-B0': EffB0, 'Freq-CNN': FreqOnly,
            'SFCA-Net': SFCANet, 'SFCA-Net (concat fusion)': lambda: SFCANet(fusion='concat'),
            'SFCA-Net (no deg. aug)': SFCANet, 'SFCA-Net (no aux loss)': SFCANet}[name]()


# ---------------------------------------------------------------- training / eval
@torch.no_grad()
def predict(model, X, bs=128, emb=False):
    model.eval(); P, E = [], []
    for i in range(0, len(X), bs):
        o = model(to_tensor(X[i:i + bs])); P.append(torch.sigmoid(o['logit']).numpy())
        if emb and 'emb' in o: E.append(o['emb'].numpy())
    return (np.concatenate(P), np.concatenate(E)) if emb else np.concatenate(P)


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr))
    tn = ((yh == 0) & (y == 0)).sum(); tp = ((yh == 1) & (y == 1)).sum()
    return dict(acc=accuracy_score(y, yh), bacc=0.5 * (tp / max(1, (y == 1).sum()) + tn / max(1, (y == 0).sum())),
                prec=precision_score(y, yh, zero_division=0), rec=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0), auc=roc_auc_score(y, p), eer=float((fpr[i] + fnr[i]) / 2))


def train(name, Xtr, ytr, Xva, yva, epochs=5, bs=32, log=print):
    seed_all(0); model = build(name)
    strong = name.startswith('SFCA') and 'no deg' not in name
    aux_w = 0.0 if ('no aux' in name or not name.startswith('SFCA')) else 0.3
    pre = [p for n, p in model.named_parameters() if n.startswith('b.')]
    new = [p for n, p in model.named_parameters() if not n.startswith('b.')]
    lr_new = 1e-3 if name in ('MesoNet-4', 'Freq-CNN') else 3e-4
    opt = torch.optim.AdamW([{'params': pre, 'lr': 1e-4}, {'params': new, 'lr': lr_new}], weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, lr_new], total_steps=steps, pct_start=0.15)
    pw = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32)
    lossf = lambda lg, t: F.binary_cross_entropy_with_logits(lg, t * 0.9 + 0.05, pos_weight=pw)  # label smoothing 0.1
    hist, best, best_state = [], -1, None
    for ep in range(epochs):
        model.train(); idx = np.random.permutation(len(Xtr)); tl, n = 0, 0
        for b in range(len(Xtr) // bs):
            j = idx[b * bs:(b + 1) * bs]
            x = to_tensor(np.stack([train_aug(Xtr[k], strong) for k in j])); t = torch.tensor(ytr[j], dtype=torch.float32)
            o = model(x); loss = lossf(o['logit'], t)
            if aux_w: loss = loss + aux_w * (lossf(o['aux_s'], t) + lossf(o['aux_f'], t))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(j); n += len(j)
        pv = predict(model, Xva); mv = metrics(yva, pv)
        vl = F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yva, dtype=torch.float32)).item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_loss=vl, val_auc=mv['auc'], val_acc=mv['acc']))
        log(f'  [{name}] ep{ep + 1} loss {tl / n:.4f} val_loss {vl:.4f} val_auc {mv["auc"]:.4f} val_acc {mv["acc"]:.4f}')
        if mv['auc'] > best: best = mv['auc']; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return model, hist

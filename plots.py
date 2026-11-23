"""Generates every graph from the saved experiment results.
Usage: python3 plots.py  (reads results/*.json and results/ckpt/*.pt, writes figures/)"""
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.manifold import TSNE
from dfd import *

torch.set_num_threads(2)
RES = 'results'; OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.5, 'axes.axisbelow': True, 'savefig.bbox': 'tight', 'savefig.dpi': 200})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK2 = '#52514e'
MAIN = ['MesoNet-4', 'Freq-CNN', 'EfficientNet-B0', 'SFCA-Net']
ABL = ['SFCA-Net', 'SFCA-Net (concat fusion)', 'SFCA-Net (no deg. aug)']
COL = {m: C[i] for i, m in enumerate(MAIN)}
COL.update({'SFCA-Net (concat fusion)': C[4], 'SFCA-Net (no deg. aug)': C[6]})

R = {}
for f in glob.glob(f'{RES}/*__*.json'):
    r = json.load(open(f)); R[(r['model'], r['protocol'])] = r
have = lambda m, p='all': (m, p) in R
Xte, yte, mte = load_split('test')
def save(fig, name): fig.savefig(f'{OUT}/{name}.png'); fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig); print('  ', name)

# ---------------------------------------------------------------- 1. dataset samples and mean spectra
def fig_samples():
    fig, axs = plt.subplots(2, 5, figsize=(8, 3.4))
    names = ['Real'] + MANIPS; rng = np.random.RandomState(3)
    for j in range(5):
        idx = rng.choice(np.where(mte == j)[0], 2, replace=False)
        for i in range(2):
            axs[i, j].imshow(Xte[idx[i]]); axs[i, j].axis('off')
        axs[0, j].set_title(names[j])
    fig.suptitle('Test-set face crops (96 x 96) from FaceForensics++', y=1.0); save(fig, '01_dataset_samples')

def fig_spectra():
    Xtr, ytr, mtr = load_split('train')
    g = Xtr.astype(np.float32).mean(-1)
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))))
    names = ['Real'] + MANIPS
    mean = [spec[mtr == k].mean(0) for k in range(5)]
    fig, axs = plt.subplots(1, 5, figsize=(9, 2.2))
    for k, ax in enumerate(axs):
        d = mean[k] - mean[0] if k else mean[0]
        im = ax.imshow(d, cmap='RdBu_r' if k else 'viridis', vmin=(-0.25 if k else None), vmax=(0.25 if k else None))
        ax.set_title(names[k] if k == 0 else f'{names[k]} - Real', fontsize=8); ax.axis('off')
    fig.suptitle('Mean log-amplitude FFT spectrum (real) and difference per manipulation', y=1.05); save(fig, '02_frequency_spectra')
    # radial profile
    h = spec.shape[1]; yy, xx = np.indices((h, h)); rr = np.hypot(yy - h // 2, xx - h // 2).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for k in range(5):
        prof = np.bincount(rr.ravel(), mean[k].ravel()) / np.bincount(rr.ravel())
        ax.plot(prof[:h // 2], color=C[k] if k else '#0b0b0b', lw=1.6 if k == 0 else 1.1, label=names[k])
    ax.set_xlabel('Radial frequency (bins)'); ax.set_ylabel('Mean log amplitude'); ax.legend(frameon=False)
    ax.set_title('Azimuthally averaged spectrum'); save(fig, '03_radial_spectrum')

# ---------------------------------------------------------------- 2. training curves
def fig_curves():
    ms = [m for m in MAIN + ABL[1:] if have(m)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3))
    for m in ms:
        h = R[(m, 'all')]['hist']; e = [x['epoch'] for x in h]
        a.plot(e, [x['train_loss'] for x in h], color=COL[m], label=m, marker='o', ms=3)
        a.plot(e, [x['val_loss'] for x in h], color=COL[m], ls='--', marker='o', ms=3)
        b.plot(e, [x['val_auc'] for x in h], color=COL[m], label=m, marker='o', ms=3)
    a.set_xlabel('Epoch'); a.set_ylabel('Loss'); a.set_title('Training (solid) and validation (dashed) loss')
    b.set_xlabel('Epoch'); b.set_ylabel('Validation AUC'); b.set_title('Validation AUC'); b.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, '04_training_curves')

# ---------------------------------------------------------------- 3. ROC + confusion matrices
def fig_roc_cm():
    ms = [m for m in MAIN + ABL[1:] if have(m)]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for m in ms:
        p = np.array(R[(m, 'all')]['pred']); fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=COL[m], lw=1.6, label=f'{m} (AUC {R[(m, "all")]["E"]["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], color=INK2, ls=':', lw=0.8); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title('ROC, in-dataset test (all manipulations)'); ax.legend(frameon=False, fontsize=7, loc='lower right'); save(fig, '05_roc_curves')
    ms = [m for m in MAIN if have(m)]
    fig, axs = plt.subplots(1, len(ms), figsize=(2.3 * len(ms), 2.4))
    for ax, m in zip(np.atleast_1d(axs), ms):
        p = np.array(R[(m, 'all')]['pred']); cm = confusion_matrix(yte, p >= 0.5)
        ax.imshow(cm, cmap='Blues'); ax.grid(False)
        for i in range(2):
            for j in range(2): ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Real', 'Fake']); ax.set_yticks([0, 1]); ax.set_yticklabels(['Real', 'Fake'])
        ax.set_xlabel('Predicted'); ax.set_title(m, fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel('True'); fig.tight_layout(); save(fig, '06_confusion_matrices')

# ---------------------------------------------------------------- 4. per-manipulation AUC + metric table bar
def fig_per_manip():
    ms = [m for m in MAIN if have(m)]; x = np.arange(4); w = 0.8 / len(ms)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for i, m in enumerate(ms):
        v = [R[(m, 'all')]['per_manip_auc'][k] for k in MANIPS]
        bars = ax.bar(x + (i - (len(ms) - 1) / 2) * w, v, w * 0.92, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(MANIPS); ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC (real vs. manipulation)')
    ax.axhline(0.5, color=INK2, ls=':', lw=0.8); ax.legend(frameon=False, ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis='x', visible=False); save(fig, '07_per_manipulation_auc')
    mets = ['acc', 'bacc', 'prec', 'rec', 'f1', 'auc']; lab = ['Accuracy', 'Bal. acc.', 'Precision', 'Recall', 'F1', 'AUC']
    fig, ax = plt.subplots(figsize=(6.5, 2.8)); x = np.arange(len(mets))
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, [R[(m, 'all')]['E'][k] for k in mets], w * 0.92, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0.3, 1.0); ax.set_ylabel('Score')
    ax.legend(frameon=False, ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.15)); ax.grid(axis='x', visible=False)
    save(fig, '08_intra_metrics')

# ---------------------------------------------------------------- 5. robustness
def fig_robust():
    ms = [m for m in MAIN + ABL[1:] if have(m)]; degs = list(DEGRADATIONS)
    M = np.array([[R[(m, 'all')]['E']['acc']] + [R[(m, 'all')]['R'][d]['acc'] for d in degs] for m in ms]) * 100
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(ms) + 1.1))
    im = ax.imshow(M, cmap='Blues', vmin=40, vmax=100, aspect='auto'); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=7, color='white' if M[i, j] > 78 else 'black')
    ax.set_xticks(range(len(degs) + 1)); ax.set_xticklabels(['Clean'] + degs, fontsize=7.5); ax.xaxis.tick_top()
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8); fig.colorbar(im, ax=ax, label='Accuracy (%)', fraction=0.02)
    save(fig, '09_robustness_heatmap')
    fig, ax = plt.subplots(figsize=(8, 2.9)); x = np.arange(len(degs)); w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, [R[(m, 'all')]['R'][d]['auc'] for d in degs], w * 0.92, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(degs, fontsize=7.5); ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC under distortion')
    ax.axhline(0.5, color=INK2, ls=':', lw=0.8); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.2)); save(fig, '10_robustness_auc')

# ---------------------------------------------------------------- 6. leave-one-manipulation-out generalisation
def fig_loo():
    ms = [m for m in MAIN if all(have(m, p) for p in MANIPS)]
    if not ms: return
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.9)); x = np.arange(4); w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        axs[0].bar(x + (i - (len(ms) - 1) / 2) * w, [R[(m, p)]['C']['auc'] for p in MANIPS], w * 0.92, color=COL[m], label=m)
        axs[1].bar(x + (i - (len(ms) - 1) / 2) * w, [R[(m, p)]['E']['auc'] - R[(m, p)]['C']['auc'] for p in MANIPS], w * 0.92, color=COL[m])
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels([f'unseen:\n{p}' for p in MANIPS], fontsize=7.5); ax.grid(axis='x', visible=False)
    axs[0].set_ylim(0.3, 1.0); axs[0].axhline(0.5, color=INK2, ls=':', lw=0.8); axs[0].set_ylabel('AUC on unseen manipulation')
    axs[0].set_title('(a) Cross-manipulation AUC'); axs[1].set_ylabel('AUC(seen) - AUC(unseen)'); axs[1].set_title('(b) Generalisation gap')
    axs[0].legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout(); save(fig, '11_cross_manipulation')
    # heatmap: rows held-out protocol, cols test manipulation AUC
    fig, axs = plt.subplots(1, len(ms), figsize=(2.6 * len(ms), 2.7))
    for ax, m in zip(np.atleast_1d(axs), ms):
        H = np.array([[R[(m, p)]['per_manip_auc'][t] for t in MANIPS] for p in MANIPS])
        im = ax.imshow(H, cmap='Blues', vmin=0.4, vmax=1.0); ax.grid(False)
        for i in range(4):
            for j in range(4): ax.text(j, i, f'{H[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if H[i, j] > 0.8 else 'black',
                                      weight='bold' if i == j else 'normal')
        ab = ['DF', 'F2F', 'FS', 'NT']; ax.set_xticks(range(4)); ax.set_xticklabels(ab); ax.set_yticks(range(4)); ax.set_yticklabels(ab)
        ax.set_title(m, fontsize=8); ax.set_xlabel('Test manipulation')
    np.atleast_1d(axs)[0].set_ylabel('Held out in training'); fig.colorbar(im, ax=axs, fraction=0.015, label='AUC')
    save(fig, '12_loo_heatmaps')

# ---------------------------------------------------------------- 7. RECAF (Pandey et al. 2026) on our runs
def recaf_table():
    ms = [m for m in MAIN if all(have(m, p) for p in MANIPS)]
    if not ms: return None
    vals = {m: [(np.mean([R[(m, p)]['R'][d]['acc'] for d in DEGRADATIONS]), R[(m, p)]['E']['acc'], R[(m, p)]['C']['bacc']) for p in MANIPS] for m in ms}
    X = np.array([[v[k] for m in ms for v in vals[m]] for k in range(3)])  # 3 x (M*D)
    n = X.shape[1]; P = X / X.sum(1, keepdims=True); H = -(P * np.log(P)).sum(1) / np.log(n); w = (1 - H) / (1 - H).sum()
    a1 = X[2].std() / X[2].mean(); a2 = X[0].std() / X[0].mean()
    out = {}
    for m in ms:
        sc = []
        for Rv, Ev, Cv in vals[m]:
            wgm = Rv ** w[0] * Ev ** w[1] * Cv ** w[2]; pen = np.exp(-a1 * abs(Ev - Cv) - a2 * (1 - Rv)); sc.append(wgm * pen)
        out[m] = dict(R=float(np.mean([v[0] for v in vals[m]])), E=float(np.mean([v[1] for v in vals[m]])), C=float(np.mean([v[2] for v in vals[m]])), RECAF=float(np.mean(sc)))
    out['_weights'] = dict(wR=float(w[0]), wE=float(w[1]), wC=float(w[2]), alpha1=float(a1), alpha2=float(a2))
    json.dump(out, open(f'{RES}/recaf_scores.json', 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(6, 2.8)); x = np.arange(len(ms)); w_ = 0.2
    for j, (k, col) in enumerate([('E', C[0]), ('R', C[3]), ('C', C[7]), ('RECAF', '#0b0b0b')]):
        bars = ax.bar(x + (j - 1.5) * w_, [out[m][k] for m in ms], w_ * 0.92, color=col, label=k)
        if k == 'RECAF':
            for b_ in bars: ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01, f'{b_.get_height():.3f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ms); ax.set_ylim(0.3, 1.0); ax.axhline(0.5, color=INK2, ls=':', lw=0.8)
    ax.set_title('RECAF assessment over 4 leave-one-manipulation-out protocols'); ax.legend(frameon=False, ncol=4, fontsize=7)
    ax.grid(axis='x', visible=False); save(fig, '13_recaf_scores')
    return out

# ---------------------------------------------------------------- 8. ablation + efficiency
def fig_ablation_eff():
    ab = [m for m in ABL if have(m)]
    if len(ab) > 1:
        mets = [('E', 'auc', 'Clean AUC'), ('R', None, 'Mean distorted AUC')]
        fig, ax = plt.subplots(figsize=(5, 2.6)); x = np.arange(2); w = 0.8 / len(ab)
        for i, m in enumerate(ab):
            v = [R[(m, 'all')]['E']['auc'], np.mean([R[(m, 'all')]['R'][d]['auc'] for d in DEGRADATIONS])]
            bars = ax.bar(x + (i - (len(ab) - 1) / 2) * w, v, w * 0.92, color=COL[m], label=m)
            for b_ in bars: ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.005, f'{b_.get_height():.3f}', ha='center', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([m[2] for m in mets]); ax.set_ylim(0.5, 1.0); ax.legend(frameon=False, fontsize=7)
        ax.set_title('Ablation of SFCA-Net components'); ax.grid(axis='x', visible=False); save(fig, '14_ablation')
    ms = [m for m in MAIN if have(m)]
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.5))
    for ax, (k, lab) in zip(axs, [('params', 'Parameters (millions)'), ('infer_ms_per_img', 'CPU inference (ms / image)'), ('train_time_s', 'Training time (min)')]):
        v = [R[(m, 'all')][k] / (1e6 if k == 'params' else 60 if k == 'train_time_s' else 1) for m in ms]
        bars = ax.bar(range(len(ms)), v, color=[COL[m] for m in ms]); ax.set_title(lab)
        for b_ in bars: ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f'{b_.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=25, ha='right', fontsize=7.5); ax.grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, '15_efficiency')

# ---------------------------------------------------------------- 9. t-SNE + Grad-CAM (needs checkpoints)
def load_model(name, proto='all'):
    p = f'{RES}/ckpt/{name}__{proto}.pt'
    if not os.path.exists(p): return None
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()): m = build(name)
    m.load_state_dict(torch.load(p, map_location='cpu')); m.eval(); return m

def fig_tsne():
    ms = [m for m in ['EfficientNet-B0', 'SFCA-Net'] if have(m)]
    fig, axs = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.6))
    names = ['Real'] + MANIPS
    for ax, m in zip(np.atleast_1d(axs), ms):
        model = load_model(m); _, E = predict(model, Xte, emb=True)
        Z = TSNE(2, random_state=0, perplexity=30, init='pca').fit_transform(E)
        for k in range(5):
            s = mte == k; ax.scatter(Z[s, 0], Z[s, 1], s=7, color=C[k] if k else '#0b0b0b', label=names[k], alpha=0.75, lw=0)
        ax.set_title(f't-SNE of {m} embeddings (test)'); ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7, markerscale=2); fig.tight_layout(); save(fig, '16_tsne_embeddings')

def gradcam(model, x):
    feats = {}
    h = model.b._conv_head.register_forward_hook(lambda mod, i, o: feats.__setitem__('a', o))
    x = x.clone().requires_grad_(False); out = model(x)['logit']
    A = feats['a']; g = torch.autograd.grad(out.sum(), A)[0]; h.remove()
    cam = F.relu((g.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8); return cam.detach().numpy(), torch.sigmoid(out).detach().numpy()

def fig_gradcam():
    model = load_model('SFCA-Net')
    if model is None: return
    rng = np.random.RandomState(7); idx = [rng.choice(np.where(mte == k)[0]) for k in range(5)]
    cam, p = gradcam(model, to_tensor(Xte[idx])); names = ['Real'] + MANIPS
    fig, axs = plt.subplots(2, 5, figsize=(8, 3.6))
    for j in range(5):
        axs[0, j].imshow(Xte[idx[j]]); axs[0, j].set_title(names[j], fontsize=8)
        axs[1, j].imshow(Xte[idx[j]]); axs[1, j].imshow(cam[j], cmap='jet', alpha=0.45)
        axs[1, j].set_title(f'P(fake) = {p[j]:.2f}', fontsize=8)
        for a in axs[:, j]: a.axis('off')
    fig.suptitle('Grad-CAM of SFCA-Net spatial branch', y=1.0); save(fig, '17_gradcam')

if __name__ == '__main__':
    only = sys.argv[1:]
    steps = [fig_samples, fig_spectra, fig_curves, fig_roc_cm, fig_per_manip, fig_robust, fig_loo, recaf_table, fig_ablation_eff, fig_tsne, fig_gradcam]
    for f in steps:
        if only and f.__name__ not in only: continue
        try: f()
        except Exception as e: print('  skip', f.__name__, type(e).__name__, e)

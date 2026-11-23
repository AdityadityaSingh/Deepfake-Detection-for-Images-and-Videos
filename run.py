import argparse, json, os, time, numpy as np, torch
from dfd import *

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='MesoNet-4,Freq-CNN,EfficientNet-B0,SFCA-Net')
ap.add_argument('--protocols', default='all')  # all, or comma list of held-out manip names
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--out', default='/home/claude/dfd/results')
a = ap.parse_args()
torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True); os.makedirs(a.out + '/ckpt', exist_ok=True)
logf = open(a.out + '/log.txt', 'a')
def log(s): print(s, flush=True); logf.write(s + '\n'); logf.flush()

Xtr0, ytr0, mtr0 = load_split('train'); Xte, yte, mte = load_split('test')
rng = np.random.RandomState(0); va_mask = np.zeros(len(ytr0), bool)
for m in np.unique(mtr0):
    ii = np.where(mtr0 == m)[0]; va_mask[rng.choice(ii, int(0.1 * len(ii)), replace=False)] = True
if a.quick:
    keep = rng.rand(len(ytr0)) < 0.1; va_mask &= keep | va_mask

for proto in a.protocols.split(','):
    held = None if proto == 'all' else MANIPS.index(proto) + 1
    keep_tr = np.ones(len(mtr0), bool) if held is None else (mtr0 != held)
    tr = (~va_mask) & keep_tr; va = va_mask & keep_tr
    if a.quick: tr &= rng.rand(len(tr)) < 0.1
    e_mask = np.ones(len(yte), bool) if held is None else (mte != held)
    for name in a.models.split(','):
        key = f'{name}__{proto}'; fo = f'{a.out}/{key}.json'
        if os.path.exists(fo) and not a.quick: log(f'skip {key}'); continue
        log(f'=== {key}: train {tr.sum()} (fake {ytr0[tr].sum()}) val {va.sum()} E-test {e_mask.sum()}')
        t0 = time.time()
        model, hist = train(name, Xtr0[tr], ytr0[tr], Xtr0[va], ytr0[va], epochs=1 if a.quick else a.epochs, log=log)
        ttrain = time.time() - t0
        res = dict(model=name, protocol=proto, hist=hist, train_time_s=ttrain,
                   params=sum(p.numel() for p in model.parameters()))
        t1 = time.time(); p_all = predict(model, Xte); res['infer_ms_per_img'] = 1000 * (time.time() - t1) / len(Xte)
        res['E'] = metrics(yte[e_mask], p_all[e_mask])
        res['per_manip_auc'] = {MANIPS[k - 1]: float(roc_auc_score(np.r_[np.zeros((mte == 0).sum()), np.ones((mte == k).sum())],
                                np.r_[p_all[mte == 0], p_all[mte == k]])) for k in range(1, 5)}
        res['per_manip_recall'] = {MANIPS[k - 1]: float((p_all[mte == k] >= 0.5).mean()) for k in range(1, 5)}
        res['real_specificity'] = float((p_all[mte == 0] < 0.5).mean())
        if held:
            c_mask = (mte == 0) | (mte == held); res['C'] = metrics(yte[c_mask], p_all[c_mask])
        res['R'] = {}
        for dn in DEGRADATIONS:
            pd_ = predict(model, degrade_set(Xte[e_mask], dn)); res['R'][dn] = metrics(yte[e_mask], pd_)
        res['pred'] = p_all.tolist()
        log(f'  -> E acc {res["E"]["acc"]:.4f} auc {res["E"]["auc"]:.4f} | R mean acc {np.mean([v["acc"] for v in res["R"].values()]):.4f}'
            + (f' | C bacc {res["C"]["bacc"]:.4f} auc {res["C"]["auc"]:.4f}' if held else '') + f' | {ttrain / 60:.1f} min')
        if a.quick: continue
        json.dump(res, open(fo, 'w'))
        if proto == 'all' or name == 'SFCA-Net': torch.save(model.state_dict(), f'{a.out}/ckpt/{key}.pt')

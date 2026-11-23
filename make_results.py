"""Writes RESULTS.md from results/*.json."""
import glob, json, os, numpy as np
from dfd import MANIPS, DEGRADATIONS
R = {}
for f in glob.glob('results/*__*.json'):
    r = json.load(open(f)); R[(r['model'], r['protocol'])] = r
ORDER = ['MesoNet-4', 'Freq-CNN', 'EfficientNet-B0', 'SFCA-Net', 'SFCA-Net (concat fusion)', 'SFCA-Net (no deg. aug)']
L = ['# Results\n', 'All numbers come from `results/*.json`, produced by `run.py` on a 2-core CPU. '
     'Test set: 624 FF++ face crops (201 real, 423 fake). Threshold 0.5. Best epoch picked by validation AUC.\n']
L += ['## In-dataset (train and test on all four manipulations, 6 epochs)\n',
      '| Model | Params (M) | Acc (%) | Bal. acc (%) | Precision | Recall | F1 | AUC | EER | CPU ms/img |', '|---|---|---|---|---|---|---|---|---|---|']
for m in ORDER:
    if (m, 'all') not in R: continue
    r = R[(m, 'all')]; E = r['E']
    L.append(f"| {m} | {r['params']/1e6:.2f} | {100*E['acc']:.2f} | {100*E['bacc']:.2f} | {E['prec']:.3f} | {E['rec']:.3f} | {E['f1']:.3f} | {E['auc']:.3f} | {E['eer']:.3f} | {r['infer_ms_per_img']:.1f} |")
L += ['\n### AUC per manipulation (real test vs. one manipulation)\n', '| Model | ' + ' | '.join(MANIPS) + ' |', '|---' * 5 + '|']
for m in ORDER:
    if (m, 'all') in R: L.append(f'| {m} | ' + ' | '.join(f"{R[(m, 'all')]['per_manip_auc'][k]:.3f}" for k in MANIPS) + ' |')
L += ['\n## Robustness (accuracy %, model trained on clean data)\n', '| Model | Clean | ' + ' | '.join(DEGRADATIONS) + ' | Mean |', '|---' * 12 + '|']
for m in ORDER:
    if (m, 'all') not in R: continue
    r = R[(m, 'all')]; v = [r['R'][d]['acc'] for d in DEGRADATIONS]
    L.append(f"| {m} | {100*r['E']['acc']:.1f} | " + ' | '.join(f'{100*x:.1f}' for x in v) + f' | {100*np.mean(v):.1f} |')
loo = [m for m in ORDER if all((m, p) in R for p in MANIPS)]
if loo:
    L += ['\n## Leave-one-manipulation-out (2 epochs)\n', 'C = test on real + the manipulation held out of training. E = test on real + seen manipulations.\n',
          '| Model | Held out | E AUC | C AUC | C bal. acc (%) | Gap (E-C AUC) |', '|---|---|---|---|---|---|']
    for m in loo:
        for p in MANIPS:
            r = R[(m, p)]; L.append(f"| {m} | {p} | {r['E']['auc']:.3f} | {r['C']['auc']:.3f} | {100*r['C']['bacc']:.2f} | {r['E']['auc']-r['C']['auc']:.3f} |")
        L.append(f"| **{m}** | **mean** | {np.mean([R[(m,p)]['E']['auc'] for p in MANIPS]):.3f} | {np.mean([R[(m,p)]['C']['auc'] for p in MANIPS]):.3f} | {100*np.mean([R[(m,p)]['C']['bacc'] for p in MANIPS]):.2f} | |")
if os.path.exists('results/recaf_scores.json'):
    rc = json.load(open('results/recaf_scores.json')); w = rc.pop('_weights')
    L += ['\n## RECAF (Pandey et al., 2026) over the four leave-one-out protocols\n',
          f"Entropy weights: wR = {w['wR']:.4f}, wE = {w['wE']:.4f}, wC = {w['wC']:.4f}; alpha1 = {w['alpha1']:.4f}, alpha2 = {w['alpha2']:.4f}.\n",
          '| Model | R | E | C | RECAF |', '|---|---|---|---|---|']
    for m, v in rc.items(): L.append(f"| {m} | {v['R']:.3f} | {v['E']:.3f} | {v['C']:.3f} | {v['RECAF']:.3f} |")
if os.path.exists('results/video_inference.json'):
    L += ['\n## Video inference (SFCA-Net, up to 120 face frames per video)\n', '| Video | Label | Frames | P(fake) mean-logit | P(fake) top-25% | Fake frames | Verdict |', '|---|---|---|---|---|---|---|']
    for v in json.load(open('results/video_inference.json')):
        L.append(f"| {v['video']} | {v['label']} | {v['frames_scored']} | {v['p_fake_mean_logit']:.3f} | {v['p_fake_topk']:.3f} | {100*v['frac_frames_fake']:.0f}% | {v['verdict']} |")
    L.append('\nThese FF++ sample videos may overlap with the training videos; treat this as a pipeline demo, not a test result.')
L += ['\n## Notes\n', '- Small images (96x96), few epochs and CPU-only training keep absolute numbers well below GPU results in the literature.',
      '- Leave-one-out runs use 2 epochs, so compare models within that table, not against the 6-epoch in-dataset table.']
open('RESULTS.md', 'w').write('\n'.join(L) + '\n'); print('RESULTS.md written')

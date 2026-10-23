"""Summarize toy/slopes/train_results.jsonl: same-family table (mean over seeds), data-efficiency figure, cross-family table.

usage: uv run python -I toy/slopes/analyze_training.py
"""
import json, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
rows = [json.loads(l) for l in open('toy/slopes/train_results.jsonl')]
def agg(filt):
    sel = [r for r in rows if filt(r)]
    if not sel: return None
    return {m: (float(np.mean([r[m] for r in sel])), float(np.std([r[m] for r in sel])), len(sel)) for m in ('mae', 'rmse', 'ssim')}
fams = ['flatvel-a', 'flatfault-a']; variants = ['raw', 'phys', 'both', 'physR', 'bothR', 'physL', 'physP']; ns = [25, 100, 452]
print('Same family, test = models 452-499 of the file (mean +- std over seeds)')
for fam in fams:
    g = agg(lambda r: r['variant'] == 'geo' and r['test'] == fam)
    print(f'\n{fam}: geo (no learning)  MAE {g["mae"][0]:.4f}  RMSE {g["rmse"][0]:.4f}  SSIM {g["ssim"][0]:.4f}' if g else f'\n{fam}: geo missing')
    g = agg(lambda r: r['variant'] == 'refr' and r['test'] == fam)
    if g: print(f'{fam}: refr two-layer channel (no learning)  MAE {g["mae"][0]:.4f}  RMSE {g["rmse"][0]:.4f}  SSIM {g["ssim"][0]:.4f}')
    for n in ns:
        for v in variants:
            a = agg(lambda r: r['variant'] == v and r['train'] == fam and r['test'] == fam and r['ntrain'] == n)
            if a: print(f'  n={n:3d} {v:5s}  MAE {a["mae"][0]:.4f}+-{a["mae"][1]:.4f}  RMSE {a["rmse"][0]:.4f}  SSIM {a["ssim"][0]:.4f}  ({a["mae"][2]} seeds)')
print('\nCross family (train on one, test on the other, n=452)')
for tr, te in [('flatvel-a', 'flatfault-a'), ('flatfault-a', 'flatvel-a')]:
    for v in variants:
        a = agg(lambda r: r['variant'] == v and r['train'] == tr and r['test'] == te)
        if a: print(f'  {tr} -> {te} {v:5s}  MAE {a["mae"][0]:.4f}  RMSE {a["rmse"][0]:.4f}  SSIM {a["ssim"][0]:.4f}')
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
colors = {'raw': '#888888', 'phys': '#1f77b4', 'both': '#d62728', 'physR': '#17becf', 'bothR': '#9467bd', 'physL': '#bcbd22', 'physP': '#8c564b'}
for i, fam in enumerate(fams):
    for v in ['raw', 'phys', 'both', 'physR', 'bothR']:
        pts = [(n, agg(lambda r: r['variant'] == v and r['train'] == fam and r['test'] == fam and r['ntrain'] == n)) for n in ns]
        pts = [(n, a) for n, a in pts if a]
        if pts: ax[i].errorbar([p[0] for p in pts], [p[1]['mae'][0] for p in pts], yerr=[p[1]['mae'][1] for p in pts], marker='o', label=v, color=colors[v], capsize=3)
    g = agg(lambda r: r['variant'] == 'geo' and r['test'] == fam)
    if g: ax[i].axhline(g['mae'][0], color='k', ls=':', label='geometric start, no training')
    ax[i].set_xscale('log'); ax[i].set_xticks(ns); ax[i].set_xticklabels(ns); ax[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[i].set_xlabel('training models'); ax[i].set_ylabel('MAE on [-1, 1]'); ax[i].set_title(fam); ax[i].legend(fontsize=8)
plt.tight_layout(); plt.savefig('toy/slopes/data_efficiency.png', dpi=110); print('\nwrote toy/slopes/data_efficiency.png')

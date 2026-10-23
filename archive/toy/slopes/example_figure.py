"""One test model per family, from recording to prediction: the model whose joint-model error is the median of its family.
usage: uv run python -I toy/slopes/example_figure.py"""
import glob, numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from fwi import metrics
B = 'flatvel-a+flatfault-a'; fams = ['flatvel-a', 'flatfault-a']
fig, ax = plt.subplots(len(fams), 7, figsize=(21, 3.4 * len(fams)))
for r, fam in enumerate(fams):
    J = np.load(f'toy/slopes/preds2/both_{B}_{fam}_n452_s0.npz'); R = np.load(f'toy/slopes/preds2/raw_{B}_{fam}_n452_s0.npz'); idx = list(J['idx'])
    errs = [metrics(p, np.load(f'toy/slopes/feat/{fam}/{k:03d}.npz')['model'])['mae'] for p, k in zip(J['pred'], idx)]
    j = int(np.argsort(errs)[len(errs) // 2]); k = idx[j]; d = np.load(f'toy/slopes/feat/{fam}/{k:03d}.npz'); dr = np.load(f'toy/slopes/feat_refr/{fam}/{k:03d}.npz')
    true = d['model']; lo, hi = true.min(), true.max(); g = d['raw'][2] * (1 + np.arange(100) * 0.02)[:, None]
    m_geo = metrics(d['ch'][0] * 1500 + 3000, true)['mae']; m_raw = metrics(R['pred'][idx.index(k)], true)['mae']; m_j = errs[j]
    panels = [(g, 'one shot gather (gained)', dict(cmap='gray', vmin=-np.abs(g).max() * 0.3, vmax=np.abs(g).max() * 0.3, aspect='auto')),
              (d['ch'][0] * 1500 + 3000, f'geometric start, MAE {m_geo:.3f}', dict(cmap='viridis', vmin=lo, vmax=hi)),
              (d['ch'][1], 'migrated reflectivity', dict(cmap='magma')),
              (dr['ch_refr'][0], 'refractor likelihood', dict(cmap='magma', vmin=0, vmax=1)),
              (R['pred'][idx.index(k)], f'raw gathers only, MAE {m_raw:.3f}', dict(cmap='viridis', vmin=lo, vmax=hi)),
              (J['pred'][j], f'physics + raw, MAE {m_j:.3f}', dict(cmap='viridis', vmin=lo, vmax=hi)),
              (true, f'{fam} model {k}: truth', dict(cmap='viridis', vmin=lo, vmax=hi))]
    for c, (im, t, kw) in enumerate(panels): ax[r, c].imshow(im, **kw); ax[r, c].set_title(t, fontsize=9); ax[r, c].axis('off')
plt.suptitle('One model for both families (trained jointly, 452 models each); example = median-error test model of each family', fontsize=10)
plt.tight_layout(); plt.savefig('toy/slopes/examples_per_family.png', dpi=90); print('wrote toy/slopes/examples_per_family.png')

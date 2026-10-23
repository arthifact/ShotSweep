"""Step 4, testing: score every trained network on the held-out models (452-499) of every family.

Metrics are the OpenFWI ones on velocity scaled to [-1, 1] (1500-4500 m/s): MAE, RMSE, SSIM; MAE x 1500 is m/s.
Writes results/scores.jsonl (one row per network and family), results/table.md (mean over seeds, per family),
results/preds/<network>.npz and results/figures/examples.png (one median-error test model per family).

usage: uv run python -I openfwi/4_test.py [--checkpoints work/models/*.pt]
"""
import argparse, glob, importlib.util, json, os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'physics'))
from metrics import metrics
spec = importlib.util.spec_from_file_location('train', os.path.join(HERE, '3_train.py')); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
RES = os.path.join(HERE, 'results')

@torch.no_grad()
def predict(ck, d):
    net = T.build(ck['model']).to(T.DEV); net.load_state_dict(ck['state']); net.eval(); out = []
    for i in range(0, len(d['y']), 48):
        ids = np.arange(i, min(i + 48, len(d['y']))); g, im, vr, _ = T.batch(d, ids, ck['scale'])
        out.append(net(g, im, vr).cpu().numpy())
    return np.clip(np.concatenate(out) * 1500 + 3000, 1500, 4500)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoints', default=os.path.join(HERE, 'work', 'models', '*.pt')); ap.add_argument('--families', default='all'); a = ap.parse_args()
    os.makedirs(os.path.join(RES, 'preds'), exist_ok=True); os.makedirs(os.path.join(RES, 'figures'), exist_ok=True)
    d = T.load_data('test', a.families); true = d['y'] * 1500 + 3000; rows = []; preds = {}
    for p in sorted(glob.glob(a.checkpoints)):
        ck = torch.load(p, map_location='cpu', weights_only=False); name = os.path.basename(p)[:-3]; pr = predict(ck, d); preds[name] = pr
        np.savez_compressed(os.path.join(RES, 'preds', f'{name}.npz'), pred=pr.astype(np.float32), family=d['family'], index=d['index'])
        per = [metrics(x, t) for x, t in zip(pr, true)]
        for f in dict.fromkeys(d['family']):
            sel = [m for m, ff in zip(per, d['family']) if ff == f]
            rows.append(dict(network=name, model=ck['model'], loss=ck['loss'], seed=ck['seed'], params=ck['params'], family=f, n=len(sel),
                             **{k: round(float(np.mean([m[k] for m in sel])), 4) for k in ('mae', 'rmse', 'ssim')}))
        print(name, 'mean MAE', round(float(np.mean([m['mae'] for m in per])), 4), 'SSIM', round(float(np.mean([m['ssim'] for m in per])), 4), flush=True)
    with open(os.path.join(RES, 'scores.jsonl'), 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    # table: mean over seeds per (model, loss) and family
    keys = sorted({(r['model'], r['loss']) for r in rows}); fams = list(dict.fromkeys(d['family'])); lines = []
    for metric in ('mae', 'ssim'):
        lines += [f'\n### {metric.upper()} (mean over seeds; {"lower" if metric == "mae" else "higher"} is better)\n', '| family | ' + ' | '.join(f'{m} ({l})' for m, l in keys) + ' |', '|---' * (len(keys) + 1) + '|']
        for f in fams + ['mean']:
            cells = []
            for m, l in keys:
                v = [r[metric] for r in rows if r['model'] == m and r['loss'] == l and (r['family'] == f or f == 'mean')]
                cells.append(f'{np.mean(v):.4f}')
            lines.append(f'| {f} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(RES, 'table.md'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
    # figure: one median-error test model per family, truth / InversionNet / ours
    base = next((n for n in preds if n.startswith('inversionnet_l1ssim')), None); ours = next((n for n in preds if n.startswith('ours_l1ssim')), None)
    if base and ours:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(fams), 3, figsize=(9, 3 * len(fams)), squeeze=False)
        for r_, f in enumerate(fams):
            ids = np.where(d['family'] == f)[0]; e = [metrics(preds[ours][i], true[i])['mae'] for i in ids]; i = ids[int(np.argsort(e)[len(e) // 2])]
            lo, hi = true[i].min(), true[i].max()
            for c, (im, t) in enumerate([(true[i], f'{f} model {d["index"][i]}: truth'), (preds[base][i], f'InversionNet, MAE {metrics(preds[base][i], true[i])["mae"]:.3f}'),
                                         (preds[ours][i], f'ours, MAE {metrics(preds[ours][i], true[i])["mae"]:.3f}')]):
                ax[r_, c].imshow(im, cmap='viridis', vmin=lo, vmax=hi); ax[r_, c].set_title(t, fontsize=9); ax[r_, c].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(RES, 'figures', 'examples.png'), dpi=70)

if __name__ == '__main__':
    main()

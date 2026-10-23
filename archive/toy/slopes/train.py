"""One model for every family: physics channels (+ raw gathers) -> velocity. No per-family code.

Inputs per model (same for every family):
  physics  8 channels: v_geo, reflectivity, dip, cross-shot agreement, illumination, FWI gradient (features.py) and
           refractor likelihood, two-layer velocity (features_refr.py)
  raw      the 5 gathers (5 x 100 x 35, time gain, standardized) -> PCA 128 (fit on the training set) -> MLP -> 70 x 70 map
Model: inputs -> 6-layer dilated conv refiner (64 channels, dilations 1 2 4 8 4 1, receptive field 41 cells) ->
correction added to v_geo (raw only: the refiner output is the velocity). With raw + physics, the raw map is dropped
(zeroed) for half of the training batches, so the network cannot lean on family-specific gather patterns and the
physics channels carry the prediction.
Training: all listed training families pooled (models 0-451 of each file, or a random subset of --ntrain per family),
L1 on velocity scaled to [-1, 1], AdamW 1e-3 cosine, batch 32, left-right flips (dip sign flipped, gathers mirrored),
Apple GPU when available. Test: models 452-499 of every family in --test, OpenFWI MAE / RMSE / SSIM.

usage: uv run python -I toy/slopes/train.py --inputs both --train flatvel-a,flatfault-a --test flatvel-a,flatfault-a [--steps 6000] [--ntrain 452] [--seed 0] [--save DIR]
Appends one JSON line per test family to toy/slopes/train2_results.jsonl.
"""
import argparse, glob, json, os, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwi import metrics
NTR_POOL = 452; NCH = 8

def load(family, feat=os.environ.get('FEAT', 'toy/slopes/feat'), feat_r='toy/slopes/feat_refr'):
    ch = np.zeros((500, NCH, 70, 70), np.float32); raw = np.zeros((500, 5, 100, 35), np.float32); mod = np.zeros((500, 70, 70), np.float32); ok = np.zeros(500, bool)
    for k in range(500):
        f, fr = f'{feat}/{family}/{k:03d}.npz', f'{feat_r}/{family}/{k:03d}.npz'
        if not (os.path.exists(f) and os.path.exists(fr)): continue
        d = np.load(f); ch[k, :6], raw[k], mod[k] = d['ch'], d['raw'], d['model']; ch[k, 6:] = np.load(fr)['ch_refr'][:2]; ok[k] = True
    raw *= (1 + np.arange(100) * 0.01 * 2.0)[None, None, :, None]
    return dict(ch=ch, raw=raw, y=(mod - 3000) / 1500, ok=ok)

def flip(c, r, y):
    c = c[..., ::-1].copy(); c[2] *= -1; return c, r[::-1, :, ::-1].copy(), y[..., ::-1].copy()

class Net(nn.Module):
    def __init__(self, inputs, width=64, npca=128):
        super().__init__(); self.inputs = inputs
        if inputs in ('raw', 'both'): self.mlp = nn.Sequential(nn.Linear(npca, 512), nn.GELU(), nn.Linear(512, 4900))
        cin = {'raw': 1, 'physics': NCH, 'both': NCH + 1}[inputs]; layers = []
        for d in (1, 2, 4, 8, 4, 1): layers += [nn.Conv2d(cin, width, 3, padding=d, dilation=d), nn.GELU()]; cin = width
        self.body = nn.Sequential(*layers); self.head = nn.Conv2d(width, 1, 1)
    def forward(self, ch, z, drop_raw=None):
        parts = [] if self.inputs == 'raw' else [ch]
        if self.inputs != 'physics':
            rmap = self.mlp(z).view(-1, 1, 70, 70)
            if drop_raw is not None: rmap = rmap * drop_raw[:, None, None, None]
            parts.append(rmap)
        out = self.head(self.body(torch.cat(parts, 1)))[:, 0]
        return out if self.inputs == 'raw' else ch[:, 0] + out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', required=True, choices=['physics', 'raw', 'both']); ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--ntrain', type=int, default=452); ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--save', default=None)
    ap.add_argument('--out', default='toy/slopes/train2_results.jsonl'); ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu'); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device(a.device); t0 = time.time()
    fams = sorted(set(a.train.split(',')) | set(a.test.split(','))); D = {f: load(f) for f in fams}
    tr = []                                                                      # (family, index) pairs, pooled
    for f in a.train.split(','):
        pool = [k for k in range(NTR_POOL) if D[f]['ok'][k]]; tr += [(f, k) for k in sorted(rng.choice(pool, size=min(a.ntrain, len(pool)), replace=False).tolist())]
    X = np.stack([D[f]['raw'][k] for f, k in tr] + [D[f]['raw'][k][::-1, :, ::-1] for f, k in tr]).reshape(2 * len(tr), -1)
    mu = X.mean(0); sd = X.std(0) + 0.05 * np.sqrt(X.var(0).mean()); _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    Pm = np.zeros((X.shape[1], 128), np.float32); nc = min(128, Vt.shape[0]); Pm[:, :nc] = Vt[:nc].T
    mu_t, sd_t, P = (torch.tensor(x, dtype=torch.float32) for x in (mu, sd, Pm))
    def pca(raw): return (((torch.tensor(np.ascontiguousarray(raw).reshape(len(raw), -1), dtype=torch.float32) - mu_t) / sd_t) @ P / 30.0).to(dev)
    net = Net(a.inputs).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for step in range(a.steps):
        ids = rng.integers(0, len(tr), 32); cs, rs, ys = [], [], []
        for i in ids:
            f, k = tr[i]; c, r, y = D[f]['ch'][k], D[f]['raw'][k], D[f]['y'][k]
            if rng.random() < 0.5: c, r, y = flip(c, r, y)
            cs.append(c); rs.append(r); ys.append(y)
        drop = torch.tensor((rng.random(32) > 0.5).astype(np.float32), device=dev) if a.inputs == 'both' else None
        pred = net(torch.tensor(np.stack(cs), device=dev), pca(np.stack(rs)), drop)
        loss = (pred - torch.tensor(np.stack(ys), device=dev)).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); rows = []
    for f in a.test.split(','):
        te = [k for k in range(NTR_POOL, 500) if D[f]['ok'][k]]
        with torch.no_grad(): pred = net(torch.tensor(D[f]['ch'][te], device=dev), pca(D[f]['raw'][te])).cpu().numpy()
        ms = [metrics(p * 1500 + 3000, D[f]['y'][k] * 1500 + 3000) for p, k in zip(pred, te)]
        res = {m: round(float(np.mean([x[m] for x in ms])), 4) for m in ('mae', 'rmse', 'ssim')}
        geo = round(float(np.mean([metrics(D[f]['ch'][k, 0] * 1500 + 3000, D[f]['y'][k] * 1500 + 3000)['mae'] for k in te])), 4)
        row = dict(inputs=a.inputs, train=a.train, test=f, ntrain=a.ntrain, steps=a.steps, seed=a.seed, ntest=len(te), **res, geo_mae=geo, seconds=round(time.time() - t0))
        rows.append(row); print(row, flush=True); open(a.out, 'a').write(json.dumps(row) + '\n')
        if a.save:
            os.makedirs(a.save, exist_ok=True); np.savez_compressed(f'{a.save}/{a.inputs}_{a.train.replace(",", "+")}_{f}_n{a.ntrain}_s{a.seed}.npz', pred=pred * 1500 + 3000, idx=np.array(te))

if __name__ == '__main__':
    main()

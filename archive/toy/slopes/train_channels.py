"""Does more physics per pixel make a network better and more data-efficient? Same small network, four inputs.

  geo   no learning: the geometric start v_geo (channel 0)
  raw   gathers (5 x 100 x 35, time gain, standardized) -> PCA 128 (fit on the training set) -> MLP -> 70 x 70 map ->
        conv refiner -> velocity
  phys  the six physics channels (features.py) -> conv refiner -> correction added to v_geo
  both  [MLP map from raw, six channels] -> conv refiner -> correction added to v_geo
  physR / bothR  as phys / both with three more channels (features_refr.py): refractor likelihood, soft two-layer
        velocity from the low-frequency two-layer waveform fit, post-critical mask
  physL / physP  ablations: phys + the two refraction channels only / phys + the post-critical mask only
  refr  no learning: the soft two-layer velocity channel

Conv refiner: 6 dilated 3x3 conv layers (48 channels, dilations 1 2 4 8 4 1, receptive field 41 cells). L1 loss on
velocity scaled to [-1, 1] (1500-4500 m/s), AdamW 1e-3 with cosine decay over a fixed 2000 steps of batch 32 (same
effort for every training-set size), weight decay 1e-4, Apple GPU when available, left-right flip
augmentation (dip channel sign flipped; raw gathers: receivers and shot order reversed). Metrics: OpenFWI MAE / RMSE /
SSIM on [-1, 1] (fwi.metrics). Split: models 0-451 of the file are the training pool, 452-499 the test set.

usage: uv run python -I toy/slopes/train_channels.py --train flatvel-a --test flatvel-a --variant phys --ntrain 452 [--seed 0] [--steps 2000]
Appends one JSON line to toy/slopes/train_results.jsonl.
"""
import argparse, glob, json, os, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwi import metrics
FEAT = 'toy/slopes/feat'; FEAT_R = 'toy/slopes/feat_refr'; NTR_POOL = 452
VARIANTS = {'raw': ([], True), 'phys': (list(range(6)), False), 'both': (list(range(6)), True), 'physR': (list(range(9)), False),
            'bothR': (list(range(9)), True), 'physL': (list(range(8)), False), 'physP': (list(range(6)) + [8], False)}

def load(family, need_refr=False):
    """All channels (6 from features.py, plus 3 from features_refr.py when present). With need_refr, a model counts only
    if its refraction channels exist."""
    fs = sorted(glob.glob(f'{FEAT}/{family}/*.npz')); idx = [int(os.path.basename(f)[:3]) for f in fs]
    ch = np.zeros((500, 9, 70, 70), np.float32); raw = np.zeros((500, 5, 100, 35), np.float32); mod = np.zeros((500, 70, 70), np.float32); ok = np.zeros(500, bool)
    for f, k in zip(fs, idx):
        d = np.load(f); ch[k, :6], raw[k], mod[k] = d['ch'], d['raw'], d['model']; ok[k] = True
        fr = f'{FEAT_R}/{family}/{k:03d}.npz'
        if os.path.exists(fr): ch[k, 6:] = np.load(fr)['ch_refr']
        elif need_refr: ok[k] = False
    gain = (1 + np.arange(100) * 0.01 * 2.0)[None, None, :, None]
    return dict(ch=ch, raw=raw * gain, model=(mod - 3000) / 1500, ok=ok)

def flip(ch, raw, y):
    """One sample: ch (9, 70, 70), raw (5, 100, 35), y (70, 70). Mirror left-right: the dip channel changes sign, the
    gathers reverse receiver order and shot order."""
    ch = ch[..., ::-1].copy(); ch[2] *= -1
    return ch, raw[::-1, :, ::-1].copy(), y[..., ::-1].copy()

class Refiner(nn.Module):
    def __init__(self, cin, width=48):
        super().__init__(); layers = []; c = cin
        for d in (1, 2, 4, 8, 4, 1): layers += [nn.Conv2d(c, width, 3, padding=d, dilation=d), nn.GELU()]; c = width
        self.body = nn.Sequential(*layers); self.head = nn.Conv2d(width, 1, 1)
    def forward(self, x): return self.head(self.body(x))[:, 0]

class Net(nn.Module):
    def __init__(self, variant, npca=128):
        super().__init__(); self.variant = variant; self.chans, self.use_raw = VARIANTS[variant]
        if self.use_raw: self.mlp = nn.Sequential(nn.Linear(npca, 512), nn.GELU(), nn.Linear(512, 4900))
        self.ref = Refiner(len(self.chans) + (1 if self.use_raw else 0))
    def forward(self, ch, z):
        if self.variant == 'raw': return self.ref(self.mlp(z).view(-1, 1, 70, 70))
        x = ch[:, self.chans]
        if self.use_raw: x = torch.cat([x, self.mlp(z).view(-1, 1, 70, 70)], 1)
        return ch[:, 0] + self.ref(x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--test', required=True); ap.add_argument('--variant', required=True, choices=['geo', 'refr'] + list(VARIANTS))
    ap.add_argument('--ntrain', type=int, default=452); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu'); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default='toy/slopes/train_results.jsonl'); ap.add_argument('--save-pred', default=None, help='save test predictions (m/s) and indices to this .npz'); a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time(); dev = torch.device(a.device)
    nr = a.variant in ('physR', 'bothR', 'physL', 'physP', 'refr'); Dtr = load(a.train, nr); Dte = Dtr if a.test == a.train else load(a.test, nr)
    pool = [k for k in range(NTR_POOL) if Dtr['ok'][k]]; tr = sorted(rng.choice(pool, size=min(a.ntrain, len(pool)), replace=False).tolist()); te = [k for k in range(NTR_POOL, 500) if Dte['ok'][k]]
    def evaluate(pred_scaled, idx):
        ms = [metrics(p * 1500 + 3000, Dte['model'][k] * 1500 + 3000) for p, k in zip(pred_scaled, idx)]
        return {m: round(float(np.mean([x[m] for x in ms])), 4) for m in ('mae', 'rmse', 'ssim')}
    if a.variant in ('geo', 'refr'):
        res = evaluate(Dte['ch'][te, 0 if a.variant == 'geo' else 7], te)
    else:
        X = Dtr['raw'][tr].reshape(len(tr), -1); mu = X.mean(0); sd = X.std(0) + 0.05 * np.sqrt(X.var(0).mean())
        Xa = np.concatenate([X, np.stack([flip(Dtr['ch'][k], Dtr['raw'][k], Dtr['model'][k])[1] for k in tr]).reshape(len(tr), -1)])
        _, _, Vt = np.linalg.svd((Xa - mu) / sd, full_matrices=False); Pm = np.zeros((Xa.shape[1], 128), np.float32); nc = min(128, Vt.shape[0]); Pm[:, :nc] = Vt[:nc].T; P = torch.tensor(Pm)   # fewer samples than 128: zero-padded basis
        mu_t, sd_t = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
        def pca(raw): return ((torch.tensor(np.ascontiguousarray(raw).reshape(len(raw), -1), dtype=torch.float32) - mu_t) / sd_t) @ P / 30.0
        net = Net(a.variant).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        for step in range(a.steps):                                              # fixed step budget: same effort for every training-set size
            ids = rng.choice(tr, size=32, replace=len(tr) < 32); chs, raws, ys = [], [], []
            for k in ids:
                c, r, y = Dtr['ch'][k], Dtr['raw'][k], Dtr['model'][k]
                if rng.random() < 0.5: c, r, y = flip(c, r, y)
                chs.append(c); raws.append(r); ys.append(y)
            pred = net(torch.tensor(np.stack(chs), dtype=torch.float32, device=dev), pca(np.stack(raws)).to(dev))
            loss = (pred - torch.tensor(np.stack(ys), dtype=torch.float32, device=dev)).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): pred = net(torch.tensor(Dte['ch'][te], dtype=torch.float32, device=dev), pca(Dte['raw'][te]).to(dev)).cpu().numpy()
        res = evaluate(pred, te)
        if a.save_pred: np.savez_compressed(a.save_pred, pred=pred * 1500 + 3000, idx=np.array(te))
    row = dict(train=a.train, test=a.test, variant=a.variant, ntrain=a.ntrain if a.variant not in ('geo', 'refr') else 0, ntest=len(te), seed=a.seed, steps=a.steps, **res, seconds=round(time.time() - t0))
    print(row, flush=True); open(a.out, 'a').write(json.dumps(row) + '\n')

if __name__ == '__main__':
    main()

"""Step 3, training: one network for all ten families.

Models (same training recipe for all):
  inversionnet  the OpenFWI baseline (Wu & Lin 2019, as in the OpenFWI repository): gathers (5 x 1000 x 70) -> velocity
  ours          the same gather encoder, plus an image branch: for each shot, its three background images (slow /
                medium / fast) go through one encoder shared by all shots; the five shot features are fused by their
                mean and standard deviation (agreement between shots); a U-Net at 70 x 70 combines them with the
                medium background velocity and with the gather encoder's code, and predicts velocity
  ours_k1       ablation: only the medium-background images
  ours_noraw    ablation: no gather encoder (images and background only)
Training: models 0-451 of every family pooled (4520 models), left-right mirroring of models, gathers and images
together, image branch switched off for 30 % of samples (modality dropout), loss = L1 + 0.5 (1 - multi-scale SSIM) on
velocity scaled to [-1, 1] (--loss l1 for the original InversionNet recipe), AdamW with warm-up and cosine decay,
weights averaged (EMA) for testing.

usage: uv run python -I openfwi/3_train.py --model ours [--seed 0] [--steps 8000] [--batch 64] [--loss l1ssim]
Writes work/models/<model>_<loss>_s<seed>.pt
"""
import argparse, json, math, os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'physics'))
import background as BG
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- data
def load_data(split, families='all', images=True):
    """split 'train' (models 0-451) or 'test' (452-499) of every family -> dict of numpy arrays, gathers log-scaled."""
    man = json.load(open(os.path.join(HERE, 'data', 'manifest.json'))); ntr, n = man['n_train'], man['n_total']
    if families != 'all': man['families'] = {f: v for f, v in man['families'].items() if f in families.split(',')}
    idx = np.arange(0, ntr) if split == 'train' else np.arange(ntr, n); trend = np.load(os.path.join(HERE, 'work', 'depth_trend.npy'))
    out = dict(gathers=[], images=[], vref=[], y=[], family=[], index=[])
    for f, v in man['families'].items():
        g = np.asarray(np.load(os.path.join(HERE, v['gathers']), mmap_mode='r')[idx], np.float32)
        out['gathers'].append((np.sign(g) * np.log1p(np.abs(g))).astype(np.float16))
        if images: inp = np.load(os.path.join(HERE, 'work', 'inputs', f'{f}.npz')); out['images'].append(inp['images'][idx]); v1 = inp['surface_velocity'][idx]
        else: out['images'].append(np.zeros((len(idx), 1, 1, 1, 1), np.float16)); v1 = np.full(len(idx), 3000.0, np.float32)   # InversionNet uses gathers only
        vref = v1[:, None, None] + trend[None, :, None] + np.zeros((1, 1, 70))      # medium background, laterally constant
        out['vref'].append((vref - 3000) / 1500)
        out['y'].append(((np.asarray(np.load(os.path.join(HERE, v['models']), mmap_mode='r')[idx, 0], np.float32)) - 3000) / 1500)
        out['family'] += [f] * len(idx); out['index'] += list(idx)
    d = {k: (np.concatenate(v) if k not in ('family', 'index') else np.array(v)) for k, v in out.items()}
    d['vref'] = np.clip(d['vref'], -1, 1).astype(np.float32)
    return d

def batch(d, ids, scale, mirror=None, drop=None):
    g = torch.tensor(d['gathers'][ids].astype(np.float32) / scale); im = torch.tensor(d['images'][ids].astype(np.float32))
    vr = torch.tensor(d['vref'][ids]); y = torch.tensor(d['y'][ids])
    if mirror is not None and mirror.any():                       # mirror the subsurface: reverse receivers, shot order, image x
        m = torch.tensor(mirror)
        g[m] = g[m].flip(1).flip(3); im[m] = im[m].flip(2).flip(4); y[m] = y[m].flip(2)
    if drop is not None: im = im * torch.tensor(1.0 - drop, dtype=torch.float32)[:, None, None, None, None]
    return g.to(DEV), im.to(DEV), vr.to(DEV), y.to(DEV)

# ---------------------------------------------------------------- networks
def cb(i, o, k=(3, 3), s=(1, 1), p=(1, 1)): return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
def db(i, o, k, s=1, p=0): return nn.Sequential(nn.ConvTranspose2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))

class InvEncoder(nn.Module):
    """InversionNet encoder: (B, 5, 1000, 70) -> (B, 512, 1, 1)."""
    def __init__(self, d=(32, 64, 128, 256, 512)):
        super().__init__(); d1, d2, d3, d4, d5 = d
        self.net = nn.Sequential(cb(5, d1, (7, 1), (2, 1), (3, 0)), cb(d1, d2, (3, 1), (2, 1), (1, 0)), cb(d2, d2, (3, 1), (1, 1), (1, 0)),
                                 cb(d2, d2, (3, 1), (2, 1), (1, 0)), cb(d2, d2, (3, 1), (1, 1), (1, 0)), cb(d2, d3, (3, 1), (2, 1), (1, 0)),
                                 cb(d3, d3, (3, 1), (1, 1), (1, 0)), cb(d3, d3, s=(2, 2)), cb(d3, d3), cb(d3, d4, s=(2, 2)), cb(d4, d4),
                                 cb(d4, d4, s=(2, 2)), cb(d4, d4), cb(d4, d5, (8, 9), (1, 1), (0, 0)))
    def forward(self, x): return self.net(x)

class InvDecoder(nn.Module):
    """InversionNet decoder: (B, 512, 1, 1) -> (B, 70, 70) in [-1, 1]."""
    def __init__(self, d=(32, 64, 128, 256, 512)):
        super().__init__(); d1, d2, d3, d4, d5 = d
        self.net = nn.Sequential(db(d5, d5, 5), cb(d5, d5), db(d5, d4, 4, 2, 1), cb(d4, d4), db(d4, d3, 4, 2, 1), cb(d3, d3),
                                 db(d3, d2, 4, 2, 1), cb(d2, d2), db(d2, d1, 4, 2, 1), cb(d1, d1), nn.Conv2d(d1, 1, 3, 1, 1), nn.Tanh())
    def forward(self, z): return self.net(z)[:, 0, 5:-5, 5:-5]

class InversionNet(nn.Module):
    def __init__(self): super().__init__(); self.enc = InvEncoder(); self.dec = InvDecoder()
    def forward(self, g, im=None, vr=None): return self.dec(self.enc(g))

class Ours(nn.Module):
    def __init__(self, k=3, use_raw=True, w=(64, 128, 256, 512)):
        super().__init__(); self.k = k; self.use_raw = use_raw
        self.enc = InvEncoder() if use_raw else None
        self.shot = nn.Sequential(cb(k, 32), cb(32, 32))                                   # shared by all five shots
        self.d0 = nn.Sequential(cb(65, w[0]), cb(w[0], w[0]))                              # 70 x 70: mean + std over shots + background
        self.d1 = nn.Sequential(cb(w[0], w[1], s=(2, 2)), cb(w[1], w[1]))                  # 35
        self.d2 = nn.Sequential(cb(w[1], w[2], s=(2, 2)), cb(w[2], w[2]))                  # 18
        self.d3 = nn.Sequential(cb(w[2], w[3], s=(2, 2)), cb(w[3], w[3]))                  # 9
        self.mid = cb(w[3] + (512 if use_raw else 0), w[3])
        self.u2 = nn.Sequential(cb(w[3] + w[2], w[2]), cb(w[2], w[2])); self.u1 = nn.Sequential(cb(w[2] + w[1], w[1]), cb(w[1], w[1]))
        self.u0 = nn.Sequential(cb(w[1] + w[0], w[0]), cb(w[0], w[0])); self.head = nn.Sequential(nn.Conv2d(w[0], 1, 1), nn.Tanh())
    def forward(self, g, im, vr):
        B = im.shape[0]; im = im if self.k == 3 else im[:, 1:2]                            # im: (B, K, 5, 70, 70)
        f = self.shot(im.permute(0, 2, 1, 3, 4).reshape(B * 5, self.k, 70, 70)).reshape(B, 5, 32, 70, 70)
        x0 = self.d0(torch.cat([f.mean(1), f.std(1), vr[:, None]], 1)); x1 = self.d1(x0); x2 = self.d2(x1); x3 = self.d3(x2)
        if self.use_raw: x3 = torch.cat([x3, self.enc(g).expand(-1, -1, x3.shape[2], x3.shape[3])], 1)
        x = self.mid(x3)
        x = self.u2(torch.cat([F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False), x2], 1))
        x = self.u1(torch.cat([F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False), x1], 1))
        x = self.u0(torch.cat([F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False), x0], 1))
        return self.head(x)[:, 0]

def build(name):
    return {'inversionnet': InversionNet, 'ours': lambda: Ours(3, True), 'ours_k1': lambda: Ours(1, True), 'ours_noraw': lambda: Ours(3, False)}[name]()

# ---------------------------------------------------------------- loss
def _gauss(win=7, sigma=1.5):
    x = torch.arange(win, dtype=torch.float32) - win // 2; g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum(); return (g[:, None] * g[None, :])[None, None]
def ssim(a, b, L=2.0):
    w = _gauss().to(a.device); a = a[:, None]; b = b[:, None]; C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a = F.conv2d(a, w); mu_b = F.conv2d(b, w); va = F.conv2d(a * a, w) - mu_a ** 2; vb = F.conv2d(b * b, w) - mu_b ** 2; cov = F.conv2d(a * b, w) - mu_a * mu_b
    return (((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))).mean()
def ms_ssim(a, b, scales=3):
    vals = []
    for s in range(scales):
        vals.append(ssim(a, b))
        if s < scales - 1: a = F.avg_pool2d(a[:, None], 2)[:, 0]; b = F.avg_pool2d(b[:, None], 2)[:, 0]
    return torch.stack(vals).mean()
def loss_fn(p, y, kind): return (p - y).abs().mean() + (0.5 * (1 - ms_ssim(p, y)) if kind == 'l1ssim' else 0.0)

# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=['inversionnet', 'ours', 'ours_k1', 'ours_noraw'])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--loss', default='l1ssim', choices=['l1ssim', 'l1']); ap.add_argument('--drop', type=float, default=0.3); ap.add_argument('--families', default='all'); ap.add_argument('--tag', default=''); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()
    d = load_data('train', a.families, images=a.model != 'inversionnet'); scale = float(np.abs(d['gathers']).max()); N = len(d['y'])
    print(f'{a.model}: {N} training models from {len(set(d["family"]))} families, data loaded in {time.time() - t0:.0f}s', flush=True)
    net = build(a.model).to(DEV); ema = build(a.model).to(DEV); ema.load_state_dict(net.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    nparam = sum(p.numel() for p in net.parameters()); print(f'{nparam / 1e6:.1f}M parameters', flush=True)
    for step in range(a.steps):
        ids = rng.integers(0, N, a.batch); mirror = rng.random(a.batch) < 0.5
        drop = (rng.random(a.batch) < a.drop).astype(np.float32) if a.model.startswith('ours') and a.model != 'ours_noraw' else None
        g, im, vr, y = batch(d, ids, scale, mirror, drop)
        net.train(); loss = loss_fn(net(g, im, vr), y, a.loss); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.999).add_(pn, alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0 or step == a.steps - 1: print(f'step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(os.path.join(HERE, 'work', 'models'), exist_ok=True); path = os.path.join(HERE, 'work', 'models', f'{a.model}_{a.loss}_s{a.seed}{a.tag}.pt')
    torch.save(dict(families=a.families, model=a.model, loss=a.loss, seed=a.seed, steps=a.steps, scale=scale, params=nparam, state=ema.state_dict(), seconds=round(time.time() - t0)), path)
    print('saved', path, flush=True)

if __name__ == '__main__':
    main()

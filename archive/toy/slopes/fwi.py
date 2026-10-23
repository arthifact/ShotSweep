"""Geometric initializer + multiscale FWI finisher on official OpenFWI gathers, with the benchmark's own metrics.

Forward: 2D acoustic FD (4th order space, 2nd order time), 10 m cells, 1 ms, a 60-cell damped-wave absorbing layer on
all four sides (OpenFWI has absorbing boundaries on all sides, no free surface), sources and receivers one cell deep,
source x from the earliest first break of each official shot. On the true FlatVel-A model 452 this reproduces the
official gathers to a 2 % relative residual. Gradient: exact discrete adjoint. The source wavelet is estimated per model and per frequency band by a
Wiener filter between the simulated and observed gathers (so a Ricker of unknown delay, scale and shape is not needed).
FWI: three bands (low-pass 4, 8 Hz, then full band), L2 misfit, autograd gradient, illumination-normalised, top rows
muted, steepest descent with backtracking line search, velocity clipped to [1400, 4600].
Starts: 'geo' = flat-layer pipeline v(z) (slopes.py) on the same gathers; 'grad' = v1 + 2 z (a generic gradient);
'mean' = the dataset-mean model as a static prediction (no FWI).
Metrics: OpenFWI style on velocity min-max normalised to [-1, 1] over 1500-4500 m/s: MAE, RMSE, SSIM (7x7 uniform
window, K1 = 0.01, K2 = 0.03, data range 2).

usage: uv run python -I toy/slopes/fwi.py --family flatvel-a --data DATA.npy --models 452-499 --start geo [--iters 12] [--out dir]
"""
import argparse, glob, json, os, sys, time, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slopes as S
from slopes import DX, DT, NX, NZ, DATUM, direct_wave_fit, oriented_points, vote, pick_interfaces, dix, layers_to_vz, avo_velocity
ROOT = '/Users/torus/Desktop/Research/renderwave/data/openfwi'
NT = 1000; F0 = 15.0; PAD = 20; VMIN, VMAX = 1500.0, 4500.0

# ----------------------------------------------------------------------------------------------------- forward
class Forward:
    """Shots batched, float32. Absorbing layer = damped wave equation u_tt + sigma u_t = c^2 lap u in a PAD-cell zone,
    sigma quadratic, sigma_max = 3 vref / (2 L) ln(1/R) (2 % residual against the official gathers on the true model).
    gradient() is the exact discrete adjoint of this recurrence (no autograd: only the model-region Laplacian of the
    forward field is stored, 100 MB)."""
    def __init__(self, src_cells, nthreads=8, pad=60, R=1e-5, vref=4500.0, margin=20):
        torch.set_num_threads(nthreads); self.src = torch.tensor(list(src_cells)); self.pad = pad; self.margin = margin; nz = nx = NZ + 2 * pad
        iz = np.arange(nz)[:, None]; ix = np.arange(nx)[None, :]
        dz = np.clip(np.maximum(pad - iz, iz - (nz - 1 - pad)), 0, None) / pad; dxx = np.clip(np.maximum(pad - ix, ix - (nx - 1 - pad)), 0, None) / pad
        sig = 3 * vref / (2 * pad * DX) * np.log(1 / R) * (dz ** 2 + dxx ** 2); a = sig * DT / 2
        self.A = torch.tensor(1 / (1 + a), dtype=torch.float32); self.B = torch.tensor(1 - a, dtype=torch.float32)
        t = np.arange(NT) * DT; aa = (np.pi * F0 * (t - 1.2 / F0)) ** 2; self.ricker = (1 - 2 * aa) * np.exp(-aa)
    @staticmethod
    def lap(p):
        out = torch.zeros_like(p)
        out[..., 2:-2, 2:-2] = (-p[..., :-4, 2:-2] + 16 * p[..., 1:-3, 2:-2] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 3:-1, 2:-2] - p[..., 4:, 2:-2]
                                - p[..., 2:-2, :-4] + 16 * p[..., 2:-2, 1:-3] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 2:-2, 3:-1] - p[..., 2:-2, 4:]) / 12
        return out
    def c2(self, v):
        return torch.tensor(np.pad(np.asarray(v, np.float64), self.pad, mode='edge') ** 2 * (DT / DX) ** 2, dtype=torch.float32)
    def run(self, v, wavelet, store=False):
        P = self.pad; c2 = self.c2(v); ns = len(self.src); nz = NZ + 2 * P
        w = torch.tensor(wavelet * 1e3, dtype=torch.float32); si = torch.arange(ns); sx = self.src + P
        Mg = self.margin; lo, hiz, hix = P - Mg, P + NZ + Mg, P + NX + Mg
        p0 = torch.zeros(ns, nz, nz); p1 = torch.zeros_like(p0); out = torch.zeros(ns, NT, NX); Ls = torch.zeros(NT, ns, NZ + 2 * Mg, NX + 2 * Mg) if store else None
        for it in range(NT):
            L = self.lap(p1)
            if store: Ls[it] = L[:, lo:hiz, lo:hix]
            p2 = (2 * p1 - self.B * p0 + c2 * L) * self.A; p2[si, P + 1, sx] += w[it]
            out[:, it] = p2[:, P + 1, P:P + NX]; p0, p1 = p1, p2
        return out, Ls, c2
    def __call__(self, v, wavelet):
        with torch.no_grad(): return self.run(v, wavelet)[0].numpy()
    def gradient(self, v, wavelet, obs):
        """J = 0.5 ||sim - obs||^2 and dJ/dv on the model grid (exact discrete adjoint)."""
        P = self.pad; ns = len(self.src); nz = NZ + 2 * P
        with torch.no_grad():
            sim, Ls, c2 = self.run(v, wavelet, store=True); res = sim - torch.tensor(obs, dtype=torch.float32); J = 0.5 * float((res ** 2).sum())
            Mg = self.margin; lo, hiz, hix = P - Mg, P + NZ + Mg, P + NX + Mg
            m1 = torch.zeros(ns, nz, nz); m2 = torch.zeros_like(m1); g = torch.zeros(NZ + 2 * Mg, NX + 2 * Mg); A, B = self.A, self.B
            for it in range(NT - 1, -1, -1):
                Am1 = A * m1
                mu = 2 * Am1 + self.lap(c2 * Am1) - A * B * m2; mu[:, P + 1, P:P + NX] += res[:, it]
                g += ((A * mu)[:, lo:hiz, lo:hix] * Ls[it]).sum(0)
                m2, m1 = m1, mu
            # dJ/dc2 on the model + margin region -> dJ/dv: c2 = v_pad^2 (dt/dx)^2 and v_pad replicates the model edges, so
            # each margin cell's gradient folds back onto the nearest edge cell (adjoint of edge replication)
            gm = g.numpy().astype(np.float64) * 2 * np.pad(np.asarray(v, np.float64), Mg, mode='edge') * (DT / DX) ** 2
            iz = np.clip(np.arange(NZ + 2 * Mg) - Mg, 0, NZ - 1); ix = np.clip(np.arange(NX + 2 * Mg) - Mg, 0, NX - 1)
            gv = np.zeros((NZ, NX)); np.add.at(gv, (iz[:, None].repeat(NX + 2 * Mg, 1), ix[None, :].repeat(NZ + 2 * Mg, 0)), gm)
        return J, gv

def lowpass(x, fc, axis=1):
    """Gaussian low-pass in frequency along the time axis (fc = None: unchanged)."""
    if fc is None: return x
    n = x.shape[axis]; f = np.fft.rfftfreq(n, DT); H = np.exp(-0.5 * (f / fc) ** 4)
    sh = [1] * x.ndim; sh[axis] = len(f)
    return np.fft.irfft(np.fft.rfft(x, axis=axis) * H.reshape(sh), n=n, axis=axis)

def estimate_source(sim, obs, ricker, eps=1e-3):
    """Wiener filter F = sum conj(S) O / (sum |S|^2 + eps) over all traces; returns the filtered Ricker (effective source)."""
    Sf = np.fft.rfft(sim, axis=1); Of = np.fft.rfft(obs, axis=1)
    num = (np.conj(Sf) * Of).sum(axis=(0, 2)); den = (np.abs(Sf) ** 2).sum(axis=(0, 2))
    F = num / (den + eps * den.max())
    return np.fft.irfft(np.fft.rfft(ricker) * F, n=len(ricker))

def source_x_from_data(obs):
    """Source cell of each shot = receiver with the earliest first break (parabolic refinement)."""
    cells = []
    for g in obs:
        env = np.abs(g); fb = np.array([np.argmax(env[:, r] > 0.3 * env[:, r].max()) for r in range(NX)]).astype(float)
        r0 = int(np.argmin(fb)); cells.append(r0)
    return cells

# ----------------------------------------------------------------------------------------------------- starts
def geometric_init(obs):
    G = obs.astype(np.float64); v1, tau = direct_wave_fit(G)
    t0_bins = np.arange(0, 1.0, 0.004); v_bins = np.arange(1300, 5000, 25.0); t0c = 0.5 * (t0_bins[1:] + t0_bins[:-1]); vc = 0.5 * (v_bins[1:] + v_bins[:-1])
    pts = oriented_points(G, tau, v1); V = vote(pts, t0_bins, v_bins); _, _, picks = pick_interfaces(V, t0c, vc, v1)
    kept = []                                    # primaries only: a multiple travels in the slow overburden, so its v_rms drops;
    for t0, vr in picks:                          # and nothing below the model bottom
        if kept and vr < 0.97 * kept[-1][1]: continue
        if dix(kept + [(t0, vr)], v1)[-2][1] > NZ * DX: continue
        kept.append((t0, vr))
    layers = dix(kept, v1); crit = avo_velocity(G, tau, layers)
    if crit is not None: layers[-1] = (layers[-1][0], layers[-1][1], crit[0])
    vz = layers_to_vz(layers); return np.tile(vz[:, None], (1, NX)), v1, layers

def gradient_init(v1):
    return np.tile((v1 + 2.0 * (np.arange(NZ) * DX))[:, None], (1, NX))

# ----------------------------------------------------------------------------------------------------- metrics
def norm(v): return (np.clip(v, VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 2 - 1
def ssim(a, b, win=7, K1=0.01, K2=0.03, L=2.0):
    def box(x):
        c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1); w = win
        return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)
    mu_a, mu_b = box(a), box(b); va = box(a * a) - mu_a ** 2; vb = box(b * b) - mu_b ** 2; cov = box(a * b) - mu_a * mu_b
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    return float((((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))).mean())
def metrics(pred, true):
    a, b = norm(pred), norm(true); return dict(mae=round(float(np.abs(a - b).mean()), 4), rmse=round(float(np.sqrt(((a - b) ** 2).mean())), 4), ssim=round(ssim(a, b), 4))

# ----------------------------------------------------------------------------------------------------- FWI
def lbfgs_direction(g, S_hist, Y_hist):
    q = g.copy(); al = []
    for s_, y_ in reversed(list(zip(S_hist, Y_hist))):
        rho = 1.0 / (y_ * s_).sum(); a_ = rho * (s_ * q).sum(); q -= a_ * y_; al.append((rho, a_, s_, y_))
    if S_hist: q *= (S_hist[-1] * Y_hist[-1]).sum() / (Y_hist[-1] * Y_hist[-1]).sum()
    for rho, a_, s_, y_ in reversed(al): b_ = rho * (y_ * q).sum(); q += s_ * (a_ - b_)
    return -q

def fwi(obs, v0, bands=(8.0, None), iters=(10, 25), smooth=2.0, nthreads=8, log=None, true=None, mem=6):
    """L-BFGS on velocity per band; the gradient is preconditioned by a Gaussian smoothing (smooth cells) and a mute of
    the two top rows; first step capped at 50 m/s, then Armijo backtracking on the L-BFGS step."""
    fwd = Forward(source_x_from_data(obs), nthreads, margin=0); v = np.asarray(v0, np.float64).copy()
    mute = np.clip((np.arange(NZ) - 1) / 3.0, 0, 1)[:, None]
    def precond(g): return S.gauss1d(S.gauss1d(g * mute, smooth, 0), smooth, 1) * mute
    for fc, nit in zip(bands, iters):
        obs_b = lowpass(obs.astype(np.float64), fc); ric = lowpass(fwd.ricker[None, :, None], fc)[0, :, 0]
        wav = estimate_source(fwd(v, ric), obs_b, ric); Sh, Yh = [], []
        J, g = fwd.gradient(v, wav, obs_b); g = precond(g)
        for it in range(nit):
            d = lbfgs_direction(g, Sh, Yh)
            if not Sh: d = d / (np.abs(d).max() + 1e-30) * 50.0
            if (d * g).sum() >= 0: d = -g / (np.abs(g).max() + 1e-30) * 50.0; Sh, Yh = [], []
            step = 1.0; ok = False
            for _ in range(6):
                vt = np.clip(v + step * d, 1400.0, 4600.0); Jt = 0.5 * float(((fwd(vt, wav) - obs_b) ** 2).sum())
                if Jt < J + 1e-4 * step * (g * d).sum(): ok = True; break
                step *= 0.5
            if log is not None: log.append(dict(band=fc, iter=it, misfit=round(J, 2), step=step if ok else 0.0, **({'mae': metrics(v, true)['mae']} if true is not None else {})))
            if not ok: break
            Jn, gn = fwd.gradient(vt, wav, obs_b); gn = precond(gn)
            sk, yk = vt - v, gn - g
            if (sk * yk).sum() > 1e-12: Sh.append(sk); Yh.append(yk); Sh, Yh = Sh[-mem:], Yh[-mem:]
            v, J, g = vt, Jn, gn
        wav = estimate_source(fwd(v, ric), obs_b, ric)
    return v

# ----------------------------------------------------------------------------------------------------- layered FWI
ZC = np.arange(NZ) * DX                                                      # OpenFWI grid nodes
def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
def render_layers(z, v, w):
    """Interfaces z (K,), layer velocities v (K+1,) -> v(z) on the grid with smooth steps of width w (w=0: hard)."""
    out = np.full(NZ, v[0], dtype=np.float64)
    for k in range(len(z)):
        st = (ZC >= z[k]).astype(float) if w == 0 else sig((ZC - z[k]) / w); out += (v[k + 1] - v[k]) * st
    return out

def candidate_layers(layers, split_min=120.0):
    """Geometric layers -> (z, v); every layer thicker than split_min gets one extra interface at its middle with equal
    velocities on both sides, so FWI can open an interface the slopes merged."""
    z, v = [], [layers[0][2]]
    for i, (zt, zb, vi) in enumerate(layers):
        zt = 0.0 if i == 0 else zt
        if zb - zt > split_min and zb < NZ * DX - 30: z.append(0.5 * (zt + zb)); v.append(vi)
        if i + 1 < len(layers) and zb < NZ * DX - 15: z.append(zb); v.append(layers[i + 1][2])
    return np.array(z), np.array(v, dtype=np.float64)

def isotonic(y):
    """Pool-adjacent-violators: closest non-decreasing sequence (L2)."""
    blocks = [[float(a), 1] for a in y]; i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]; i = max(i - 1, 0)
        else: i += 1
    return np.concatenate([[b[0]] * b[1] for b in blocks])

XC = np.arange(NX) * DX
class Layers:
    """Scene: K interfaces z_k(x) (piecewise linear between NKX lateral knots) and K+1 layer velocities.
    Unknowns scaled: depth / 10 m, velocity / 100 m/s."""
    def __init__(self, z, v, nkx=8, w=8.0):
        self.K = len(z); self.nkx = nkx; self.w = w; self.kx = np.linspace(0, (NX - 1) * DX, nkx)
        self.Bx = np.stack([np.interp(XC, self.kx, np.eye(nkx)[j]) for j in range(nkx)], 0)       # (nkx, NX) hat functions
        self.x = np.concatenate([np.repeat(np.asarray(z, float)[:, None], nkx, 1).ravel() / 10.0, np.asarray(v, float) / 100.0])
    def unpack(self, x): return x[:self.K * self.nkx].reshape(self.K, self.nkx) * 10.0, x[self.K * self.nkx:] * 100.0
    def depth(self, zk): return zk @ self.Bx                                                       # (K, NX)
    def render(self, x, w=None):
        w = self.w if w is None else w; zk, vv = self.unpack(x); zx = self.depth(zk); out = np.full((NZ, NX), vv[0])
        for k in range(self.K):
            st = (ZC[:, None] >= zx[k][None, :]).astype(float) if w == 0 else sig((ZC[:, None] - zx[k][None, :]) / w); out = out + (vv[k + 1] - vv[k]) * st
        return np.clip(out, 1400, 4600)
    def grad(self, x, gpix):
        zk, vv = self.unpack(x); zx = self.depth(zk); w = self.w; K = self.K
        st = [sig((ZC[:, None] - zx[k][None, :]) / w) for k in range(K)]; gz = np.zeros((K, self.nkx)); gv = np.zeros(K + 1)
        for k in range(K):
            ds = st[k] * (1 - st[k]) / w; colg = (gpix * (vv[k + 1] - vv[k]) * (-ds)).sum(0)            # dJ/dz_k(x), (NX,)
            gz[k] = (self.Bx * colg[None, :]).sum(1) * 10.0
        for k in range(K + 1):
            dv = (st[k - 1] if k > 0 else 1.0) - (st[k] if k < K else 0.0); gv[k] = (gpix * dv).sum() * 100.0
        return np.concatenate([gz.ravel(), gv])
    def project(self, x, monotone):
        zk, vv = self.unpack(x); zk = np.clip(zk, 15.0, NZ * DX - 5.0)
        zk = np.maximum.accumulate(zk, axis=0)                                                   # interfaces do not cross
        if monotone: vv = isotonic(vv)
        return np.concatenate([zk.ravel() / 10.0, vv / 100.0])
    def merge(self, x, min_contrast=0.015, min_gap=20.0):
        """Prune: drop interfaces with < min_contrast velocity change or closer than min_gap to the one above (all knots)."""
        zk, vv = self.unpack(x); keep_z, keep_v = [], [vv[0]]
        for k in range(self.K):
            weak = abs(vv[k + 1] - keep_v[-1]) / keep_v[-1] < min_contrast; close = keep_z and np.all(np.abs(zk[k] - keep_z[-1]) < min_gap)
            if weak or close:
                keep_v[-1] = 0.5 * (keep_v[-1] + vv[k + 1]) if not weak else keep_v[-1]; continue
            keep_z.append(zk[k]); keep_v.append(vv[k + 1])
        self.K = len(keep_z)
        return np.concatenate([(np.array(keep_z).reshape(self.K, self.nkx) if self.K else np.zeros((0, self.nkx))).ravel() / 10.0, np.array(keep_v) / 100.0])

def fwi_layers(obs, layers, bands=(8.0, None), iters=(15, 40), nkx=8, monotone=False, smooth_x=0.05, nthreads=8, log=None, true=None, mem=8):
    """L-BFGS on (interface depth knots, layer velocities) with the full-waveform misfit; parameter gradient = chain rule
    of the exact adjoint pixel gradient through the smooth-step rendering. Projection keeps interfaces ordered (and
    velocities non-decreasing when monotone=True); after each band weak or coincident interfaces are merged.
    smooth_x: penalty on second differences of each interface across knots, relative to the band's initial misfit."""
    fwd = Forward(source_x_from_data(obs), nthreads, margin=0); z, v = candidate_layers(layers); sc = Layers(z, v, nkx)
    x = sc.project(sc.x, monotone)
    for fc, nit in zip(bands, iters):
        obs_b = lowpass(obs.astype(np.float64), fc); ric = lowpass(fwd.ricker[None, :, None], fc)[0, :, 0]
        wav = estimate_source(fwd(sc.render(x), ric), obs_b, ric)
        J0 = 0.5 * float(((fwd(sc.render(x), wav) - obs_b) ** 2).sum()); lam = smooth_x * J0
        def objective(xx, need_grad=True):
            Jd, gp = fwd.gradient(sc.render(xx), wav, obs_b) if need_grad else (0.5 * float(((fwd(sc.render(xx), wav) - obs_b) ** 2).sum()), None)
            zk, _ = sc.unpack(xx); d2 = np.diff(zk / 10.0, 2, axis=1) if sc.nkx > 2 else np.zeros((sc.K, 0)); Jr = lam * float((d2 ** 2).sum())
            if not need_grad: return Jd + Jr, None
            g = sc.grad(xx, gp); gr = np.zeros((sc.K, sc.nkx))
            if sc.nkx > 2: gr[:, :-2] += 2 * lam * d2; gr[:, 1:-1] -= 4 * lam * d2; gr[:, 2:] += 2 * lam * d2
            g[:sc.K * sc.nkx] += gr.ravel(); return Jd + Jr, g
        J, g = objective(x); Sh, Yh = [], []; fails = 0
        for it in range(nit):
            d = lbfgs_direction(g, Sh, Yh)
            if not Sh or (d * g).sum() >= 0: d = -g / (np.abs(g).max() + 1e-30); Sh, Yh = [], []
            step = 1.0; ok = False
            for _ in range(8):
                xt = sc.project(x + step * d, monotone); Jt, _ = objective(xt, need_grad=False)
                if Jt < J + 1e-4 * (g * (xt - x)).sum(): ok = True; break
                step *= 0.5
            if log is not None:
                zk, vv = sc.unpack(x); log.append(dict(band=fc, iter=it, misfit=round(J, 2), step=step if ok else 0.0, K=sc.K, v=np.round(vv).tolist(),
                                                    **({'mae': metrics(sc.render(x, 0), true)['mae']} if true is not None else {})))
            if not ok:
                if Sh: Sh, Yh = [], []; continue                                                   # retry once with steepest descent
                break
            Jn, gn = objective(xt); sk, yk = xt - x, gn - g
            if (sk * yk).sum() > 1e-12: Sh.append(sk); Yh.append(yk); Sh, Yh = Sh[-mem:], Yh[-mem:]
            x, J, g = xt, Jn, gn
        x = sc.merge(x)
    zk, vv = sc.unpack(x); return sc.render(x, 0), zk, vv

# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); ap.add_argument('--data', required=True); ap.add_argument('--models', default='452-499')
    ap.add_argument('--start', default='geo', choices=['geo', 'grad', 'mean']); ap.add_argument('--iters', type=int, default=12); ap.add_argument('--out', default='toy/slopes/fwi_runs')
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--figure', action='store_true'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mf = sorted(glob.glob(f'{ROOT}/{a.family}/train/*.npy'))[0]; M = np.load(mf, mmap_mode='r'); D = np.load(a.data, mmap_mode='r')
    lo, hi = (int(x) for x in a.models.split('-')) if '-' in a.models else (int(a.models), int(a.models)); mean_model = np.asarray(M[:452, 0], np.float64).mean(0)
    for k in range(lo, hi + 1):
        outp = f'{a.out}/{a.family}_{k}_{a.start}.json'
        if os.path.exists(outp): continue
        t0 = time.time(); obs = np.array(D[k], np.float64); true = np.array(M[k, 0], np.float64)
        if a.start == 'mean':
            res = dict(family=a.family, model=k, start=a.start, final=metrics(mean_model, true)); json.dump(res, open(outp, 'w')); print(res, flush=True); continue
        v_geo, v1, layers = geometric_init(obs); v0 = v_geo if a.start == 'geo' else gradient_init(v1)
        log = []; v = fwi(obs, v0, nthreads=a.threads, log=log)
        res = dict(family=a.family, model=k, start=a.start, v1=round(v1), init=metrics(v0, true), final=metrics(v, true), mean_model=metrics(mean_model, true),
                   misfit_first=log[0]['misfit'], misfit_last=log[-1]['misfit'], iterations=len(log), seconds=round(time.time() - t0))
        json.dump(res, open(outp, 'w')); np.save(outp.replace('.json', '.npy'), v.astype(np.float32)); print(res, flush=True)
        if a.figure:
            fig, ax = plt.subplots(1, 3, figsize=(13, 4)); ext = [0, NX * DX, NZ * DX, 0]
            for axx, (img, tt) in zip(ax, [(true, 'true'), (v0, f'start ({a.start}) {res["init"]}'), (v, f'FWI {res["final"]}')]):
                axx.imshow(img, extent=ext, cmap='viridis', vmin=true.min(), vmax=true.max()); axx.set_title(tt, fontsize=9)
            plt.tight_layout(); plt.savefig(outp.replace('.json', '.png'), dpi=100); plt.close()

if __name__ == '__main__':
    main()

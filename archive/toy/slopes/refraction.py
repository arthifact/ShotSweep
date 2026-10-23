"""Refractor velocity and depth from official gathers by a low-frequency two-layer waveform fit (GPU-batched).

Why not pick head waves: in the OpenFWI geometry (700 m spread, strong interfaces 150-400 m deep) the true first
arrival is the direct wave at every receiver (checked with an eikonal solve on true models), and the head wave runs
within one wavelet of the post-critical reflection over the whole spread. Amplitude-vs-slope and phase-vs-slope
critical-angle estimates were also too noisy (interference of several events within one 100 ms wavelet). So the
refractor is fitted as a model instead of picked as an event: a two-layer medium (v1 above, v_r below an interface
z_r(x) = z0 + dz (x - 345)/690) is simulated for many candidates at once on a 20 m grid at low frequency, and the
candidate whose gathers best correlate with the observed ones wins. Head waves and post-critical reflections are then
modelled, not detected.

Calibration (once): OpenFWI injects its 15 Hz Ricker with weight v^2 at the source cell; the simulator does the same,
and its effective source wavelet at 20 m / 2 ms / 6 Hz is a Wiener filter pooled over ten training models with known
velocity (source_filter20.npy, calibrate20). Per model: low-pass and decimate the official gathers, grid search over
(z0, v_r) with dz = 0, then (v_r, dz), then three local 3 x 3 x 3 refinements; v1 from the direct wave. Objective: the scattered field (candidate minus the homogeneous-v1 simulation) against the observed
data minus the same homogeneous simulation, per-shot normalized correlation (relative amplitudes across offsets kept,
so the critical-angle amplitude jump counts; no time mute, so the far-offset head waves and post-critical reflections
that arrive right after the direct wave are used). A refractor is reported when that correlation exceeds 0.5.

usage (test): uv run python -I toy/slopes/refraction.py --family flatfault-a --data DATA.npy --models 475,143,21,0
"""
import argparse, glob, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slopes import SRC_X, DX, DT, NX, NZ, DATUM
ROOT = '/Users/torus/Desktop/Research/renderwave/data/openfwi'
DX2, DT2, NT2, N2, PAD2, F0 = 20.0, 0.002, 500, 35, 30, 15.0
NZ2 = 36                                                                  # rows at z = 10 + 20 (i - 1): row 1 is the 10 m source/receiver line
X2 = np.arange(N2) * DX2                                                   # receivers at 0, 20, ..., 680 m (every 2nd)
FAR = (np.abs(X2[None, :] - np.array(SRC_X)[:, None]) >= 60.0).astype(float)[:, None, :]   # (5, 1, N2): the point-source near field is grid dependent; offsets < 60 m are left out
Z2 = DATUM + DX2 * (np.arange(NZ2) - 1)

def lowpass(x, fc, dt, axis):
    n = x.shape[axis]; f = np.fft.rfftfreq(n, dt); H = np.exp(-0.5 * (f / fc) ** 4); sh = [1] * x.ndim; sh[axis] = len(f)
    return np.fft.irfft(np.fft.rfft(x, axis=axis) * H.reshape(sh), n=n, axis=axis)

def prep_obs(obs, fc=6.0):
    """(5, 1000, 70) at 10 m / 1 ms -> low-passed (5, 500, 35) at 20 m / 2 ms."""
    return lowpass(obs.astype(np.float64), fc, DT, 1)[:, ::2, ::2]

class LowFwd:
    """Batched 2D acoustic FD, 4th order in space, 20 m cells, 2 ms, damped-wave absorbing layer on all sides. Rows sit
    at z = 10 + 20 (i - 1) so sources and receivers (10 m deep) are exactly on row 1; sources at their true x (split
    between two columns), receivers at x = 0, 20, ..., 680. Velocity arrays are (B, NZ2, N2)."""
    def __init__(self, device):
        self.dev = device; nz, nx = NZ2 + 2 * PAD2, N2 + 2 * PAD2
        iz = np.arange(nz); ix = np.arange(nx)
        dz = np.clip(np.maximum(PAD2 - iz, iz - (nz - 1 - PAD2)), 0, None) / PAD2; dx = np.clip(np.maximum(PAD2 - ix, ix - (nx - 1 - PAD2)), 0, None) / PAD2
        sig = 3 * 4500.0 / (2 * PAD2 * DX2) * np.log(1e5) * (dz[:, None] ** 2 + dx[None, :] ** 2); a = sig * DT2 / 2
        self.A = torch.tensor(1 / (1 + a), dtype=torch.float32, device=device); self.B = torch.tensor(1 - a, dtype=torch.float32, device=device)
        t = np.arange(NT2) * DT2; aa = (np.pi * F0 * (t - 1.2 / F0)) ** 2; self.ricker = (1 - 2 * aa) * np.exp(-aa)
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_filter20.npy')
        self.wav = np.load(f) if os.path.exists(f) else lowpass(self.ricker, 6.0, DT2, 0)
        self.src = []                                                        # per shot: list of (col, weight)
        for xs in SRC_X:
            c = xs / DX2; c0 = int(np.floor(c)); fc_ = c - c0; self.src.append([(c0, 1 - fc_)] + ([(c0 + 1, fc_)] if fc_ > 1e-6 else []))
    def __call__(self, v, wavelet=None):
        """v: (B, NZ2, N2) velocities -> gathers (B, 5, NT2, N2). OpenFWI injects the source with weight v^2 at the
        source cell; done the same here."""
        wavelet = self.wav if wavelet is None else wavelet; B = v.shape[0]; P = PAD2; nz, nx = NZ2 + 2 * P, N2 + 2 * P
        amp = (np.stack([v[:, 1, min(int(round(xs / DX2)), N2 - 1)] for xs in SRC_X], 1) / 2000.0) ** 2   # (B, 5)
        amp_t = torch.tensor(amp, dtype=torch.float32, device=self.dev)
        vp = np.pad(v, ((0, 0), (P, P), (P, P)), mode='edge'); c2 = torch.tensor(vp ** 2 * (DT2 / DX2) ** 2, dtype=torch.float32, device=self.dev)[:, None]
        w = torch.tensor(wavelet * 1e3, dtype=torch.float32, device=self.dev)
        p0 = torch.zeros(B, 5, nz, nx, device=self.dev); p1 = torch.zeros_like(p0); out = torch.zeros(B, 5, NT2, N2, device=self.dev)
        A, Bm = self.A, self.B
        for it in range(NT2):
            L = torch.zeros_like(p1)
            L[..., 2:-2, 2:-2] = (-p1[..., :-4, 2:-2] + 16 * p1[..., 1:-3, 2:-2] - 30 * p1[..., 2:-2, 2:-2] + 16 * p1[..., 3:-1, 2:-2] - p1[..., 4:, 2:-2]
                                  - p1[..., 2:-2, :-4] + 16 * p1[..., 2:-2, 1:-3] - 30 * p1[..., 2:-2, 2:-2] + 16 * p1[..., 2:-2, 3:-1] - p1[..., 2:-2, 4:]) / 12
            p2 = (2 * p1 - Bm * p0 + c2 * L) * A
            for s, taps in enumerate(self.src):
                for cc, wt in taps: p2[:, s, P + 1, P + cc] += wt * w[it] * amp_t[:, s]
            out[:, :, it] = p2[:, :, P + 1, P:P + N2]; p0, p1 = p1, p2
        return out.cpu().numpy().astype(np.float64)

def two_layer(v1, vr, z0, dz):
    """Candidate models on the 20 m grid (NZ2, N2); z_r(x) = z0 + dz (x - 345)/690, smooth over about one cell."""
    zr = np.clip(z0 + dz * (X2 - 345.0) / 690.0, 30.0, 680.0)
    st = 1 / (1 + np.exp(-np.clip((Z2[:, None] - zr[None, :]) / 8.0, -50, 50))); return v1 + (vr - v1) * st

def wiener(sim, obs, eps=1e-2):
    Sf = np.fft.rfft(sim, axis=1); Of = np.fft.rfft(obs, axis=1); num = (np.conj(Sf) * Of).sum(axis=(0, 2)); den = (np.abs(Sf) ** 2).sum(axis=(0, 2))
    return num / (den + eps * den.max())

def downsample_model(m):
    """(70, 70) at 10 m -> (NZ2, N2) on the 20 m grid (rows z = 10 + 20 (i - 1), columns x = 20 j): slowness averaged
    over the 3 x 3 cells of the 10 m model around each node (clipped at the edges)."""
    out = np.zeros((NZ2, N2)); S = 1.0 / m
    for i, z in enumerate(Z2):
        r = int(round(z / DX)); rs = slice(max(r - 1, 0), min(r + 2, NZ))
        for j in range(N2):
            c = 2 * j; out[i, j] = 1.0 / S[rs, max(c - 1, 0):min(c + 2, NX)].mean()
    return out

def calibrate20(models, gathers, eps=1e-3):
    """Effective source wavelet for the 20 m / 2 ms simulator: Wiener filter pooled over training models with known
    velocity, between LowFwd(model, low-passed Ricker) and the low-passed official gathers."""
    f = LowFwd(torch.device('mps' if torch.backends.mps.is_available() else 'cpu')); ric = lowpass(f.ricker, 6.0, DT2, 0)
    sims = f(np.stack([downsample_model(m) for m in models]), ric) * FAR[None]; obs = np.stack([prep_obs(g) for g in gathers]) * FAR[None]
    Sf = np.fft.rfft(sims, axis=2); Of = np.fft.rfft(obs, axis=2)
    F = (np.conj(Sf) * Of).sum(axis=(0, 1, 3)) / ((np.abs(Sf) ** 2).sum(axis=(0, 1, 3)) + eps * (np.abs(Sf) ** 2).sum(axis=(0, 1, 3)).max())
    return np.fft.irfft(np.fft.rfft(ric) * F, n=NT2)

class RefractorFit:
    """Joint search over (z0, dz, v_r) of a two-layer model (v1 from the direct wave) on the calibrated 20 m simulator;
    misfit = relative L2 of the scattered field (candidate minus homogeneous v1) against observed minus homogeneous."""
    def __init__(self, device=None):
        self.dev = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu')); self.fwd = LowFwd(self.dev)
    def fit(self, obs, v1, rel_gain=0.25):
        t0 = time.time(); obs2 = prep_obs(obs) * FAR; hom = self.fwd(np.full((1, NZ2, N2), v1))[0] * FAR; osc = obs2 - hom; e = (osc ** 2).sum()
        def run(cands):
            S = self.fwd(np.stack([two_layer(v1, vr, z0, dz) for vr, z0, dz in cands])) * FAR[None]; return ((S - hom[None] - osc[None]) ** 2).sum(axis=(1, 2, 3)) / e
        vrs = np.geomspace(1.12 * v1, 4600.0, 10) if 1.12 * v1 < 4600 else np.array([4600.0]); z0s = np.arange(80.0, 601.0, 20.0)
        cands = [(vr, z0, 0.0) for vr in vrs for z0 in z0s]; J = run(cands); i = int(np.argmin(J)); b, Jb = cands[i], float(J[i])
        cands = [(vr, b[1], dz) for vr in vrs for dz in (-240.0, -160.0, -80.0, 0.0, 80.0, 160.0, 240.0)]; J = run(cands); i = int(np.argmin(J))
        if J[i] < Jb: b, Jb = cands[i], float(J[i])
        for scale in (1.0, 0.5, 0.25):
            cands = [(min(b[0] * fv, 4600.0), b[1] + d0, b[2] + dd) for fv in (1 - 0.08 * scale, 1.0, 1 + 0.08 * scale) for d0 in (-20 * scale, 0, 20 * scale) for dd in (-80 * scale, 0, 80 * scale)]
            J = run(cands); i = int(np.argmin(J))
            if J[i] < Jb: b, Jb = cands[i], float(J[i])
        found = Jb < 1.0 - rel_gain and b[0] > 1.1 * v1             # the refractor must explain >= rel_gain of the scattered energy
        return dict(found=bool(found), v1=float(v1), v_r=float(b[0]), z0=float(b[1]), dz=float(b[2]), misfit=round(Jb, 3), seconds=round(time.time() - t0, 1))

    def landscape(self, obs, v1, z0s=np.arange(80.0, 601.0, 30.0), nv=8):
        """Misfit over (v_r, z0) at dz = 0, then over (v_r, dz) at the best z0. Returns dict of grids for channels."""
        obs2 = prep_obs(obs) * FAR; hom = self.fwd(np.full((1, NZ2, N2), v1))[0] * FAR; osc = obs2 - hom; e = (osc ** 2).sum() + 1e-30
        def run(cands):
            S = self.fwd(np.stack([two_layer(v1, vr, z0, dz) for vr, z0, dz in cands])) * FAR[None]; return ((S - hom[None] - osc[None]) ** 2).sum(axis=(1, 2, 3)) / e
        vrs = np.geomspace(1.12 * v1, 4600.0, nv) if 1.12 * v1 < 4600 else np.full(nv, 4600.0)
        J1 = run([(vr, z0, 0.0) for z0 in z0s for vr in vrs]).reshape(len(z0s), nv); iz = int(np.argmin(J1.min(1)))
        dzs = np.array([-240.0, -160.0, -80.0, 0.0, 80.0, 160.0, 240.0]); J2 = run([(vr, z0s[iz], dz) for dz in dzs for vr in vrs]).reshape(len(dzs), nv)
        dz_best = float(dzs[int(np.argmin(J2.min(1)))]) if J2.min() < J1.min() else 0.0
        return dict(v1=float(v1), vrs=vrs, z0s=z0s, J1=J1, dzs=dzs, J2=J2, dz=dz_best, Jmin=float(min(J1.min(), J2.min())))

def refractor_depth_x(res):
    """z_r(x) on the 70 OpenFWI columns."""
    x = np.arange(NX) * DX; return np.clip(res['z0'] + res['dz'] * (x - 345.0) / 690.0, 30.0, 680.0)

def truth_refractor(m, ratio=1.25):
    """First interface (per column) where velocity 40 m below exceeds ratio x top velocity; its depth and that velocity."""
    v1 = m[0, 0]; z = np.full(NX, np.nan); vb = np.full(NX, np.nan)
    for x in range(NX):
        col = m[:, x]; j = [i for i in range(1, NZ) if col[min(i + 4, NZ - 1)] > ratio * v1 and col[i] > col[i - 1]]
        if j: z[x] = j[0] * DX; vb[x] = col[min(j[0] + 4, NZ - 1)]
    return z, vb

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--family', default='flatfault-a'); ap.add_argument('--data', required=True); ap.add_argument('--models', default='475,143,21,0'); a = ap.parse_args()
    from slopes import direct_wave_fit
    M = np.load(sorted(glob.glob(f'{ROOT}/{a.family}/train/*.npy'))[0], mmap_mode='r'); D = np.load(a.data, mmap_mode='r'); rf = RefractorFit()
    for k in [int(x) for x in a.models.split(',')]:
        obs = np.array(D[k], np.float64); m = np.array(M[k, 0], np.float64); v1, _ = direct_wave_fit(obs); r = rf.fit(obs, v1)
        zt, vt = truth_refractor(m); zf = refractor_depth_x(r)
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items()},
              '| true z at x=50/350/650:', [None if np.isnan(zt[i]) else int(zt[i]) for i in (5, 35, 65)], 'fit:', [int(zf[i]) for i in (5, 35, 65)],
              '| true v_r:', None if np.all(np.isnan(vt)) else int(np.nanmedian(vt)), 'top', int(m[0, 0]), flush=True)

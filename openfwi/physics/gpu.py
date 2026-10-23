"""Batched wave simulation and per-shot wave-equation imaging on the GPU (Apple MPS, or CUDA, or CPU).

Sim reproduces the official OpenFWI gathers to 4-6 % on true models: 4th-order FD, 10 m / 1 ms, damped-wave absorbing
layer on all sides, sources and receivers one cell deep, the source injected with weight v^2 at the source cell (as
OpenFWI does) and one effective source wavelet calibrated once from training models (calibrate()).

images(v_bg, obs): for each model in the batch and each of the 5 shots, the adjoint (reverse-time migration) image of
the data that the background v_bg does not explain (obs - sim(v_bg)), i.e. the per-shot gradient of
0.5 ||sim - obs||^2 with respect to velocity, with the sign flipped (positive = the data want faster rock here),
divided by that shot's source illumination. Exact discrete adjoint of the forward recurrence.
"""
import numpy as np, torch
from constants import DX, DT, NX, NZ, NT, F0, SRC_X

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
SRC = [int(round(x / DX)) for x in SRC_X]
_t = np.arange(NT) * DT; _a = (np.pi * F0 * (_t - 1.2 / F0)) ** 2; RICKER = (1 - 2 * _a) * np.exp(-_a)

def lap(p):
    out = torch.zeros_like(p)
    out[..., 2:-2, 2:-2] = (-p[..., :-4, 2:-2] + 16 * p[..., 1:-3, 2:-2] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 3:-1, 2:-2] - p[..., 4:, 2:-2]
                            - p[..., 2:-2, :-4] + 16 * p[..., 2:-2, 1:-3] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 2:-2, 3:-1] - p[..., 2:-2, 4:]) / 12
    return out

class Sim:
    def __init__(self, wavelet=None, pad=40, R=1e-5):
        self.pad = pad; n = NZ + 2 * pad; i = np.arange(n); d = np.clip(np.maximum(pad - i, i - (n - 1 - pad)), 0, None) / pad
        a = 3 * 4500 / (2 * pad * DX) * np.log(1 / R) * (d[:, None] ** 2 + d[None, :] ** 2) * DT / 2
        self.A = torch.tensor(1 / (1 + a), dtype=torch.float32, device=DEV); self.B = torch.tensor(1 - a, dtype=torch.float32, device=DEV)
        self.wav = RICKER if wavelet is None else np.asarray(wavelet)

    def _setup(self, v):
        P = self.pad; vp = np.pad(np.asarray(v, np.float64), ((0, 0), (P, P), (P, P)), mode='edge')
        c2 = torch.tensor(vp ** 2 * (DT / DX) ** 2, dtype=torch.float32, device=DEV)[:, None]
        amp = torch.tensor((np.asarray(v)[:, 1, SRC] / 2000.0) ** 2, dtype=torch.float32, device=DEV)          # (B, 5): injection weight v^2
        return c2, amp

    def forward(self, v, store=False):
        """v: (B, 70, 70) m/s -> gathers (B, 5, 1000, 70) torch on DEV; with store, also the model-region Laplacian of the
        wavefield at every step (NT, B, 5, 70, 70), needed by the adjoint."""
        P = self.pad; B = len(v); n = NZ + 2 * P; c2, amp = self._setup(v); w = torch.tensor(self.wav * 1e3, dtype=torch.float32, device=DEV)
        p0 = torch.zeros(B, 5, n, n, device=DEV); p1 = torch.zeros_like(p0); out = torch.zeros(B, 5, NT, NX, device=DEV)
        Ls = torch.zeros(NT, B, 5, NZ, NX, device=DEV) if store else None; si = torch.arange(5, device=DEV); sx = torch.tensor(SRC, device=DEV) + P
        for it in range(NT):
            L = lap(p1)
            if store: Ls[it] = L[..., P:P + NZ, P:P + NX]
            p2 = (2 * p1 - self.B * p0 + c2 * L) * self.A; p2[:, si, P + 1, sx] += w[it] * amp
            out[:, :, it] = p2[:, :, P + 1, P:P + NX]; p0, p1 = p1, p2
        return out, Ls, c2

    @torch.no_grad()
    def gathers(self, v, chunk=20):
        return np.concatenate([self.forward(v[i:i + chunk])[0].cpu().numpy() for i in range(0, len(v), chunk)])

    @torch.no_grad()
    def images(self, v_bg, obs, v_mute=None, t_mute=0.12, eps=1e-4, laplacian=True):
        """v_bg: (B, 70, 70) background velocities, obs: (B, 5, 1000, 70) observed gathers -> per-shot images (B, 5, 70, 70)
        and per-shot source illumination (B, 5, 70, 70). v_mute (B,): surface velocity; the residual is muted before
        t = |offset| / v_mute + t_mute (30 ms cosine ramp), which removes what is left of the direct wave. The image is
        divided by the source illumination (+ eps x its maximum) and, with laplacian, high-pass filtered by -lap to
        suppress the smooth backscatter typical of reverse-time migration."""
        P = self.pad; B = len(v_bg); n = NZ + 2 * P
        sim, Ls, c2 = self.forward(v_bg, store=True); res = sim - torch.tensor(np.asarray(obs), dtype=torch.float32, device=DEV)
        if v_mute is not None:
            h = torch.tensor(np.abs(np.arange(NX)[None, :] * DX - np.asarray(SRC_X)[:, None]), dtype=torch.float32, device=DEV)   # (5, 70)
            t = torch.arange(NT, device=DEV, dtype=torch.float32)[:, None] * DT
            t0 = h[None] / torch.tensor(np.asarray(v_mute), dtype=torch.float32, device=DEV)[:, None, None] + t_mute            # (B, 5, 70)
            ramp = torch.clamp((t[None, None] - t0[:, :, None, :]) / 0.03, 0, 1)                                               # (B, 5, NT, 70)
            res = res * (0.5 - 0.5 * torch.cos(np.pi * ramp))
        m1 = torch.zeros(B, 5, n, n, device=DEV); m2 = torch.zeros_like(m1); g = torch.zeros(B, 5, NZ, NX, device=DEV)
        for it in range(NT - 1, -1, -1):
            Am1 = self.A * m1
            mu = 2 * Am1 + lap(c2 * Am1) - self.A * self.B * m2; mu[:, :, P + 1, P:P + NX] += res[:, :, it]
            g += (self.A * mu)[..., P:P + NZ, P:P + NX] * Ls[it]; m2, m1 = m1, mu
        vb = torch.tensor(np.asarray(v_bg), dtype=torch.float32, device=DEV)[:, None]
        grad_v = g * 2 * vb * (DT / DX) ** 2                                          # dJ/dv per shot
        illum = (Ls ** 2).sum(0)                                                       # source illumination per shot
        img = -grad_v / (illum + eps * illum.amax(dim=(2, 3), keepdim=True))
        if laplacian: img = -lap(torch.nn.functional.pad(img, (2, 2, 2, 2), mode='replicate'))[..., 2:-2, 2:-2]
        return img.cpu().numpy(), illum.cpu().numpy()

def calibrate(models, gathers, eps=1e-3):
    """One effective source wavelet for all models: Wiener filter pooled over training models with known velocity."""
    s = Sim(RICKER); sims = s.gathers(np.asarray(models, np.float64))
    Sf = np.fft.rfft(sims, axis=2); Of = np.fft.rfft(np.asarray(gathers, np.float64), axis=2)
    num = (np.conj(Sf) * Of).sum(axis=(0, 1, 3)); den = (np.abs(Sf) ** 2).sum(axis=(0, 1, 3))
    return np.fft.irfft(np.fft.rfft(RICKER) * num / (den + eps * den.max()), n=NT)

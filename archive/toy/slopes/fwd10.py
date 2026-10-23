"""Batched calibrated forward model on the Apple GPU: 10 m, 1 ms, 4th-order FD, damped-wave absorbing layer.

The OpenFWI source is the same 15 Hz Ricker for every model, injected with a weight proportional to v^2 at the source
cell (measured: best-fit scale / v_top^2 is constant to 0.3 % over 8 models of both families). The source term here is
scaled the same way; its effective filter (amplitude, phase, delay as seen by this simulator) is then calibrated once
by a Wiener filter over training models with known velocity (source_filter.npy) and reused for every model, train or
test. True-model simulations then match the official gathers to 4-6 % relative residual. sims(v_batch) -> gathers (B, 5, 1000, 70).
"""
import os, glob, numpy as np, torch
from slopes import DX, DT, NX, NZ
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = [0, 17, 34, 52, 69]
t = np.arange(1000) * DT; _a = (np.pi * 15.0 * (t - 1.2 / 15.0)) ** 2; RICKER = (1 - 2 * _a) * np.exp(-_a)

class Fwd10:
    def __init__(self, pad=40, R=1e-5):
        self.pad = pad; nz = NZ + 2 * pad; i = np.arange(nz); d = np.clip(np.maximum(pad - i, i - (nz - 1 - pad)), 0, None) / pad
        a = 3 * 4500 / (2 * pad * DX) * np.log(1 / R) * (d[:, None] ** 2 + d[None, :] ** 2) * DT / 2
        self.A = torch.tensor(1 / (1 + a), dtype=torch.float32, device=DEV); self.B = torch.tensor(1 - a, dtype=torch.float32, device=DEV)
        f = os.path.join(HERE, 'source_filter.npy'); self.wav = np.load(f) if os.path.exists(f) else RICKER
    def __call__(self, v, wav=None, chunk=20):
        wav = self.wav if wav is None else wav; outs = []
        for i in range(0, len(v), chunk): outs.append(self._run(v[i:i + chunk], wav))
        return np.concatenate(outs)
    def _run(self, v, wav):
        P = self.pad; B = v.shape[0]; nz = NZ + 2 * P; vp = np.pad(v, ((0, 0), (P, P), (P, P)), mode='edge')
        c2 = torch.tensor(vp ** 2 * (DT / DX) ** 2, dtype=torch.float32, device=DEV)[:, None]; w = torch.tensor(wav * 1e3, dtype=torch.float32, device=DEV)
        p0 = torch.zeros(B, 5, nz, nz, device=DEV); p1 = torch.zeros_like(p0); out = torch.zeros(B, 5, 1000, NX, device=DEV)
        si = torch.arange(5, device=DEV); sx = torch.tensor(SRC, device=DEV) + P
        amp = torch.tensor((v[:, 1, SRC] / 2000.0) ** 2, dtype=torch.float32, device=DEV)            # (B, 5): OpenFWI injects f * v^2
        for it in range(1000):
            L = torch.zeros_like(p1)
            L[..., 2:-2, 2:-2] = (-p1[..., :-4, 2:-2] + 16 * p1[..., 1:-3, 2:-2] - 30 * p1[..., 2:-2, 2:-2] + 16 * p1[..., 3:-1, 2:-2] - p1[..., 4:, 2:-2]
                                  - p1[..., 2:-2, :-4] + 16 * p1[..., 2:-2, 1:-3] - 30 * p1[..., 2:-2, 2:-2] + 16 * p1[..., 2:-2, 3:-1] - p1[..., 2:-2, 4:]) / 12
            p2 = (2 * p1 - self.B * p0 + c2 * L) * self.A; p2[:, si, P + 1, sx] += w[it] * amp
            out[:, :, it] = p2[:, :, P + 1, P:P + NX]; p0, p1 = p1, p2
        return out.cpu().numpy().astype(np.float64)

def calibrate(models, gathers, eps=1e-3):
    """Wiener filter F (frequency domain) with gathers ~ sim(model, Ricker) * F, pooled over training models."""
    f = Fwd10(); sims = f(np.asarray(models, np.float64), RICKER)
    Sf = np.fft.rfft(sims, axis=2); Of = np.fft.rfft(np.asarray(gathers, np.float64), axis=2)
    num = (np.conj(Sf) * Of).sum(axis=(0, 1, 3)); den = (np.abs(Sf) ** 2).sum(axis=(0, 1, 3)); F = num / (den + eps * den.max())
    return np.fft.irfft(np.fft.rfft(RICKER) * F, n=1000)

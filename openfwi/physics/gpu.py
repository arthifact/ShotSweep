"""Batched finite-difference simulation and exact discrete-adjoint images.

The velocity derivative includes the replicated absorbing-layer velocities and
velocity-dependent source strength. Images are negative data-misfit gradients,
illumination normalized and optionally Laplacian filtered; their signs are not
reliable velocity-update directions when the data are cycle skipped.
"""
import math
import numpy as np
import torch
from constants import DX, DT, NX, NZ, NT, F0, SRC_X

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
SRC = [int(round(x / DX)) for x in SRC_X]
_t = np.arange(NT) * DT
_a = (np.pi * F0 * (_t - 1.2 / F0)) ** 2
RICKER = (1 - 2 * _a) * np.exp(-_a)


def resolve_device(name='auto'):
    if name == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but this PyTorch installation has no available CUDA device')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested, but no MPS device is available')
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'unsupported device: {name}')
    return device


def lap(p):
    out = torch.zeros_like(p)
    out[..., 2:-2, 2:-2] = (-p[..., :-4, 2:-2] + 16 * p[..., 1:-3, 2:-2] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 3:-1, 2:-2] - p[..., 4:, 2:-2]
                            - p[..., 2:-2, :-4] + 16 * p[..., 2:-2, 1:-3] - 30 * p[..., 2:-2, 2:-2] + 16 * p[..., 2:-2, 3:-1] - p[..., 2:-2, 4:]) / 12
    return out


def lap_adjoint(p):
    """Transpose of lap, including its two-cell zero-output boundary."""
    out = torch.zeros_like(p)
    q = p[..., 2:-2, 2:-2] / 12
    out[..., 2:-2, 2:-2] -= 60 * q
    out[..., :-4, 2:-2] -= q
    out[..., 1:-3, 2:-2] += 16 * q
    out[..., 3:-1, 2:-2] += 16 * q
    out[..., 4:, 2:-2] -= q
    out[..., 2:-2, :-4] -= q
    out[..., 2:-2, 1:-3] += 16 * q
    out[..., 2:-2, 3:-1] += 16 * q
    out[..., 2:-2, 4:] -= q
    return out


def fold_edge_padding(g, pad):
    """Transpose of np.pad(..., mode='edge') for the last two axes."""
    p = pad
    out = g[..., p:p + NZ, p:p + NX].clone()
    out[..., 0, :] += g[..., :p, p:p + NX].sum(-2)
    out[..., -1, :] += g[..., p + NZ:, p:p + NX].sum(-2)
    out[..., :, 0] += g[..., p:p + NZ, :p].sum(-1)
    out[..., :, -1] += g[..., p:p + NZ, p + NX:].sum(-1)
    for z, zs in ((0, slice(None, p)), (-1, slice(p + NZ, None))):
        for x, xs in ((0, slice(None, p)), (-1, slice(p + NX, None))):
            out[..., z, x] += g[..., zs, xs].sum((-2, -1))
    return out


def imaging_memory_bytes(batch, pad=40, checkpoint_interval=40, bytes_per_value=4):
    """Conservative tensor-storage estimate; excludes allocator/runtime overhead."""
    fields = 2 * math.ceil(NT / checkpoint_interval) + checkpoint_interval + 24
    return int(batch * 5 * bytes_per_value * ((NZ + 2 * pad) * (NX + 2 * pad) * fields + 3 * NT * NX))


class Sim:
    def __init__(self, wavelet=None, pad=40, R=1e-5, checkpoint_interval=40,
                 device=None, dtype=torch.float32):
        if pad < 2 or checkpoint_interval < 1:
            raise ValueError('pad must be >= 2 and checkpoint_interval must be positive')
        self.pad = pad
        self.checkpoint_interval = checkpoint_interval
        self.device = DEV if device is None else resolve_device(str(device))
        self.dtype = dtype
        n = NZ + 2 * pad
        i = np.arange(n)
        d = np.clip(np.maximum(pad - i, i - (n - 1 - pad)), 0, None) / pad
        a = 3 * 4500 / (2 * pad * DX) * np.log(1 / R) * (d[:, None] ** 2 + d[None, :] ** 2) * DT / 2
        self.A = self._tensor(1 / (1 + a))
        self.B = self._tensor(1 - a)
        self.wav = np.asarray(RICKER if wavelet is None else wavelet)
        if self.wav.shape != (NT,) or not np.isfinite(self.wav).all() or not np.any(self.wav):
            raise ValueError(f'wavelet must contain {NT} finite samples and nonzero energy')
        self.w = self._tensor(self.wav * 1e3)
        self.si = torch.arange(5, device=self.device)
        self.sx = torch.tensor(SRC, device=self.device) + pad

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _setup(self, v):
        v = np.asarray(v, np.float64)
        if v.ndim != 3 or v.shape[1:] != (NZ, NX) or not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError(f'velocities must be finite positive (B, {NZ}, {NX}) arrays')
        p = self.pad
        vp = np.pad(v, ((0, 0), (p, p), (p, p)), mode='edge')
        c2 = self._tensor(vp ** 2 * (DT / DX) ** 2)[:, None]
        amp = self._tensor((v[:, 1, SRC] / 2000.0) ** 2)
        return c2, amp

    def _step(self, p0, p1, c2, amp, it):
        L = lap(p1)
        p2 = (2 * p1 - self.B * p0 + c2 * L) * self.A
        p2[:, self.si, self.pad + 1, self.sx] += self.w[it] * amp
        return p2, L

    def forward(self, v, store=False):
        """Return gathers, optional wavefield checkpoints, and squared Courant numbers.

        Checkpoints hold states before each segment; the adjoint recomputes one
        segment at a time instead of storing all 1,000 padded wavefields.
        """
        p = self.pad
        c2, amp = self._setup(v)
        n = NZ + 2 * p
        p0 = torch.zeros(len(v), 5, n, n, device=self.device, dtype=self.dtype)
        p1 = torch.zeros_like(p0)
        out = torch.zeros(len(v), 5, NT, NX, device=self.device, dtype=self.dtype)
        checkpoints = {} if store else None
        for it in range(NT):
            if store and it % self.checkpoint_interval == 0:
                checkpoints[it] = (p0.clone(), p1.clone())
            p2, _ = self._step(p0, p1, c2, amp, it)
            out[:, :, it] = p2[:, :, p + 1, p:p + NX]
            p0, p1 = p1, p2
        return out, checkpoints, c2

    @torch.no_grad()
    def gathers(self, v, chunk=20):
        return np.concatenate([self.forward(v[i:i + chunk])[0].cpu().numpy() for i in range(0, len(v), chunk)])

    def mute_weights(self, v_mute, t_mute=0.12):
        v_mute = np.asarray(v_mute)
        if not np.isfinite(v_mute).all() or np.any(v_mute <= 0):
            raise ValueError('mute velocities must be finite and positive')
        h = self._tensor(np.abs(np.arange(NX)[None, :] * DX - np.asarray(SRC_X)[:, None]))
        t = torch.arange(NT, device=self.device, dtype=self.dtype)[:, None] * DT
        t0 = h[None] / self._tensor(v_mute)[:, None, None] + t_mute
        ramp = torch.clamp((t[None, None] - t0[:, :, None, :]) / 0.03, 0, 1)
        return 0.5 - 0.5 * torch.cos(np.pi * ramp)

    def _gradient(self, v_bg, obs, v_mute=None, t_mute=0.12):
        p = self.pad
        sim, checkpoints, c2 = self.forward(v_bg, store=True)
        observed = self._tensor(np.asarray(obs))
        if observed.shape != sim.shape or not torch.isfinite(observed).all():
            raise ValueError(f'observations must be finite with shape {tuple(sim.shape)}')
        res = sim - observed
        if v_mute is not None:
            res *= self.mute_weights(v_mute, t_mute)
        _, amp = self._setup(v_bg)
        shape = (len(v_bg), 5, NZ + 2 * p, NX + 2 * p)
        m1 = torch.zeros(shape, device=self.device, dtype=self.dtype)
        m2 = torch.zeros_like(m1)
        g = torch.zeros_like(m1)
        source_grad = torch.zeros(len(v_bg), 5, device=self.device, dtype=self.dtype)
        illum = torch.zeros(len(v_bg), 5, NZ, NX, device=self.device, dtype=self.dtype)
        source_derivative = 2 * self._tensor(np.asarray(v_bg)[:, 1, SRC]) / 2000.0 ** 2
        for start in reversed(list(checkpoints)):
            p0, p1 = checkpoints.pop(start)
            end = min(start + self.checkpoint_interval, NT)
            laplacians = []
            for it in range(start, end):
                p2, L = self._step(p0, p1, c2, amp, it)
                laplacians.append(L)
                p0, p1 = p1, p2
            for it in range(end - 1, start - 1, -1):
                L = laplacians.pop()
                mu = 2 * self.A * m1 + lap_adjoint(c2 * self.A * m1) - self.A * self.B * m2
                mu[:, :, p + 1, p:p + NX] += res[:, :, it]
                g += self.A * mu * L
                source_grad += mu[:, self.si, p + 1, self.sx] * self.w[it] * source_derivative
                illum += L[..., p:p + NZ, p:p + NX] ** 2
                m2, m1 = m1, mu
        vp = self._tensor(np.pad(np.asarray(v_bg), ((0, 0), (p, p), (p, p)), mode='edge'))[:, None]
        grad = fold_edge_padding(g * 2 * vp * (DT / DX) ** 2, p)
        grad[:, self.si, 1, torch.tensor(SRC, device=self.device)] += source_grad
        return grad, illum

    @torch.no_grad()
    def gradient(self, v_bg, obs, v_mute=None, t_mute=0.12):
        """Per-shot derivative of .5*sum(weight*(sim-obs)**2), with fixed mute weights."""
        grad, illum = self._gradient(v_bg, obs, v_mute, t_mute)
        return grad.cpu().numpy(), illum.cpu().numpy()

    @torch.no_grad()
    def images(self, v_bg, obs, v_mute=None, t_mute=0.12, eps=1e-4, laplacian=True):
        """Return illumination-normalized negative gradients and source illumination.

        The direct-wave cosine mute has a 30 ms ramp starting at offset/v_mute
        + t_mute. Its weights are fixed during differentiation.
        """
        if eps <= 0:
            raise ValueError('illumination regularization eps must be positive')
        grad, illum = self._gradient(v_bg, obs, v_mute, t_mute)
        denominator = illum + eps * illum.amax(dim=(2, 3), keepdim=True)
        img = -grad / denominator.clamp_min(torch.finfo(self.dtype).tiny)
        if laplacian:
            img = -lap(torch.nn.functional.pad(img, (2, 2, 2, 2), mode='replicate'))[..., 2:-2, 2:-2]
        return img.cpu().numpy(), illum.cpu().numpy()


def calibrate(models, gathers, eps=1e-3):
    """One effective source wavelet pooled over training models with known velocity."""
    s = Sim(RICKER)
    sims = s.gathers(np.asarray(models, np.float64))
    Sf = np.fft.rfft(sims, axis=2)
    Of = np.fft.rfft(np.asarray(gathers, np.float64), axis=2)
    num = (np.conj(Sf) * Of).sum(axis=(0, 1, 3))
    den = (np.abs(Sf) ** 2).sum(axis=(0, 1, 3))
    return np.fft.irfft(np.fft.rfft(RICKER) * num / (den + eps * den.max()), n=NT)

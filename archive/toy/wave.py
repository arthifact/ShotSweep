"""Tiny 2D acoustic wave solver (2nd order time, 4th order space, Cerjan sponge).
Autograd through the time loop gives the exact adjoint-state gradient dJ/dv.
"""
import math
import torch
import torch.nn.functional as F

NZ, NX = 32, 40          # grid (depth, width)
DX = 15.0                # m
DT = 2.0e-3              # s  (stable for vmax<=4500 with 4th-order stencil)
NT = 240                 # 0.48 s record
F0 = 8.0                 # Ricker peak frequency, Hz
SRC_Z = 2
SRC_X = [6, 20, 33]      # 3 shots
REC_Z = 2
REC_X = torch.arange(2, NX - 2)   # 36 receivers
SPONGE = 6
VMIN, VMAX = 1400.0, 4500.0
NS, NR = len(SRC_X), len(REC_X)

_k1 = torch.tensor([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
_K = torch.zeros(1, 1, 5, 5)
_K[0, 0, 2, :] += _k1
_K[0, 0, :, 2] += _k1
_K = _K / DX ** 2


def _damp():
    d = torch.ones(NZ, NX)
    a = 0.09
    for i in range(SPONGE):
        w = math.exp(-(a * (SPONGE - i)) ** 2)
        d[i, :] = torch.minimum(d[i, :], torch.tensor(w))
        d[NZ - 1 - i, :] = torch.minimum(d[NZ - 1 - i, :], torch.tensor(w))
        d[:, i] = torch.minimum(d[:, i], torch.tensor(w))
        d[:, NX - 1 - i] = torch.minimum(d[:, NX - 1 - i], torch.tensor(w))
    return d


_DAMP = _damp()


def ricker(f0=F0):
    t = torch.arange(NT) * DT - 1.0 / f0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a)


_WAV = ricker()
_SRC = torch.zeros(NS, 1, NZ, NX)
for s, x in enumerate(SRC_X):
    _SRC[s, 0, SRC_Z, x] = 1.0


def forward(v):
    """v: (B, NZ, NX) m/s -> data (B, NS, NT, NR)."""
    B = v.shape[0]
    vv = v[:, None, None].expand(B, NS, 1, NZ, NX).reshape(B * NS, 1, NZ, NX)
    c2 = (vv * DT) ** 2
    src = _SRC.repeat(B, 1, 1, 1) * 1e3
    u_prev = torch.zeros_like(c2)
    u = torch.zeros_like(c2)
    recs = []
    for it in range(NT):
        lap = F.conv2d(u, _K, padding=2)
        u_next = (2 * u - u_prev + c2 * lap + src * _WAV[it]) * _DAMP
        u_prev, u = u * _DAMP, u_next
        recs.append(u[:, 0, REC_Z, REC_X])
    d = torch.stack(recs, 1)  # (B*NS, NT, NR)
    return d.reshape(B, NS, NT, NR)


def misfit(v, d_obs):
    d = forward(v)
    return 0.5 * ((d - d_obs) ** 2).sum(dim=(1, 2, 3))  # per sample


def gradient(v, d_obs):
    """Exact adjoint gradient dJ/dv, per sample. Returns (J, g), both detached."""
    v = v.detach().requires_grad_(True)
    J = misfit(v, d_obs)
    (g,) = torch.autograd.grad(J.sum(), v)
    return J.detach(), g.detach()


def start_model(B):
    z = torch.arange(NZ).float() / (NZ - 1)
    v0 = 1500 + 1500 * z
    return v0[None, :, None].expand(B, NZ, NX).clone()

"""Gaussian scene state: coarse smooth background + K Gaussian slots (amplitude = existence)."""
import math
import torch
import torch.nn.functional as F
from wave import NZ, NX, VMIN, VMAX

K = 12
BGZ, BGX = 6, 8
_zz = torch.linspace(0, 1, NZ)[:, None].expand(NZ, NX)
_xx = torch.linspace(0, 1, NX)[None, :].expand(NZ, NX)


def init_state(v0):
    """v0 (B,NZ,NX) m/s -> bg (B,BGZ,BGX) km/s, slots (B,K,5) = [cz, cx, log sz, log sx, amp km/s]."""
    B = v0.shape[0]
    bg = F.interpolate(v0[:, None], size=(BGZ, BGX), mode="bilinear", align_corners=True)[:, 0] / 1000.0
    cz, cx = torch.meshgrid(torch.linspace(0.17, 0.83, 3), torch.linspace(0.125, 0.875, 4), indexing="ij")
    slots = torch.zeros(B, K, 5)
    slots[:, :, 0] = cz.reshape(-1)
    slots[:, :, 1] = cx.reshape(-1)
    slots[:, :, 2:4] = math.log(0.12)
    return bg, slots


def render(bg, slots):
    v = F.interpolate(bg[:, None], size=(NZ, NX), mode="bilinear", align_corners=True)[:, 0]
    cz, cx = slots[..., 0, None, None], slots[..., 1, None, None]
    sz, sx = slots[..., 2, None, None].exp(), slots[..., 3, None, None].exp()
    a = slots[..., 4, None, None]
    e = torch.exp(-0.5 * (((_zz - cz) / sz) ** 2 + ((_xx - cx) / sx) ** 2))
    v = v + (a * e).sum(1)
    return (v * 1000.0).clamp(VMIN, VMAX)


def param_grads(bg, slots, g):
    """Exact chain rule: dJ/d(bg), dJ/d(slots) from the pixel gradient g = dJ/dv."""
    bg = bg.detach().requires_grad_(True)
    slots = slots.detach().requires_grad_(True)
    v = render(bg, slots)
    gb, gs = torch.autograd.grad(v, (bg, slots), grad_outputs=g)
    return gb.detach(), gs.detach()


def render_pixel(vkm):
    return (vkm * 1000.0).clamp(VMIN, VMAX)

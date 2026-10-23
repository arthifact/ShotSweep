"""Synthetic velocity models: dipping layers + (ellipse anomaly | fault).
Deliberately NOT built from Gaussians, so the Gaussian state has a representation gap."""
import math
import torch
from wave import NZ, NX


def _layers(g):
    nl = int(torch.randint(2, 5, (1,), generator=g))
    depths = torch.sort(torch.rand(nl - 1, generator=g) * 20 + 6).values
    slope = (torch.rand(1, generator=g) * 0.5 - 0.25).item()
    vels = [1500 + 700 * torch.rand(1, generator=g).item()]
    for _ in range(nl - 1):
        vels.append(min(vels[-1] + 200 + 500 * torch.rand(1, generator=g).item(), 4000))
    z = torch.arange(NZ).float()[:, None].expand(NZ, NX)
    x = torch.arange(NX).float()[None, :].expand(NZ, NX)
    return z, x, depths, slope, vels


def _fill(z, x, depths, slope, vels, zshift=None):
    zz = z if zshift is None else z + zshift
    v = torch.full((NZ, NX), vels[0])
    for d, vel in zip(depths, vels[1:]):
        v = torch.where(zz >= d + slope * (x - NX / 2), torch.full_like(v, vel), v)
    return v


def make_model(g):
    z, x, depths, slope, vels = _layers(g)
    fam = int(torch.randint(0, 2, (1,), generator=g))
    if fam == 0:  # ellipse anomaly with sharp edges
        v = _fill(z, x, depths, slope, vels)
        cz = 8 + 18 * torch.rand(1, generator=g).item()
        cx = 8 + 24 * torch.rand(1, generator=g).item()
        az = 3 + 5 * torch.rand(1, generator=g).item()
        ax = 4 + 8 * torch.rand(1, generator=g).item()
        th = math.pi * torch.rand(1, generator=g).item()
        sgn = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
        dv = sgn * (300 + 500 * torch.rand(1, generator=g).item())
        zr = (z - cz) * math.cos(th) - (x - cx) * math.sin(th)
        xr = (z - cz) * math.sin(th) + (x - cx) * math.cos(th)
        inside = (zr / az) ** 2 + (xr / ax) ** 2 <= 1
        v = torch.where(inside, v + dv, v)
    else:  # fault: throw the layers on one side of a dipping line
        xf = 10 + 20 * torch.rand(1, generator=g).item()
        tan = math.tan((torch.rand(1, generator=g).item() - 0.5) * math.pi / 3)
        throw = 2 + 5 * torch.rand(1, generator=g).item()
        side = x > xf + tan * (z - NZ / 2)
        zshift = torch.where(side, torch.full_like(z, throw), torch.zeros_like(z))
        v = _fill(z, x, depths, slope, vels, zshift)
    return v.clamp(1500, 4200), fam


def make_models(n, seed):
    g = torch.Generator().manual_seed(seed)
    vs, fams = [], []
    for _ in range(n):
        v, f = make_model(g)
        vs.append(v)
        fams.append(f)
    return torch.stack(vs), torch.tensor(fams)

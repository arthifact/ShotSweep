"""Background velocities, the same recipe for every family.

surface_velocity: the direct wave is the first arrival at every receiver in this geometry, so the onset time grows
linearly with offset; a least-squares line over offsets 60-250 m on all five shots gives the surface velocity.
depth_trend: the mean increase of velocity with depth over the training models of all families (labels of training
models only). backgrounds: v_k(z) = v_surface + rho_k * trend(z), laterally constant, for slow / medium / fast rho_k,
so every background has the right surface velocity (the direct wave cancels) and a different velocity at depth.
"""
import numpy as np
from constants import DX, DT, NX, NZ, SRC_X
RHOS = (0.6, 1.0, 1.4)

def surface_velocity(obs, frac=0.05, hmin=60.0, hmax=250.0):
    hs, ts = [], []
    for s, g in enumerate(obs):
        a = np.abs(g)
        for r in range(NX):
            h = abs(r * DX - SRC_X[s])
            if hmin <= h <= hmax:
                hs.append(h); ts.append(np.argmax(a[:, r] > frac * a[:, r].max()) * DT)
    slope, _ = np.polyfit(np.array(hs), np.array(ts), 1)
    return float(np.clip(1.0 / slope, 1400.0, 4600.0))

def depth_trend(models):
    """models: (N, 70, 70) training velocity models -> mean v(z) - v(0) over models and positions, (70,)."""
    m = np.asarray(models, np.float64); return (m.mean(2) - m[:, :1, :].mean(2)).mean(0)

def backgrounds(v_surface, trend, rhos=RHOS):
    """-> (len(rhos), 70, 70) laterally constant background models."""
    return np.stack([np.tile(np.clip(v_surface + r * trend, 1500.0, 4500.0)[:, None], (1, NX)) for r in rhos])

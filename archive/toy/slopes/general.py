"""General (any-velocity-model) version of the slope pipeline: structure-from-waves bundle adjustment.

Scene  = reflector elements (position x, normal n) + smooth slowness field (coarse control grid, bilinear to the model)
Camera = the slowness field itself: traveltime tables T_i(x) for every source and receiver (batched eikonal solver)
Decoder (generic kinematic migration): an observed oriented point (shot s, receiver r, time t, slope p) maps to the
         subsurface point x where T_s(x) + T_r(x) = t and the receiver-side horizontal slowness S_r(x) = p;
         its normal is the bisector grad(T_s + T_r). Elements = clusters of migrated points across shots.
Loop   = bundle adjustment: unknowns are element positions and the coarse slowness; each matched observation gives
         T_s(x_e) + T_r(x_e) = t with ray-traced sensitivities (path length of the two rays in each control cell);
         then re-solve the tables, re-migrate, re-cluster (densify / prune), repeat.
Init   = the flat-layer pipeline's v(z) (slopes.py).

usage: uv run python -I toy/slopes/general.py --family flatfault-a --model 0 [--iters 8] [--sim]
Gathers are simulated here in the OpenFWI 5-shot geometry (official FlatFault-A gathers are not on disk).
"""
import argparse, json, os, sys, time, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slopes as S
from slopes import DX, DT, NX, NZ, SRC_X, DATUM, direct_wave_fit, oriented_points, vote, pick_interfaces, dix, layers_to_vz, MEAN_MSE
torch.set_num_threads(4)
ROOT = '/Users/torus/Desktop/Research/renderwave/data/openfwi'
F0 = 15.0; NT = 1000; PAD = 20; H = float(os.environ.get("EIK_H", 5.0))   # H: eikonal grid spacing (m), finer than the model for accuracy

# ----------------------------------------------------------------------------------------------------- simulation
def simulate(v, src_cells, rec_row=1, ref=2, free_surface=True):
    """4th-order-space, 2nd-order-time acoustic FD in the OpenFWI geometry: 15 Hz Ricker, sources and receivers one
    model cell deep, sponge in a 20-cell pad outside the model on all sides (no free surface, like OpenFWI). The model is
    upsampled `ref` times (nearest neighbour, so it stays blocky) and the time step divided by `ref`, to keep numerical
    dispersion out of the wavelet (at 10 m cells the 40 Hz content rings and the trailing lobes migrate as ghost layers)."""
    vf = np.repeat(np.repeat(v, ref, 0), ref, 1); dx = DX / ref; dt = DT / ref; pad = PAD * ref
    vp = np.pad(vf, pad, mode='edge'); nz, nx = vp.shape
    c2 = torch.tensor(vp, dtype=torch.float64) ** 2 * (dt / dx) ** 2
    iz = np.arange(nz)[:, None]; ix = np.arange(nx)[None, :]; m = np.minimum(np.minimum(ix, nx - 1 - ix), np.minimum(iz if not free_surface else 10 ** 6, nz - 1 - iz))
    d = np.where(m < pad, 0.015 * ((pad - m) / pad) ** 2 / ref, 0.0); damp = torch.tensor(np.exp(-d))
    t = np.arange(NT * ref) * dt; a = (np.pi * F0 * (t - 1.2 / F0)) ** 2; ricker = torch.tensor((1 - 2 * a) * np.exp(-a))
    out = np.zeros((len(src_cells), NT, NX)); rr = pad + rec_row * ref; cols = pad + np.arange(NX) * ref
    for s, sx in enumerate(src_cells):
        p0 = torch.zeros(nz, nx, dtype=torch.float64); p1 = torch.zeros_like(p0)
        for it in range(NT * ref):
            lap = torch.zeros_like(p1)
            lap[2:-2, 2:-2] = (-p1[:-4, 2:-2] + 16 * p1[1:-3, 2:-2] - 30 * p1[2:-2, 2:-2] + 16 * p1[3:-1, 2:-2] - p1[4:, 2:-2]
                               - p1[2:-2, :-4] + 16 * p1[2:-2, 1:-3] - 30 * p1[2:-2, 2:-2] + 16 * p1[2:-2, 3:-1] - p1[2:-2, 4:]) / 12
            p2 = (2 * p1 - p0 + c2 * lap) * damp
            if free_surface: p2[pad - 1:pad + 1] = 0                  # pressure-free surface at z = 0 (the model's top edge)
            p2[rr, pad + sx * ref] += ricker[it] * 1e3
            if it % ref == 0: out[s, it // ref] = p2[rr, cols].numpy()
            p0, p1 = p1, p2
    return out

# ----------------------------------------------------------------------------------------------------- eikonal
NZE, NXE = int(round((NZ - 1) * DX / H)) + 1, int(round((NX - 1) * DX / H)) + 1   # eikonal grid nodes (139 x 139 at H = 5)
def model_to_eik(v):
    """Bilinear resample of a (NZ, NX) model-grid field to the eikonal grid."""
    zc = np.arange(NZE) * H / DX; xc = np.arange(NXE) * H / DX
    i0 = np.clip(np.floor(zc).astype(int), 0, NZ - 2); j0 = np.clip(np.floor(xc).astype(int), 0, NX - 2)
    fz = (zc - i0)[:, None]; fx = (xc - j0)[None, :]
    return (v[i0][:, j0] * (1 - fz) * (1 - fx) + v[i0 + 1][:, j0] * fz * (1 - fx) + v[i0][:, j0 + 1] * (1 - fz) * fx + v[i0 + 1][:, j0 + 1] * fz * fx)

def eikonal(slow_e, points, max_iter=700, tol=1e-6):
    """Batched Godunov upwind eikonal solver by Jacobi iteration. slow_e: (NZE, NXE) slowness on the eikonal grid,
    points: list of (z, x) in metres. Returns T: (B, NZE, NXE) seconds. Near-source nodes get the exact homogeneous time."""
    s = torch.tensor(slow_e); B = len(points); inf = 1e9
    T = torch.full((B, NZE, NXE), inf, dtype=torch.float64)
    zz = torch.arange(NZE, dtype=torch.float64)[None, :, None] * H; xx = torch.arange(NXE, dtype=torch.float64)[None, None, :] * H
    pz = torch.tensor([p[0] for p in points])[:, None, None]; px = torch.tensor([p[1] for p in points])[:, None, None]
    dist = torch.sqrt((zz - pz) ** 2 + (xx - px) ** 2)
    iz = (pz / H).round().long().flatten(); ix = (px / H).round().long().flatten()
    s_src = s[iz, ix][:, None, None]
    near = dist <= 4 * H; T = torch.where(near, dist * s_src, T); fixed = near
    hs = (H * s)[None]
    for k in range(max_iter):
        Tp = torch.nn.functional.pad(T, (1, 1, 1, 1), value=inf)
        a = torch.minimum(Tp[:, :-2, 1:-1], Tp[:, 2:, 1:-1]); b = torch.minimum(Tp[:, 1:-1, :-2], Tp[:, 1:-1, 2:])
        d = torch.abs(a - b); mn = torch.minimum(a, b)
        two = 0.5 * (a + b + torch.sqrt(torch.clamp(2 * hs ** 2 - d ** 2, min=0)))
        Tn = torch.where(d >= hs, mn + hs, two)
        Tn = torch.where(fixed, T, torch.minimum(T, Tn))
        if k % 20 == 19 and (Tn - T).abs().max() < tol: T = Tn; break
        T = Tn
    return T.numpy()

def sample(Tb, z, x):
    """Bilinear sample of table(s) Tb (..., NZE, NXE) at metres (z, x) arrays."""
    fz = np.clip(z / H, 0, NZE - 1.001); fx = np.clip(x / H, 0, NXE - 1.001); i0 = fz.astype(int); j0 = fx.astype(int); dz = fz - i0; dx = fx - j0
    return (Tb[..., i0, j0] * (1 - dz) * (1 - dx) + Tb[..., i0 + 1, j0] * dz * (1 - dx) + Tb[..., i0, j0 + 1] * (1 - dz) * dx + Tb[..., i0 + 1, j0 + 1] * dz * dx)

def sample_each(Tb, idx, z, x):
    """Elementwise bilinear sample: value of table idx[i] at (z[i], x[i])."""
    fz = np.clip(z / H, 0, NZE - 1.001); fx = np.clip(x / H, 0, NXE - 1.001); i0 = fz.astype(int); j0 = fx.astype(int); dz = fz - i0; dx = fx - j0
    return Tb[idx, i0, j0] * (1 - dz) * (1 - dx) + Tb[idx, i0 + 1, j0] * dz * (1 - dx) + Tb[idx, i0, j0 + 1] * (1 - dz) * dx + Tb[idx, i0 + 1, j0 + 1] * dz * dx

def grad_tables(Tb):
    gz = np.gradient(Tb, H, axis=-2); gx = np.gradient(Tb, H, axis=-1); return gz, gx

# ----------------------------------------------------------------------------------------------------- scene
class Tables:
    """Traveltime tables for the 5 sources and the 70 receivers (all at depth DATUM) on the current slowness field."""
    def __init__(self, v_model):
        self.v = v_model; se = 1.0 / model_to_eik(v_model)
        pts = [(DATUM, x) for x in SRC_X] + [(DATUM, r * DX) for r in range(NX)]
        T = eikonal(se, pts); self.Ts, self.Tr = T[:5], T[5:]
        self.gTs, self.gTr = grad_tables(self.Ts), grad_tables(self.Tr)
        # receiver-side horizontal slowness of the ray from receiver r to x: d T_r(x) / d(receiver x)
        Sr = np.zeros_like(self.Tr); Sr[1:-1] = (self.Tr[2:] - self.Tr[:-2]) / (2 * DX); Sr[0] = (self.Tr[1] - self.Tr[0]) / DX; Sr[-1] = (self.Tr[-1] - self.Tr[-2]) / DX
        self.Sr = Sr; self.gSr = grad_tables(Sr)

def migrate(tab, obs, sig_t=0.004, sig_p=2e-5):
    """Generic kinematic migration: each observed point (s, r, t, p, w) -> model point (z, x) minimising
    ((T_s + T_r - t)/sig_t)^2 + ((S_r - p)/sig_p)^2 over the eikonal grid, plus its normal from grad(T_s + T_r)."""
    s, r, t, p, w = obs; n = len(t); Z = np.zeros(n); X = np.zeros(n); NZv = np.zeros(n); NXv = np.zeros(n); cost = np.zeros(n)
    zz = np.arange(NZE) * H; xx = np.arange(NXE) * H
    for i in range(n):
        Tsum = tab.Ts[s[i]] + tab.Tr[r[i]]
        c = ((Tsum - t[i]) / sig_t) ** 2 + ((tab.Sr[r[i]] - p[i]) / sig_p) ** 2
        c[:int(3 * DATUM / H), :] = 1e18                        # not in the top rows (direct wave region)
        k = np.argmin(c); iz, ix = divmod(k, NXE); Z[i], X[i], cost[i] = zz[iz], xx[ix], c[iz, ix]
        gz = tab.gTs[0][s[i], iz, ix] + tab.gTr[0][r[i], iz, ix]; gx = tab.gTs[1][s[i], iz, ix] + tab.gTr[1][r[i], iz, ix]
        nn = np.hypot(gz, gx) + 1e-12; NZv[i], NXv[i] = gz / nn, gx / nn
    return Z, X, NZv, NXv, cost

def cluster(Z, X, NZv, NXv, w, shots, radius=12.0, max_angle_deg=25.0, min_members=2):
    """Greedy radius clustering of migrated points with compatible normals -> reflector elements."""
    order = np.argsort(-w); taken = np.zeros(len(Z), bool); elems = []
    cosmax = np.cos(np.radians(max_angle_deg))
    for i in order:
        if taken[i]: continue
        d = np.hypot(Z - Z[i], X - X[i]); dots = NZv * NZv[i] + NXv * NXv[i]
        m = (~taken) & (d <= radius) & (dots >= cosmax)
        idx = np.nonzero(m)[0]
        if len(idx) < min_members: continue
        taken[idx] = True; ww = w[idx]
        elems.append(dict(z=float((Z[idx] * ww).sum() / ww.sum()), x=float((X[idx] * ww).sum() / ww.sum()), nz=float((NZv[idx] * ww).sum()), nx=float((NXv[idx] * ww).sum()),
                          members=idx, nshots=len(set(shots[idx]))))
    for e in elems:
        nn = np.hypot(e['nz'], e['nx']) + 1e-12; e['nz'] /= nn; e['nx'] /= nn
    return elems

# ----------------------------------------------------------------------------------------------------- tomography
NCZ, NCX = 8, 6                                                                   # coarse slowness control grid
def control_to_model(c):
    """Bilinear (NCZ, NCX) control values -> (NZ, NX) model-grid field."""
    zc = np.linspace(0, NCZ - 1, NZ); xc = np.linspace(0, NCX - 1, NX)
    i0 = np.clip(np.floor(zc).astype(int), 0, NCZ - 2); j0 = np.clip(np.floor(xc).astype(int), 0, NCX - 2)
    fz = (zc - i0)[:, None]; fx = (xc - j0)[None, :]
    return c[i0][:, j0] * (1 - fz) * (1 - fx) + c[i0 + 1][:, j0] * fz * (1 - fx) + c[i0][:, j0 + 1] * (1 - fz) * fx + c[i0 + 1][:, j0 + 1] * fz * fx

def basis_weights(z, x):
    """Bilinear weights of the control points at model points (z, x) in metres (arrays): (N, NCZ*NCX)."""
    z = np.atleast_1d(z); x = np.atleast_1d(x)
    fz = np.clip(z / ((NZ - 1) * DX) * (NCZ - 1), 0, NCZ - 1.001); fx = np.clip(x / ((NX - 1) * DX) * (NCX - 1), 0, NCX - 1.001)
    i0 = fz.astype(int); j0 = fx.astype(int); dz = fz - i0; dx = fx - j0; w = np.zeros((len(z), NCZ, NCX)); n = np.arange(len(z))
    w[n, i0, j0] = (1 - dz) * (1 - dx); w[n, i0 + 1, j0] = dz * (1 - dx); w[n, i0, j0 + 1] = (1 - dz) * dx; w[n, i0 + 1, j0 + 1] = dz * dx
    return w.reshape(len(z), -1)

def trace_rays(Tb, gz, gx, idx, z0, x0, step=H, max_steps=400):
    """Follow -grad T of table idx[i] from (z0[i], x0[i]) back to that table's source, all rays at once.
    Returns the control-point sensitivity rows (N, NCZ*NCX): path length of each ray weighted by the bilinear basis."""
    z = z0.astype(float).copy(); x = x0.astype(float).copy(); rows = np.zeros((len(z), NCZ * NCX)); active = np.ones(len(z), bool)
    for _ in range(max_steps):
        fz = np.clip(z / H, 0, NZE - 1.001); fx = np.clip(x / H, 0, NXE - 1.001); i0 = fz.astype(int); j0 = fx.astype(int); dz = fz - i0; dx = fx - j0
        def smp(A): return A[idx, i0, j0] * (1 - dz) * (1 - dx) + A[idx, i0 + 1, j0] * dz * (1 - dx) + A[idx, i0, j0 + 1] * (1 - dz) * dx + A[idx, i0 + 1, j0 + 1] * dz * dx
        g_z, g_x, tt = smp(gz), smp(gx), smp(Tb); nn = np.hypot(g_z, g_x)
        active &= (nn > 1e-9) & (tt > 2e-3)                       # stop within ~2 ms of the source
        if not active.any(): break
        rows[active] += basis_weights(z[active], x[active]) * step
        z[active] -= step * g_z[active] / nn[active]; x[active] -= step * g_x[active] / nn[active]
        z = np.clip(z, 0, (NZ - 1) * DX); x = np.clip(x, 0, (NX - 1) * DX)
    return rows

def tomography_step(tab, elems, obs, ctrl_slow, lam_smooth=0.01, lam_damp=0.003, max_rel=0.05, sig_t=2e-3):
    """Slowness update with the reflector elements held at their cross-shot cluster centroids (the slopes already placed
    them). For each element e and member observation i: T_s(x_e) + T_r(x_e) + L_i.dc = t_i, L_i = ray-traced sensitivity
    of the source and receiver rays to the control slowness. The inconsistency of one element's members across shots is
    exactly the common-image-gather flatness criterion. Regularisation in seconds per unit relative slowness change."""
    s, r, t, p, w = obs; ne = len(elems); nc = NCZ * NCX
    if ne == 0: return ctrl_slow, float('nan')
    ek = np.concatenate([np.full(len(e['members']), k) for k, e in enumerate(elems)]); oi = np.concatenate([e['members'] for e in elems])
    ze = np.array([elems[k]['z'] for k in ek]); xe = np.array([elems[k]['x'] for k in ek]); si = s[oi]; ri = r[oi]
    fz = np.clip(ze / H, 0, NZE - 1.001); fx = np.clip(xe / H, 0, NXE - 1.001); i0 = fz.astype(int); j0 = fx.astype(int); dz = fz - i0; dx = fx - j0
    def smp(A, idx): return A[idx, i0, j0] * (1 - dz) * (1 - dx) + A[idx, i0 + 1, j0] * dz * (1 - dx) + A[idx, i0, j0 + 1] * (1 - dz) * dx + A[idx, i0 + 1, j0 + 1] * dz * dx
    tpred = smp(tab.Ts, si) + smp(tab.Tr, ri)
    L = trace_rays(tab.Ts, tab.gTs[0], tab.gTs[1], si, ze, xe) + trace_rays(tab.Tr, tab.gTr[0], tab.gTr[1], ri, ze, xe)
    res = t[oi] - tpred; wt = np.sqrt(w[oi]) / sig_t; A = L * wt[:, None]; b = res * wt
    C = ctrl_slow.reshape(NCZ, NCX); reg = []
    for i in range(NCZ):
        for j in range(NCX):
            k = i * NCX + j
            if i + 1 < NCZ: rr = np.zeros(nc); rr[k] = lam_smooth / C[i, j]; rr[k + NCX] = -lam_smooth / C[i + 1, j]; reg.append(rr)
            if j + 1 < NCX: rr = np.zeros(nc); rr[k] = lam_smooth / C[i, j]; rr[k + 1] = -lam_smooth / C[i, j + 1]; reg.append(rr)
            rr = np.zeros(nc); rr[k] = lam_damp / C[i, j]; reg.append(rr)
    A = np.vstack([A, np.array(reg) / sig_t]); b = np.concatenate([b, np.zeros(len(reg))])
    dc = np.linalg.lstsq(A, b, rcond=None)[0]; dc = np.clip(dc, -max_rel * ctrl_slow, max_rel * ctrl_slow)
    return ctrl_slow + dc, float(np.sqrt(np.mean(res ** 2)))

def residual_rms(tab, elems, obs):
    s, r, t, p, w = obs
    if not elems: return float('nan')
    ek = np.concatenate([np.full(len(e['members']), k) for k, e in enumerate(elems)]); oi = np.concatenate([e['members'] for e in elems])
    ze = np.array([elems[k]['z'] for k in ek]); xe = np.array([elems[k]['x'] for k in ek])
    tpred = sample_each(tab.Ts, s[oi], ze, xe) + sample_each(tab.Tr, r[oi], ze, xe)
    return float(np.sqrt(np.mean((t[oi] - tpred) ** 2)))

def fit_control(v_model):
    """Least-squares fit of the coarse control slowness to a model-grid slowness field (for initialisation)."""
    zz, xx = np.meshgrid(np.arange(NZ) * DX, np.arange(NX) * DX, indexing='ij')
    B = basis_weights(zz.ravel(), xx.ravel())
    return np.linalg.lstsq(B, (1.0 / v_model).ravel(), rcond=None)[0]

# ----------------------------------------------------------------------------------------------------- consistency search
NSZ, NSX = 6, int(os.environ.get('NSX', 2))                       # low-dimensional velocity for the direct search
PRIOR = float(os.environ.get('PRIOR', 0.0))                       # weight of (log v - log v_init)^2 per band: keeps unconstrained bands near the start
VLO, VHI = 1300.0, 5000.0
def search_to_model(theta):
    """theta: log-velocity in NSZ depth bands x NSX lateral bands, piecewise constant (a staircase can be represented)."""
    c = np.exp(theta.reshape(NSZ, NSX)); iz = np.minimum((np.arange(NZ) * NSZ) // NZ, NSZ - 1); ix = np.minimum((np.arange(NX) * NSX) // NX, NSX - 1)
    return c[iz][:, ix]

def consistency(theta, obs, w_o, s_o, radius=40.0, cmax=400.0, zmax=600.0, miss_ms=20.0):
    """Multi-view consistency in time: migrate every oriented point with velocity theta, cluster across shots, and score
    each multi-shot cluster's members by (observed time - time predicted at the cluster centroid)^2; a point no multi-shot
    cluster explains pays miss_ms^2. Times are observed, so a collapsing velocity cannot cheat the way a depth spread can."""
    v = search_to_model(theta)
    if v.min() < VLO or v.max() > VHI: return 1e9, dict(elements=0, explained=0)
    tab = Tables(v); s, r, t, p, w = obs
    Z, X, NZv, NXv, cost = migrate(tab, obs); good = (cost < cmax) & (Z < zmax)
    elems = cluster(Z[good], X[good], NZv[good], NXv[good], w_o[good], s_o[good], radius=radius)
    gidx = np.nonzero(good)[0]; wsum = w_o.sum(); J = 0.0; explained = np.zeros(len(Z), bool)
    for e in elems:
        if e['nshots'] < 2: continue
        m = gidx[e['members']]; explained[m] = True
        tp = sample_each(tab.Ts, s[m], np.full(len(m), e['z']), np.full(len(m), e['x'])) + sample_each(tab.Tr, r[m], np.full(len(m), e['z']), np.full(len(m), e['x']))
        J += (w_o[m] * ((t[m] - tp) * 1e3) ** 2).sum()
    J += (w_o[~explained]).sum() * miss_ms ** 2
    return J / wsum, dict(elements=len(elems), explained=int(explained.sum()))

def overlap_consistency(theta, obs, w_o, s_o, sigma=25.0, cmax=400.0, zmax=600.0, dip_deg=25.0):
    """Normalized cross-shot overlap of migrated point clouds: migrate every point with velocity theta, then score
    sum over shot pairs of K(s, s') / sqrt(K(s, s) K(s', s')), K = sum_ij w_i w_j exp(-|x_i - x_j|^2 / 2 sigma^2) over pairs
    with compatible normals. Bounded in [0, 1] per pair, invariant to depth compression, smooth. Returns 1 - mean overlap."""
    v = search_to_model(theta)
    if v.min() < VLO or v.max() > VHI: return 1e9, {}
    tab = Tables(v); Z, X, NZv, NXv, cost = migrate(tab, obs); good = (cost < cmax) & (Z < zmax)
    Z, X, NZv, NXv, w, sh = Z[good], X[good], NZv[good], NXv[good], w_o[good], s_o[good]
    d2 = (Z[:, None] - Z[None, :]) ** 2 + (X[:, None] - X[None, :]) ** 2; dots = NZv[:, None] * NZv[None, :] + NXv[:, None] * NXv[None, :]
    K = np.exp(-d2 / (2 * sigma ** 2)) * (dots >= np.cos(np.radians(dip_deg))) * w[:, None] * w[None, :]
    M = np.zeros((5, 5))
    for a_ in range(5):
        for b_ in range(5): M[a_, b_] = K[np.ix_(sh == a_, sh == b_)].sum()
    pairs = [(a_, b_) for a_ in range(5) for b_ in range(a_ + 1, 5) if M[a_, a_] > 0 and M[b_, b_] > 0]
    ov = np.mean([M[a_, b_] / np.sqrt(M[a_, a_] * M[b_, b_]) for a_, b_ in pairs]) if pairs else 0.0
    return 1.0 - ov, dict(points=int(good.sum()), overlap=round(float(ov), 4))

# --- gradient version: smooth log-v(z) spline, finite-difference gradient, illumination-weighted prior
NK = 6; NKX = int(os.environ.get('NKX', 1))                       # spline knots in depth x lateral (NKX = 1: plain v(z))
KZ = np.linspace(0, (NZ - 1) * DX, NK); KX = np.linspace(0, (NX - 1) * DX, NKX) if NKX > 1 else np.array([0.0])
def spline_to_model(theta):
    """theta: log-velocity at NK x NKX knots (flattened, depth-major), bilinear in log v -> (NZ, NX)."""
    th = theta.reshape(NK, NKX); zc = np.arange(NZ) * DX; xc = np.arange(NX) * DX
    col = np.stack([np.interp(zc, KZ, th[:, j]) for j in range(NKX)], 1)              # (NZ, NKX)
    if NKX == 1: return np.tile(np.exp(col), (1, NX))
    return np.exp(np.stack([np.interp(xc, KX, col[i]) for i in range(NZ)], 0))

def illumination(theta, obs, w_o, s_o, cmax=400.0, zmax=600.0):
    """Fraction of migrated-point weight at or below each knot and laterally within its bilinear support: a ray to a
    reflector at depth z crosses every knot above z, so knots with no points beneath them are not constrained."""
    tab = Tables(spline_to_model(theta)); Z, X, NZv, NXv, cost = migrate(tab, obs); good = (cost < cmax) & (Z < zmax)
    out = np.zeros((NK, NKX)); dz = KZ[1] - KZ[0]; dxk = (KX[1] - KX[0]) if NKX > 1 else 1e9
    for i, kz in enumerate(KZ):
        for j, kx in enumerate(KX):
            lat = np.clip(1 - np.abs(X - kx) / dxk, 0, 1) if NKX > 1 else 1.0
            out[i, j] = (w_o[good] * (Z[good] >= kz - 0.5 * dz) * (lat[good] if NKX > 1 else 1.0)).sum() / (w_o[good] * (lat[good] if NKX > 1 else 1.0)).sum()
    return out.ravel()

def overlap_spline(theta, obs, w_o, s_o):
    v = spline_to_model(theta)
    if v.min() < VLO or v.max() > VHI: return 1e9, {}
    return overlap_consistency_model(v, obs, w_o, s_o)

def overlap_consistency_model(v, obs, w_o, s_o, sigma=25.0, cmax=400.0, zmax=600.0, dip_deg=25.0):
    """Normalized cross-shot overlap of migrated point clouds, computed per shot pair (small blocks, no N x N matrix)."""
    tab = Tables(v); Z, X, NZv, NXv, cost = migrate(tab, obs); good = (cost < cmax) & (Z < zmax)
    Z, X, NZv, NXv, w, sh = Z[good], X[good], NZv[good], NXv[good], w_o[good], s_o[good]; cmin = np.cos(np.radians(dip_deg))
    idx = [np.nonzero(sh == a_)[0] for a_ in range(5)]
    def block(ia, ib):
        if len(ia) == 0 or len(ib) == 0: return 0.0
        d2 = (Z[ia][:, None] - Z[ib][None, :]) ** 2 + (X[ia][:, None] - X[ib][None, :]) ** 2
        dots = NZv[ia][:, None] * NZv[ib][None, :] + NXv[ia][:, None] * NXv[ib][None, :]
        return float((np.exp(-d2 / (2 * sigma ** 2)) * (dots >= cmin) * w[ia][:, None] * w[ib][None, :]).sum())
    M = np.array([[block(idx[a_], idx[b_]) if b_ >= a_ else 0.0 for b_ in range(5)] for a_ in range(5)])
    pairs = [(a_, b_) for a_ in range(5) for b_ in range(a_ + 1, 5) if M[a_, a_] > 0 and M[b_, b_] > 0]
    ov = np.mean([M[a_, b_] / np.sqrt(M[a_, a_] * M[b_, b_]) for a_, b_ in pairs]) if pairs else 0.0
    return 1.0 - ov, dict(points=int(good.sum()), overlap=round(float(ov), 4))

def gradient_descent(f, theta0, eps=0.03, max_step=0.05, iters=25, tol=2e-4, log=None, central=True):
    """Finite-difference gradient in log-velocity (central, or one-sided when central=False: half the evaluations),
    steepest descent with backtracking; stops when three consecutive steps improve by less than tol."""
    th = theta0.copy(); J, _ = f(th); stall = 0
    for it in range(iters):
        g = np.zeros_like(th)
        for k in range(len(th)):
            e = np.zeros_like(th); e[k] = eps
            g[k] = (f(th + e)[0] - f(th - e)[0]) / (2 * eps) if central else (f(th + e)[0] - J) / eps
        d = -g / (np.abs(g).max() + 1e-12) * max_step; step = 1.0; accepted = False
        for _ in range(4):
            Jn, info = f(th + step * d)
            if Jn < J: accepted = True; break
            step *= 0.5
        if log is not None: log.append(dict(iter=it, J=round(float(J), 5), grad=np.round(g, 4).tolist(), step=step if accepted else 0.0, knots=np.round(np.exp(th)).tolist()))
        if not accepted: break
        stall = stall + 1 if J - Jn < tol else 0; th, J = th + step * d, Jn
        if stall >= 3: break
    return th, J

def nelder_mead(f, x0, step=0.08, iters=120, tol=1e-4):
    n = len(x0); pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]; vals = [f(x) for x in pts]
    for it in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol * max(1e-9, abs(vals[0])): break
        c = np.mean(pts[:-1], 0); xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            if fe < fr: pts[-1], vals[-1] = xe, fe
            else: pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]: pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]; vals = [vals[0]] + [f(p) for p in pts[1:]]
    o = np.argmin(vals); return pts[o], vals[o]

# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', default='flatfault-a'); ap.add_argument('--model', type=int, default=0)
    ap.add_argument('--iters', type=int, default=8); ap.add_argument('--out', default=None); ap.add_argument('--init', default='flat', choices=['flat', 'homogeneous']); ap.add_argument('--vscale', type=float, default=1.0, help='scale the init velocity (wrong start test)'); ap.add_argument('--mode', default='search', choices=['search', 'tomo', 'gradient']); ap.add_argument('--variant', default='', choices=['', 'lateral'], help="'lateral': impose a +-10 percent lateral gradient on the top layer (v < 1700) to test 2D recovery"); ap.add_argument('--prior', type=float, default=0.2); ap.add_argument('--smooth-x', type=float, default=None, help='lateral smoothness weight (default: same as --smooth)'); ap.add_argument('--smooth', type=float, default=0.0, help='weight of sum (log v_k+1 - log v_k)^2: one reflector only fixes the RMS overburden velocity, smoothness picks the flattest profile'); a = ap.parse_args()
    import glob
    f = sorted(glob.glob(f'{ROOT}/{a.family}/train/*.npy'))[0]; allm = np.load(f, mmap_mode='r'); model = allm[a.model, 0].astype(np.float64)
    mean_mse = float(((np.asarray(allm[:, 0], np.float64) - np.asarray(allm[:, 0], np.float64).mean(0)) ** 2).mean())   # this family's dataset-mean MSE
    if a.variant == 'lateral': model = np.where(model <= model[0, 0] + 1, model * (1 + 0.2 * (np.arange(NX)[None, :] / (NX - 1) - 0.5)), model)   # +-10 % across the top layer
    cache = f'toy/slopes/sim_{a.family}_{a.model}{"_" + a.variant if a.variant else ""}.npz'
    if os.path.exists(cache): gathers = np.load(cache)['gathers']
    else:
        t0 = time.time(); gathers = simulate(model, [int(round(x / DX)) for x in SRC_X]); np.savez_compressed(cache, gathers=gathers); print(f'simulated in {time.time() - t0:.0f}s')
    out = a.out or f'toy/slopes/general_{a.family}_{a.model}_{a.init}{"" if a.vscale == 1 else "_x%.2f" % a.vscale}_{a.mode}.png'
    # 1. measurement side (generic) and flat-layer initialisation
    v1, tau = direct_wave_fit(gathers); pts = oriented_points(gathers, tau, v1)
    t0_bins = np.arange(0, 1.0, 0.004); v_bins = np.arange(1300, 5000, 25.0); t0c = 0.5 * (t0_bins[1:] + t0_bins[:-1]); vc = 0.5 * (v_bins[1:] + v_bins[:-1])
    V = vote(pts, t0_bins, v_bins); _, _, picks = pick_interfaces(V, t0c, vc, v1); layers = dix(picks, v1); vz0 = layers_to_vz(layers)
    v_init = np.tile(S.gauss1d(vz0, 3.0, 0)[:, None], (1, NX)) if a.init == 'flat' else np.full((NZ, NX), v1)
    v_init = v_init * a.vscale
    floor = ((1.0 / control_to_model(fit_control(model).reshape(NCZ, NCX)) - model) ** 2).mean() / mean_mse
    print(f'representation floor of the {NCZ}x{NCX} smooth control grid on this model: MSE/mean = {floor:.3f}', flush=True)
    # observations as (shot, receiver, t, p, w), per shot so the receiver index is known
    allobs = []
    for s_ in range(5):
        tt, hh, pp, ww = oriented_points(gathers[s_:s_ + 1], tau, v1, src_x=[SRC_X[s_]])
        rr = np.round((hh + SRC_X[s_]) / DX).astype(int); allobs.append((np.full(len(tt), s_), rr, tt, pp, ww))
    obs = tuple(np.concatenate([o[k] for o in allobs]) for k in range(5)); s_o, r_o, t_o, p_o, w_o = obs
    print(f'observations: {len(t_o)} oriented points; init v(z) from flat-layer pipeline, MSE/mean = {((v_init - model) ** 2).mean() / mean_mse:.3f}', flush=True)
    if a.mode == 'gradient':
        # 2''. finite-difference gradient descent on the overlap objective over a log-v(z) spline, with a prior that is
        # weak where migrated points illuminate a knot and strong where nothing does
        theta0 = np.log(np.array([[v_init[int(round(z / DX)), int(round(x / DX))] for x in KX] for z in KZ])).ravel()
        illum = illumination(theta0, obs, w_o, s_o); lam = a.prior / (illum + 0.05)
        print(f'illumination per knot (fraction of points at or below): {np.round(illum, 3)}; prior weights {np.round(lam, 2)}', flush=True)
        glog = []; evals = [0]
        def f(th):
            J, info = overlap_spline(th, obs, w_o, s_o); evals[0] += 1
            T2 = th.reshape(NK, NKX); smx = a.smooth if a.smooth_x is None else a.smooth_x
            sm = a.smooth * float(np.sum(np.diff(T2, axis=0) ** 2)) + (smx * float(np.sum(np.diff(T2, axis=1) ** 2)) if NKX > 1 else 0.0)
            return J + float(np.sum(lam * (th - theta0) ** 2)) + sm, info
        sweep = {sc: overlap_consistency_model(np.full((NZ, NX), v1 * sc), obs, w_o, s_o)[0] for sc in (0.85, 0.92, 1.0, 1.08, 1.15)}
        print('homogeneous sweep of 1-overlap vs v1 scale (the objective should bottom near the true overburden):', {k: round(v, 3) for k, v in sweep.items()}, flush=True)
        t1 = time.time(); J0, _ = f(theta0); th, J = gradient_descent(f, theta0, log=glog); v_cur = spline_to_model(th)
        print(f'gradient descent: {len(glog)} iterations, {evals[0]} evaluations in {time.time() - t1:.0f}s, objective {J0:.4f} -> {J:.4f}', flush=True)
        for g_ in glog: print(g_, flush=True)
        tht = np.log(np.array([[model[int(round(z / DX)), int(round(x / DX))] for x in KX] for z in KZ])).ravel(); Jt, infot = overlap_spline(tht, obs, w_o, s_o)
        print(f'overlap objective at the true v(z) sampled at the knots: {Jt:.4f} {infot}', flush=True)
        print('recovered knots (m/s):'); print(np.round(np.exp(th)).reshape(NK, NKX)); print('true velocity at knots (m/s):'); print(np.round(np.exp(tht)).reshape(NK, NKX)); print('illumination:'); print(np.round(illum, 2).reshape(NK, NKX))
        def vrms_to(vz, zmax):
            zc = (np.arange(NZ) + 0.5) * DX; m = zc <= zmax; dt = DX / vz[m]; return float(np.sqrt((vz[m] ** 2 * dt).sum() / dt.sum()))
        for xcol, name in [(10, 'x=100 m'), (35, 'x=350 m'), (60, 'x=600 m')]:
            print(f'RMS velocity to 385 m at {name}: recovered {vrms_to(v_cur[:, xcol], 385):.0f}, true {vrms_to(model[:, xcol], 385):.0f}, start {vrms_to(v_init[:, xcol], 385):.0f}', flush=True)
        history = [dict(iter=g_['iter'], consistency=g_['J']) for g_ in glog]; a.iters = 0; ctrl = fit_control(v_cur)
        res_extra = dict(knots_z=KZ.tolist(), knots_v=np.round(np.exp(th)).tolist(), true_knots_v=np.round(np.exp(tht)).tolist(), illumination=np.round(illum, 3).tolist())
    if a.mode == 'search':
        # 2'. direct multi-view-consistency search over a low-dimensional velocity (log-velocity on an NSZ x NSX grid)
        band = lambda vm: np.array([[vm[(i * NZ) // NSZ:((i + 1) * NZ) // NSZ, (j * NX) // NSX:((j + 1) * NX) // NSX].mean() for j in range(NSX)] for i in range(NSZ)])
        theta0 = np.log(band(v_init)).ravel()
        log = []
        objective = overlap_consistency                                   # time-residual consistency() is flat in velocity; overlap is not
        def f(th):
            J, info = objective(th, obs, w_o, s_o); J = J + PRIOR * float(np.mean((th - theta0) ** 2)); log.append((J, info)); return J
        t1 = time.time(); J0, info0 = objective(theta0, obs, w_o, s_o); print(f'consistency at init: {J0:.4f}, {info0}', flush=True)
        th, J = nelder_mead(f, theta0, iters=220); v_cur = search_to_model(th); print(f'search: {len(log)} evaluations in {time.time() - t1:.0f}s, consistency {J0:.4f} -> {J:.4f}', flush=True)
        tht = np.log(band(model)).ravel(); Jt, infot = objective(tht, obs, w_o, s_o)
        print(f'consistency with the true velocity in the same bands: {Jt:.4f} {infot}; its MSE/mean {((search_to_model(tht) - model) ** 2).mean() / mean_mse:.3f}', flush=True)
        history = [dict(iter=i, consistency=round(float(j), 4), **inf) for i, (j, inf) in enumerate(log)]
        print('recovered velocity grid (m/s):'); print(np.round(np.exp(th).reshape(NSZ, NSX)))
        print('true velocity in those bands (m/s):'); print(np.round(band(model)))
        a.iters = 0; ctrl = fit_control(v_cur)
    # 2. loop: migrate -> cluster (densify/prune) -> slowness update with backtracking -> new tables
    if a.mode == 'tomo': ctrl = fit_control(v_init); v_cur = 1.0 / control_to_model(ctrl.reshape(NCZ, NCX)); history = []
    tab = Tables(v_cur)
    for it in range(a.iters):
        t1 = time.time()
        radius = max(60.0 * 0.8 ** it, 12.0); cmax = max(400.0 * 0.7 ** it, 25.0)   # coarse-to-fine: a wrong velocity splits shots by tens of metres
        Z, X, NZv, NXv, cost = migrate(tab, obs); good = cost < cmax
        elems = cluster(Z[good], X[good], NZv[good], NXv[good], w_o[good], s_o[good], radius=radius); gidx = np.nonzero(good)[0]
        for e in elems: e['members'] = gidx[e['members']]
        multi = [e for e in elems if e['nshots'] >= 2]
        ctrl_new, rms = tomography_step(tab, multi, obs, ctrl); step = 1.0; accepted = False
        for _ in range(4):                                                      # backtracking on the slowness step
            ctrl_try = ctrl + step * (ctrl_new - ctrl); v_try = 1.0 / control_to_model(ctrl_try.reshape(NCZ, NCX)); tab_try = Tables(v_try)
            rms_try = residual_rms(tab_try, multi, obs)
            if rms_try < rms: accepted = True; break
            step *= 0.5
        mse = ((v_cur - model) ** 2).mean() / mean_mse
        history.append(dict(iter=it, radius=round(radius), elements=len(elems), multi_shot=len(multi), migrated_ok=int(good.sum()), rms_ms=round(rms * 1e3, 2), rms_after_ms=round(rms_try * 1e3, 2), step=step if accepted else 0.0, mse_over_mean=round(float(mse), 4)))
        print(history[-1], f'{time.time() - t1:.0f}s', flush=True)
        if accepted: ctrl, v_cur, tab = ctrl_try, v_try, tab_try
        else: break
    tab = Tables(v_cur); Z, X, NZv, NXv, cost = migrate(tab, obs); good = cost < 25.0
    mse_final = ((v_cur - model) ** 2).mean() / mean_mse
    res = dict(family=a.family, model=a.model, init=a.init, mode=a.mode, **(res_extra if a.mode == 'gradient' else {}), floor=round(float(floor), 4), n_obs=len(t_o), init_mse_over_mean=round(float(((v_init - model) ** 2).mean() / mean_mse), 4), dataset_mean_mse=round(mean_mse), final_mse_over_mean=round(float(mse_final), 4), history=history)
    print(json.dumps(res, indent=1)); json.dump(res, open(out.replace('.png', '.json'), 'w'), indent=1)
    # 3. figure
    fig, ax = plt.subplots(2, 3, figsize=(17, 9)); ext = [0, NX * DX, NZ * DX, 0]; vmin, vmax = model.min(), model.max()
    ax[0, 0].imshow(model, extent=ext, cmap='viridis', vmin=vmin, vmax=vmax); ax[0, 0].set_title('true model')
    ax[0, 1].imshow(v_init, extent=ext, cmap='viridis', vmin=vmin, vmax=vmax); ax[0, 1].set_title(f'init ({a.init}), MSE/mean {res["init_mse_over_mean"]:.3f}')
    ax[0, 2].imshow(v_cur, extent=ext, cmap='viridis', vmin=vmin, vmax=vmax); ax[0, 2].set_title(f'refined smooth velocity, MSE/mean {mse_final:.3f} (floor {floor:.3f})')
    ax[1, 0].imshow(model, extent=ext, cmap='gray', vmin=vmin, vmax=vmax); sc = ax[1, 0].scatter(X[good], Z[good], c=s_o[good], s=4, cmap='tab10'); ax[1, 0].set_title('migrated oriented points (colour = shot), final velocity')
    ax[1, 0].set_xlim(0, NX * DX); ax[1, 0].set_ylim(NZ * DX, 0)
    tab0 = Tables(v_init); Z0, X0, _, _, c0 = migrate(tab0, obs); g0 = c0 < 25.0
    ax[1, 1].imshow(model, extent=ext, cmap='gray', vmin=vmin, vmax=vmax); ax[1, 1].scatter(X0[g0], Z0[g0], c=s_o[g0], s=4, cmap='tab10'); ax[1, 1].set_title('migrated points, initial velocity'); ax[1, 1].set_xlim(0, NX * DX); ax[1, 1].set_ylim(NZ * DX, 0)
    key = 'consistency' if a.mode in ('search', 'gradient') else 'rms_ms'; ax[1, 2].plot([h[key] for h in history], 'b.-', label=key); ax[1, 2].set_xlabel('evaluation' if a.mode == 'search' else 'iteration'); ax[1, 2].legend(loc='upper right'); ax[1, 2].set_title({'search': 'multi-view consistency search', 'gradient': 'gradient descent on overlap + prior', 'tomo': 'tomography'}[a.mode])
    plt.tight_layout(); plt.savefig(out, dpi=110); print('wrote', out)

if __name__ == '__main__':
    main()

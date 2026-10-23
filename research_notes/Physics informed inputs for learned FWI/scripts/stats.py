"""Empirical geology statistics of local OpenFWI velocity shards (one shard per family)."""
import sys, glob, os
import numpy as np

ROOT = sys.argv[1]
FAMS = ["flatvel-a", "flatvel-b", "curvevel-a", "curvevel-b", "flatfault-a", "flatfault-b",
        "curvefault-a", "curvefault-b", "style-a", "style-b"]
TOL = 1.0  # m/s; piecewise-constant families have exact jumps


def best_shift(a, b, maxs=40):
    """vertical shift s (cells) minimising mean |a(z) - b(z+s)|."""
    best, bs = 1e18, 0
    n = len(a)
    for s in range(-maxs, maxs + 1):
        if s >= 0:
            x, y = a[: n - s], b[s:]
        else:
            x, y = a[-s:], b[: n + s]
        if len(x) < 20:
            continue
        e = np.mean(np.abs(x - y))
        if e < best:
            best, bs = e, s
    return bs, best


def q(a, ps=(5, 50, 95)):
    return " / ".join(f"{np.percentile(a, p):.3g}" for p in ps)


for fam in FAMS:
    files = sorted(glob.glob(os.path.join(ROOT, fam, "model*.npy")) + glob.glob(os.path.join(ROOT, fam, "vel*.npy")))
    if not files:
        print(fam, "no file"); continue
    v = np.load(files[0], allow_pickle=False, mmap_mode="r")
    v = np.asarray(v, dtype=np.float64)[:, 0]  # N, z, x
    N, NZ, NX = v.shape
    dz = np.diff(v, axis=1)
    dx = np.diff(v, axis=2)
    # per-model stats
    nuniq = np.array([len(np.unique(np.round(m))) for m in v])
    # layers along columns: jumps+1, max over columns
    jumps_col = (np.abs(dz) > TOL).sum(axis=1)  # N, x
    nlay_col = jumps_col.max(axis=1) + 1
    # velocity inversions: negative vertical jumps
    neg = (dz < -TOL)
    pos = (dz > TOL)
    frac_models_monotone = np.mean(~neg.reshape(N, -1).any(axis=1))
    frac_neg_jumps = neg.sum() / max(1, (neg | pos).sum())
    # jump ratio at interfaces (below/above)
    above = v[:, :-1, :]
    below = v[:, 1:, :]
    m = np.abs(dz) > TOL
    ratios = (below[m] / above[m])
    # lateral heterogeneity: mean over rows of std_x / mean_x
    lat = (v.std(axis=2) / v.mean(axis=2)).mean(axis=1)
    frac_lateral = np.mean((np.abs(dx) > TOL).reshape(N, -1).any(axis=1))
    # smoothness: fraction of vertical steps that are "small" (<TOL) vs mean abs gradient
    mean_grad = (np.abs(dz).mean(axis=(1, 2)) + np.abs(dx).mean(axis=(1, 2))) / 2
    print(f"== {fam}  file={os.path.basename(files[0])}  N={N} grid={NZ}x{NX}")
    print(f"  v range overall: {v.min():.0f}-{v.max():.0f} m/s; per-model min p5/50/95: {q(v.min(axis=(1,2)))}; max: {q(v.max(axis=(1,2)))}")
    print(f"  surface (row0) mean v p5/50/95: {q(v[:,0,:].mean(axis=1))}; bottom row: {q(v[:,-1,:].mean(axis=1))}")
    print(f"  distinct values/model p5/50/95: {q(nuniq)}  (min {nuniq.min()}, max {nuniq.max()})")
    print(f"  layers along a column (max over x) p5/50/95: {q(nlay_col)}  hist: {np.bincount(nlay_col)[1:12].tolist()}")
    print(f"  models with NO velocity inversion: {frac_models_monotone:.2%}; fraction of vertical jumps negative: {frac_neg_jumps:.2%}")
    if len(ratios):
        print(f"  interface ratio v_below/v_above p5/50/95: {q(ratios)}; |dv| at jumps p5/50/95: {q(np.abs(dz[m]))} m/s")
    print(f"  models with any lateral jump: {frac_lateral:.2%}; lateral CoV (mean std_x/mean_x) p5/50/95: {q(lat)}")
    print(f"  mean |grad| per cell p5/50/95: {q(mean_grad)} m/s; frac of vertical steps that are jumps>1m/s: {m.mean():.3f}")
    if "curve" in fam or "flat" in fam:
        # undulation of first interface: depth of first jump in each column
        first = np.full((N, NX), np.nan)
        for i in range(N):
            for x in range(NX):
                idx = np.nonzero(np.abs(dz[i, :, x]) > TOL)[0]
                if len(idx):
                    first[i, x] = idx[0] + 1
        und = np.nanmax(first, axis=1) - np.nanmin(first, axis=1)
        print(f"  first-interface depth range across x (cells, 10 m): p5/50/95: {q(und[~np.isnan(und)])}")
        # layer thickness along centre column
        th = []
        for i in range(N):
            idx = np.nonzero(np.abs(dz[i, :, NX // 2]) > TOL)[0] + 1
            b = np.concatenate([[0], idx, [NZ]])
            th += list(np.diff(b))
        th = np.array(th)
        print(f"  layer thickness centre column (cells, incl. truncated top/bottom) p5/50/95: {q(th)}")
    if "fault" in fam:
        shifts, errs = [], []
        angles = []
        for i in range(N):
            s, e = best_shift(v[i, :, 0], v[i, :, -1])
            shifts.append(abs(s)); errs.append(e)
            zz, xx = np.nonzero(np.abs(dx[i]) > TOL)
            if len(zz) > 10 and np.ptp(zz) > 10:
                A = np.vstack([zz, np.ones_like(zz)]).T
                coef, res, *_ = np.linalg.lstsq(A, xx.astype(float), rcond=None)
                rms = np.sqrt(np.mean((A @ coef - xx) ** 2))
                if rms < 1.5:  # one clean straight fault
                    angles.append(np.degrees(np.arctan2(1.0, abs(coef[0]))))  # angle from horizontal
        shifts = np.array(shifts)
        print(f"  |best vertical shift| left vs right column (cells) p5/50/95: {q(shifts)}; zero-shift models: {np.mean(shifts==0):.2%}")
        if angles:
            angles = np.array(angles)
            print(f"  straight-fault dip from horizontal (deg), {len(angles)} clean single-line models: p5/50/95 {q(angles)}")

"""Per-pixel physics channels from official OpenFWI shot gathers ("more information per pixel").

For one model's 5 gathers, everything here is training-free and takes seconds:
  0 v_geo        geometric velocity start: slopes -> closed-form hyperbola votes -> picks (multiples rejected) -> Dix
  1 reflect      migrated reflectivity: every oriented point (shot, receiver, time, slope) moved to the subsurface point
                 where its two-way time and slope match, using traveltime tables on v_geo; splatted with its weight
  2 dip          lateral component of the migrated reflector normal (sin of dip), weight-averaged per pixel
  3 agree        cross-shot agreement: fraction of the 5 shots that put a reflector within 15 m of the pixel
  4 illum        illumination: share of reflector weight at or below the pixel within +-100 m laterally (1 = the data
                 constrain this pixel's velocity, 0 = null space below the deepest reflector)
  5 grad         FWI adjoint gradient of the full-band misfit at v_geo (source wavelet estimated), normalised by its
                 99th percentile magnitude: where and in which direction the data want the start to change
Also stored for the raw-data baseline: gathers box-averaged to 5 x 100 x 35.

usage: uv run python -I toy/slopes/features.py --family flatvel-a --data DATA.npy --models 0-499 --out toy/slopes/feat [--threads 2]
Writes one .npz per model (resumable); pack.py-style stacking happens in train_channels.py.
"""
import argparse, glob, os, sys, time, numpy as np
os.environ.setdefault('EIK_H', '5.0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general as GEN
from fwi import geometric_init, Forward, source_x_from_data, estimate_source, ROOT, NZ, NX, DX
from slopes import direct_wave_fit, oriented_points, SRC_X, gauss1d
CH = ['v_geo', 'reflect', 'dip', 'agree', 'illum', 'grad']

def channels(obs, nthreads=2):
    t = {}; t0 = time.time()
    v_geo, v1, layers = geometric_init(obs); v_geo = np.clip(v_geo, 1500.0, 4500.0); t['geo'] = time.time() - t0   # Dix can overshoot; keep the simulation stable
    G = obs.astype(np.float64); _, tau = direct_wave_fit(G); allobs = []
    for s_ in range(5):
        tt, hh, pp, ww = oriented_points(G[s_:s_ + 1], tau, v1, src_x=[SRC_X[s_]])
        rr = np.round((hh + SRC_X[s_]) / DX).astype(int); allobs.append((np.full(len(tt), s_), rr, tt, pp, ww))
    obs_pts = tuple(np.concatenate([o[j] for o in allobs]) for j in range(5)); s_o, r_o, t_o, p_o, w_o = obs_pts; t['points'] = time.time() - t0
    tab = GEN.Tables(v_geo); Z, X, NZv, NXv, cost = GEN.migrate(tab, obs_pts); good = (cost < 100.0) & (Z < NZ * DX); t['migrate'] = time.time() - t0
    iz = np.clip(np.round(Z[good] / DX).astype(int), 0, NZ - 1); ix = np.clip(np.round(X[good] / DX).astype(int), 0, NX - 1); w = w_o[good]; sh = s_o[good]
    refl = np.zeros((NZ, NX)); np.add.at(refl, (iz, ix), w); refl = gauss1d(gauss1d(refl, 1.0, 0), 1.0, 1)
    dipw = np.zeros((NZ, NX)); np.add.at(dipw, (iz, ix), w * NXv[good] * np.sign(NZv[good] + 1e-12)); dipw = gauss1d(gauss1d(dipw, 1.0, 0), 1.0, 1)
    dip = dipw / (refl + 1e-3 * refl.max() + 1e-12)
    per_shot = np.zeros((5, NZ, NX))
    for s_ in range(5):
        m = sh == s_; a = np.zeros((NZ, NX)); np.add.at(a, (iz[m], ix[m]), 1.0); per_shot[s_] = gauss1d(gauss1d(a, 1.5, 0), 1.5, 1) > 0.05
    agree = per_shot.mean(0)
    col = np.zeros((NZ, NX)); np.add.at(col, (iz, ix), w)
    col = gauss1d(col, 10.0, 1) if True else col                                          # +-100 m lateral support (2 sigma)
    below = np.flip(np.cumsum(np.flip(col, 0), 0), 0); illum = below / (below.max(0, keepdims=True).clip(min=1e-9)); illum = np.where(below.max(0, keepdims=True) > 0, illum, 0.0)
    refl = refl / (refl.max() + 1e-12)
    fwd = Forward(source_x_from_data(obs), nthreads, margin=0); wav = estimate_source(fwd(v_geo, fwd.ricker), obs, fwd.ricker)
    _, g = fwd.gradient(v_geo, wav, obs); g = np.nan_to_num(g); g[:2] = 0; g = g / (np.percentile(np.abs(g), 99) + 1e-30); g = np.clip(g, -3, 3); t['grad'] = time.time() - t0
    ch = np.stack([(np.clip(v_geo, 1500, 4500) - 3000) / 1500, refl, dip, agree, illum, g]).astype(np.float32)
    raw = obs.reshape(5, 100, 10, 35, 2).mean(axis=(2, 4)).astype(np.float32)
    return ch, raw, t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); ap.add_argument('--data', required=True); ap.add_argument('--models', default='0-499')
    ap.add_argument('--out', default='toy/slopes/feat'); ap.add_argument('--threads', type=int, default=2); a = ap.parse_args()
    od = f'{a.out}/{a.family}'; os.makedirs(od, exist_ok=True)
    M = np.load(sorted(glob.glob(f'{ROOT}/{a.family}/train/*.npy'))[0], mmap_mode='r'); D = np.load(a.data, mmap_mode='r')
    lo, hi = (int(x) for x in a.models.split('-'))
    for k in range(lo, hi + 1):
        p = f'{od}/{k:03d}.npz'
        if os.path.exists(p): continue
        t0 = time.time()
        try: ch, raw, t = channels(np.array(D[k], np.float64), a.threads)
        except Exception as e: print(k, 'failed', repr(e), flush=True); continue
        np.savez_compressed(p, ch=ch, raw=raw, model=np.array(M[k, 0], np.float32))
        print(k, f'{time.time() - t0:.1f}s', {kk: round(vv, 1) for kk, vv in t.items()}, flush=True)

if __name__ == '__main__':
    main()

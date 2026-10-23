"""Refraction channels and a post-critical mask, added to the per-model channel files (features.py).

Why not head-wave picks: in this geometry the first arrival is the direct wave at every receiver and the head wave runs
within one wavelet of the post-critical reflection (refraction.py). A two-layer waveform fit (v1 above, v_r below a
tilted interface) was tested on 8 FlatFault-A models: interface depth comes out within about 20-50 m, but the best
v_r is biased low on strong contrasts (thin transition layers and faults make two layers too crude), so no single
point estimate is trusted. The channels carry the fit's whole landscape instead:

  6 refr_like  per pixel: likelihood that the first strong refractor passes through it, sum over candidate depths z0
               (dz fixed at its best value) of exp(-(J(z0) - Jmin)/T) x Gaussian(z - z_r(x; z0), 20 m), scaled to max 1
  7 refr_v     soft two-layer velocity: likelihood-weighted mean over z0 of (v1 above z_r(x; z0), best v_r(z0) below),
               scaled like v_geo ((v - 3000)/1500)
  8 postcrit   fraction of the migrated reflector weight at the pixel carried by rays that cannot have travelled below
               the refractor: oriented points with |p| >= 1 / v_r (v_r at the most likely depth), smoothed; 0 without
               reflectivity

usage: uv run python -I toy/slopes/features_refr.py --family flatfault-a --data DATA.npy --models 0-499 [--feat toy/slopes/feat]
Writes toy/slopes/feat_refr/<family>/<k>.npz with ch_refr (3, 70, 70) and the fit landscape. Resumable.
"""
import argparse, os, sys, time, numpy as np
os.environ.setdefault('EIK_H', '5.0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general as GEN
from refraction import RefractorFit
from slopes import direct_wave_fit, oriented_points, SRC_X, gauss1d, DX, NX, NZ

def refraction_channels(obs, v_geo, rf):
    G = obs.astype(np.float64); v1, tau = direct_wave_fit(G); L = rf.landscape(G, v1)
    Jz = L['J1'].min(1); T = max(0.02, 0.25 * (np.median(Jz) - Jz.min())); w = np.exp(-(Jz - Jz.min()) / T); w /= w.sum()
    vbest = L['vrs'][np.argmin(L['J1'], 1)]; x = np.arange(NX) * DX; z = np.arange(NZ) * DX
    like = np.zeros((NZ, NX)); vel = np.zeros((NZ, NX))
    for wi, z0, vb in zip(w, L['z0s'], vbest):
        zr = np.clip(z0 + L['dz'] * (x - 345.0) / 690.0, 30.0, 680.0)
        like += wi * np.exp(-0.5 * ((z[:, None] - zr[None, :]) / 20.0) ** 2); vel += wi * np.where(z[:, None] >= zr[None, :], vb, v1)
    like /= like.max() + 1e-12
    v_ref = float(vbest[int(np.argmax(w))])
    if v_geo is None:
        ch = np.stack([like, (np.clip(vel, 1500, 4500) - 3000) / 1500, np.zeros_like(like)]).astype(np.float32)
        return ch, dict(v1=v1, v_ref=v_ref, z_ml=float(L['z0s'][int(np.argmax(w))]), dz=L['dz'], Jmin=L['Jmin'], T=T, frac_turned=0.0), L
    allobs = []
    for s_ in range(5):
        tt, hh, pp, ww = oriented_points(G[s_:s_ + 1], tau, v1, src_x=[SRC_X[s_]])
        rr = np.round((hh + SRC_X[s_]) / DX).astype(int); allobs.append((np.full(len(tt), s_), rr, tt, pp, ww))
    obs_pts = tuple(np.concatenate([o[j] for o in allobs]) for j in range(5)); p_o, w_o = obs_pts[3], obs_pts[4]
    tab = GEN.Tables(v_geo); Z, X, _, _, cost = GEN.migrate(tab, obs_pts); good = (cost < 100.0) & (Z < NZ * DX)
    iz = np.clip(np.round(Z[good] / DX).astype(int), 0, NZ - 1); ix = np.clip(np.round(X[good] / DX).astype(int), 0, NX - 1)
    turned = (np.abs(p_o[good]) >= 1.0 / v_ref).astype(float); wg = w_o[good]
    num = np.zeros((NZ, NX)); den = np.zeros((NZ, NX)); np.add.at(num, (iz, ix), wg * turned); np.add.at(den, (iz, ix), wg)
    num = gauss1d(gauss1d(num, 2.0, 0), 2.0, 1); den = gauss1d(gauss1d(den, 2.0, 0), 2.0, 1)
    post = np.where(den > 1e-3 * (den.max() + 1e-30), num / (den + 1e-30), 0.0)
    ch = np.stack([like, (np.clip(vel, 1500, 4500) - 3000) / 1500, post]).astype(np.float32)
    info = dict(v1=v1, v_ref=v_ref, z_ml=float(L['z0s'][int(np.argmax(w))]), dz=L['dz'], Jmin=L['Jmin'], T=T, frac_turned=float((wg * turned).sum() / (wg.sum() + 1e-30)))
    return ch, info, L

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); ap.add_argument('--data', required=True); ap.add_argument('--models', default='0-499')
    ap.add_argument('--feat', default='toy/slopes/feat'); ap.add_argument('--out', default='toy/slopes/feat_refr'); ap.add_argument('--no-postcrit', action='store_true', help='skip the post-critical mask (stored as zeros); no base channels needed'); a = ap.parse_args()
    od = f'{a.out}/{a.family}'; os.makedirs(od, exist_ok=True); D = np.load(a.data, mmap_mode='r'); rf = RefractorFit()
    lo, hi = (int(x) for x in a.models.split('-'))
    for k in range(lo, hi + 1):
        p = f'{od}/{k:03d}.npz'
        if os.path.exists(p): continue
        t0 = time.time(); v_geo = None if a.no_postcrit else np.load(f'{a.feat}/{a.family}/{k:03d}.npz')['ch'][0].astype(np.float64) * 1500 + 3000
        try: ch, info, L = refraction_channels(np.array(D[k], np.float64), v_geo, rf)
        except Exception as e: print(k, 'failed', repr(e), flush=True); continue
        np.savez_compressed(p, ch_refr=ch, J1=L['J1'], J2=L['J2'], vrs=L['vrs'], z0s=L['z0s'], dzs=L['dzs'], **{kk: np.float32(vv) for kk, vv in info.items()})
        print(k, f'{time.time() - t0:.1f}s', {kk: round(vv, 3) for kk, vv in info.items()}, flush=True)

if __name__ == '__main__':
    main()

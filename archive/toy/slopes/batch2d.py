"""Batch of the 2D gradient version over moderate-contrast FlatFault-A models.

Selection: effective critical angle >= 42 degrees (velocity 40 m below the first >5 % interface versus the top layer)
and first interface >= 150 m deep; the N with the most lateral variation in the top 400 m. Each model gets the lateral
variant (+-10 % across the top layer) so there is lateral structure to recover, and starts homogeneous at vscale x v1.
Reports, per model: recovered vs true top-layer knots (left / middle / right), RMS overburden velocity to the first
interface at x = 100 / 350 / 600 m (start, recovered, true), and the homogeneous-sweep minimum.

usage: NKX=3 uv run python -I toy/slopes/batch2d.py [--n 20] [--vscale 1.1] [--iters 8] [--out toy/slopes/batch2d.json]
"""
import argparse, glob, json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general as G
from general import *

def select(M, n):
    rows = []
    for k in range(len(M)):
        m = M[k]; vtop = m[0, 0]; eff = []; depths = []
        for x in range(0, 70, 5):
            col = m[:, x]; j = np.nonzero(col > vtop * 1.05)[0]
            if len(j) == 0 or j[0] + 4 >= 70: eff = None; break
            eff.append(col[j[0] + 4] / vtop); depths.append(int(j[0]) * 10)
        if eff is None: continue
        ang = np.degrees(np.arcsin(1 / max(eff))); lat = (m[:40].std(1) / m[:40].mean(1)).mean()
        if ang >= 42 and min(depths) >= 150: rows.append((k, ang, lat))
    rows.sort(key=lambda r: -r[2]); return rows[:n]

def first_interface_depth(col):
    j = np.nonzero(col > col[0] * 1.05)[0]; return int(j[0]) * DX if len(j) else (NZ - 1) * DX

def vrms_to(vz, zmax):
    zc = (np.arange(NZ) + 0.5) * DX; m = zc <= zmax; dt = DX / vz[m]; return float(np.sqrt((vz[m] ** 2 * dt).sum() / dt.sum()))

def run_one(model, k, vscale, iters, smooth, smooth_x, prior):
    cache = f'toy/slopes/sim_flatfault-a_{k}_lateral.npz'
    if os.path.exists(cache): gathers = np.load(cache)['gathers']
    else: gathers = simulate(model, [int(round(x / DX)) for x in SRC_X]); np.savez_compressed(cache, gathers=gathers)
    v1, tau = direct_wave_fit(gathers); allobs = []
    for s_ in range(5):
        tt, hh, pp, ww = oriented_points(gathers[s_:s_ + 1], tau, v1, src_x=[SRC_X[s_]])
        rr = np.round((hh + SRC_X[s_]) / DX).astype(int); allobs.append((np.full(len(tt), s_), rr, tt, pp, ww))
    obs = tuple(np.concatenate([o[j] for o in allobs]) for j in range(5)); s_o, r_o, t_o, p_o, w_o = obs
    v_init = np.full((NZ, NX), v1 * vscale)
    theta0 = np.log(np.array([[v_init[int(round(z / DX)), int(round(x / DX))] for x in KX] for z in KZ])).ravel()
    illum = illumination(theta0, obs, w_o, s_o); lam = prior / (illum + 0.05)
    sweep = {sc: overlap_consistency_model(np.full((NZ, NX), v1 * sc), obs, w_o, s_o)[0] for sc in (0.85, 0.92, 1.0, 1.08, 1.15)}
    def f(th):
        J, info = overlap_spline(th, obs, w_o, s_o); T2 = th.reshape(NK, NKX)
        sm = smooth * float(np.sum(np.diff(T2, axis=0) ** 2)) + smooth_x * float(np.sum(np.diff(T2, axis=1) ** 2))
        return J + float(np.sum(lam * (th - theta0) ** 2)) + sm, info
    glog = []; th, J = gradient_descent(f, theta0, iters=iters, log=glog, central=False); v_rec = spline_to_model(th)
    tht = np.log(np.array([[model[int(round(z / DX)), int(round(x / DX))] for x in KX] for z in KZ])).ravel()
    top_true = np.exp(tht).reshape(NK, NKX)[0]; top_rec = np.exp(th).reshape(NK, NKX)[0]; top_start = np.exp(theta0).reshape(NK, NKX)[0]
    rms = {}
    for xcol in (10, 35, 60):
        zi = first_interface_depth(model[:, xcol])
        rms[int(xcol * DX)] = dict(z_interface=zi, true=round(vrms_to(model[:, xcol], zi)), start=round(vrms_to(v_init[:, xcol], zi)), recovered=round(vrms_to(v_rec[:, xcol], zi)))
    return dict(model=k, v1=round(v1), n_obs=len(t_o), iterations=len(glog), sweep_argmin=min(sweep, key=sweep.get), sweep={str(a): round(float(b), 3) for a, b in sweep.items()},
                top_true=np.round(top_true).tolist(), top_start=np.round(top_start).tolist(), top_recovered=np.round(top_rec).tolist(),
                top_relerr_start=np.round(top_start / top_true - 1, 3).tolist(), top_relerr_recovered=np.round(top_rec / top_true - 1, 3).tolist(),
                lateral_trend_true=round(float(top_true[-1] / top_true[0] - 1), 3), lateral_trend_recovered=round(float(top_rec[-1] / top_rec[0] - 1), 3), rms=rms)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=20); ap.add_argument('--vscale', type=float, default=1.1); ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--smooth', type=float, default=1.0); ap.add_argument('--smooth-x', type=float, default=0.1); ap.add_argument('--prior', type=float, default=0.2); ap.add_argument('--out', default='toy/slopes/batch2d.json'); a = ap.parse_args()
    assert NKX == 3, 'run with NKX=3'
    f = sorted(glob.glob(f'{ROOT}/flatfault-a/train/*.npy'))[0]; M = np.load(f, mmap_mode='r')[:, 0].astype(np.float64)
    picks = select(M, a.n); print('models:', [(k, round(ang), round(lat, 3)) for k, ang, lat in picks], flush=True); rows = []; t0 = time.time()
    for k, ang, lat in picks:
        model = M[k].copy(); model = np.where(model <= model[0, 0] + 1, model * (1 + 0.2 * (np.arange(NX)[None, :] / (NX - 1) - 0.5)), model)
        t1 = time.time(); r = run_one(model, k, a.vscale, a.iters, a.smooth, a.smooth_x, a.prior); r['critical_angle'] = round(float(ang)); r['seconds'] = round(time.time() - t1); rows.append(r)
        print(k, dict(top_true=r['top_true'], top_rec=r['top_recovered'], err_start=r['top_relerr_start'], err_rec=r['top_relerr_recovered'], trend=(r['lateral_trend_true'], r['lateral_trend_recovered']), sweep_argmin=r['sweep_argmin'], s=r['seconds']), f'{time.time() - t0:.0f}s', flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1)
    es = np.array([x['top_relerr_start'] for x in rows]); er = np.array([x['top_relerr_recovered'] for x in rows])
    tt = np.array([x['lateral_trend_true'] for x in rows]); tr = np.array([x['lateral_trend_recovered'] for x in rows])
    rms_err_start = np.array([[x['rms'][p]['start'] / x['rms'][p]['true'] - 1 for p in ('100', '350', '600')] for x in rows]) if rows and '100' in rows[0]['rms'] else np.array([[x['rms'][p]['start'] / x['rms'][p]['true'] - 1 for p in (100, 350, 600)] for x in rows])
    rms_err_rec = np.array([[x['rms'][p]['recovered'] / x['rms'][p]['true'] - 1 for p in (100, 350, 600)] for x in rows])
    summary = dict(n=len(rows), top_knot_abs_relerr_start_mean=round(float(np.abs(es).mean()), 4), top_knot_abs_relerr_recovered_mean=round(float(np.abs(er).mean()), 4),
                   top_knot_abs_relerr_recovered_median=round(float(np.median(np.abs(er))), 4), models_improved=int((np.abs(er).mean(1) < np.abs(es).mean(1)).sum()),
                   lateral_trend_true_mean=round(float(tt.mean()), 3), lateral_trend_recovered_mean=round(float(tr.mean()), 3), trend_sign_correct=int((np.sign(tr) == np.sign(tt)).sum()),
                   rms_overburden_abs_relerr_start_mean=round(float(np.abs(rms_err_start).mean()), 4), rms_overburden_abs_relerr_recovered_mean=round(float(np.abs(rms_err_rec).mean()), 4),
                   sweep_argmin_at_1=int(sum(x['sweep_argmin'] == 1.0 for x in rows)), seconds_total=round(time.time() - t0))
    print(json.dumps(summary, indent=1)); json.dump(dict(summary=summary, rows=rows), open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()

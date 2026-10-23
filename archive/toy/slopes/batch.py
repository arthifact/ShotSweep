"""Run the slope pipeline on many FlatVel-A models with their official gathers and report MSE / dataset-mean MSE.

usage: uv run python -I toy/slopes/batch.py DATA.npy MODEL.npy [--start 452] [--n 48] [--out toy/slopes/batch.json]
Default range = the last 48 models of the train file, the validation split the octree-Gaussian artifact used (head B: 0.36).
"""
import argparse, json, sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slopes import direct_wave_fit, oriented_points, vote, pick_interfaces, dix, layers_to_vz, avo_velocity, MEAN_MSE, NX

def run_one(gathers, model):
    v1, tau = direct_wave_fit(gathers)
    t0_bins = np.arange(0, 1.0, 0.004); v_bins = np.arange(1300, 5000, 25.0)
    t0c = 0.5 * (t0_bins[1:] + t0_bins[:-1]); vc = 0.5 * (v_bins[1:] + v_bins[:-1])
    points = oriented_points(gathers, tau, v1); V = vote(points, t0_bins, v_bins)
    _, _, picks = pick_interfaces(V, t0c, vc, v1)
    layers = dix(picks, v1); crit = avo_velocity(gathers, tau, layers)
    if crit is not None: layers[-1] = (layers[-1][0], layers[-1][1], crit[0])
    vz = layers_to_vz(layers); pred = np.tile(vz[:, None], (1, NX))
    return float(((pred - model) ** 2).mean()), len(picks), v1, crit is not None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('model'); ap.add_argument('--start', type=int, default=452)
    ap.add_argument('--n', type=int, default=48); ap.add_argument('--out', default='toy/slopes/batch.json'); a = ap.parse_args()
    G = np.load(a.data, mmap_mode='r'); M = np.load(a.model, mmap_mode='r'); rows = []; t0 = time.time()
    for k in range(a.start, a.start + a.n):
        mse, npk, v1, used_crit = run_one(np.array(G[k], np.float64), np.array(M[k, 0], np.float64))
        rows.append(dict(model=k, ratio=round(mse / MEAN_MSE, 4), n_picks=npk, v1=round(v1), critical=used_crit))
        print(k, rows[-1], f'{time.time() - t0:.0f}s', flush=True)
    r = np.array([x['ratio'] for x in rows])
    summary = dict(n=len(r), mean_ratio=round(float(r.mean()), 4), median_ratio=round(float(np.median(r)), 4),
                   frac_below_0p1=round(float((r < 0.1).mean()), 3), frac_above_0p3=round(float((r > 0.3).mean()), 3), worst=sorted(rows, key=lambda x: -x['ratio'])[:5])
    print(json.dumps(summary, indent=1)); json.dump(dict(summary=summary, rows=rows), open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()

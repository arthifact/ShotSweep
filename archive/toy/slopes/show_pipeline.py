"""Show the whole pipeline on one model, live from the official gathers: 1 input, 2 processing, 3 what training receives.

usage: uv run python -I toy/slopes/show_pipeline.py --family flatvel-a --data DATA.npy --model 5
Writes toy/slopes/pipeline_<family>_<model>.png
"""
import argparse, glob, os, sys, numpy as np
os.environ.setdefault('EIK_H', '5.0')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general as GEN
from slopes import direct_wave_fit, oriented_points, vote, pick_interfaces, dix, SRC_X, DX, DT, NX, NZ
from fwi import geometric_init, ROOT
from features import channels
from features_refr import refraction_channels
from refraction import RefractorFit

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); ap.add_argument('--data', required=True); ap.add_argument('--model', type=int, default=5); a = ap.parse_args()
    M = np.load(sorted(glob.glob(f'{ROOT}/{a.family}/train/*.npy'))[0], mmap_mode='r'); obs = np.array(np.load(a.data, mmap_mode='r')[a.model], np.float64); true = np.array(M[a.model, 0], np.float64)
    # ---- processing, step by step (the same calls the channel builders make)
    v1, tau = direct_wave_fit(obs)
    pts_shot = [oriented_points(obs[s:s + 1], tau, v1, src_x=[SRC_X[s]]) for s in range(5)]
    pts = tuple(np.concatenate([p[j] for p in pts_shot]) for j in range(4))
    t0_bins = np.arange(0, 1.0, 0.004); v_bins = np.arange(1300, 5000, 25.0); t0c = 0.5 * (t0_bins[1:] + t0_bins[:-1]); vc = 0.5 * (v_bins[1:] + v_bins[:-1])
    V = vote(pts, t0_bins, v_bins); _, _, picks = pick_interfaces(V, t0c, vc, v1)
    v_geo, _, layers = geometric_init(obs); v_geo = np.clip(v_geo, 1500, 4500)
    allobs = []
    for s in range(5):
        tt, hh, pp, ww = pts_shot[s]; allobs.append((np.full(len(tt), s), np.round((hh + SRC_X[s]) / DX).astype(int), tt, pp, ww))
    obs_pts = tuple(np.concatenate([o[j] for o in allobs]) for j in range(5))
    Z, X, _, _, cost = GEN.migrate(GEN.Tables(v_geo), obs_pts); good = (cost < 100.0) & (Z < NZ * DX)
    ch, raw, _ = channels(obs, nthreads=8)
    chr_, info, L = refraction_channels(obs, None, RefractorFit())
    # ---- figure
    fig = plt.figure(figsize=(22, 15)); gs = fig.add_gridspec(4, 8, hspace=0.45, wspace=0.25)
    ext_g = [0, NX * DX, 1.0, 0]; ext_m = [0, NX * DX, NZ * DX, 0]; lo, hi = true.min(), true.max()
    def gather_ax(ax, s, title):
        g = obs[s] * (1 + np.arange(1000) * DT * 3)[:, None]; c = np.abs(g).max() * 0.15
        ax.imshow(g, aspect='auto', cmap='gray', extent=ext_g, vmin=-c, vmax=c); ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    # row 1: input
    for s in range(5): gather_ax(fig.add_subplot(gs[0, s]), s, f'shot {s + 1}, source at {SRC_X[s]:.0f} m')
    ax = fig.add_subplot(gs[0, 6]); ax.imshow(true, extent=ext_m, cmap='viridis', vmin=lo, vmax=hi); ax.set_title('label: true velocity\n(used only to train and to score)', fontsize=9); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 7]); ax.axis('off')
    ax.text(0, 0.95, f'{a.family}, model {a.model}\n\ninput = 5 shot gathers\n1000 samples x 70 receivers each\n(1 s at 1 ms, 10 m spacing)\n\nsurface velocity from direct wave:\n{v1:.0f} m/s (true {true[0, 35]:.0f})', fontsize=10, va='top')
    # row 2: processing
    ax = fig.add_subplot(gs[1, 0:2]); gather_ax(ax, 2, '2a oriented points on shot 3 (colour = local slope)')
    tt, hh, pp, ww = pts_shot[2]; ax.scatter(hh + SRC_X[2], tt + tau, c=pp * 1e3, s=4, cmap='coolwarm', vmin=-0.6, vmax=0.6)
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(np.log1p(V / (V.max() + 1e-12) * 100), aspect='auto', cmap='viridis', extent=[v_bins[0], v_bins[-1], t0_bins[-1], 0])
    ax.plot([p[1] for p in picks], [p[0] for p in picks], 'r+', ms=12, mew=2); ax.set_title('2b closed-form votes (t0, v_rms)\nred = picked interfaces', fontsize=9); ax.set_xlabel('v_rms (m/s)', fontsize=7); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, 3]); zc = (np.arange(NZ) + 0.5) * DX
    ax.step(true[:, 35], zc, 'k', label='true (x = 350 m)'); ax.step(v_geo[:, 35], zc, 'r', label='geometric start'); ax.invert_yaxis(); ax.legend(fontsize=7); ax.set_title('2c layers by Dix: geometric start', fontsize=9); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, 4]); ax.imshow(true, extent=ext_m, cmap='gray', vmin=lo, vmax=hi); ax.scatter(X[good], Z[good], c=obs_pts[0][good], s=3, cmap='tab10'); ax.set_xlim(0, 700); ax.set_ylim(700, 0)
    ax.set_title('2d points moved underground\n(colour = shot), on the true model', fontsize=9); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, 5:7]); Jz = L['J1']
    ax.imshow(Jz, aspect='auto', cmap='viridis_r', extent=[L['vrs'][0], L['vrs'][-1], L['z0s'][-1], L['z0s'][0]]); ax.set_xscale('log')
    ax.set_title('2e two-layer waveform fit, misfit over\n(refractor depth, velocity below), low = good', fontsize=9); ax.set_xlabel('velocity below refractor (m/s)', fontsize=7); ax.set_ylabel('refractor depth (m)', fontsize=7); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, 7]); ax.axis('off')
    ax.text(0, 0.95, f'processing, no learning:\n\n{len(pts[0])} oriented points\n{len(picks)} interfaces picked\n{int(good.sum())} points placed underground\n{Jz.size} two-layer models simulated\n\nabout 15 s per model', fontsize=10, va='top')
    # rows 3-4: what training receives
    names = ['geometric start', 'reflectivity', 'dip', 'shot agreement', 'illumination', 'FWI gradient', 'refractor likelihood', 'two-layer velocity']
    imgs = list(ch) + list(chr_[:2])
    for i, (im, n) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4]); kw = dict(cmap='RdBu', vmin=-1, vmax=1) if i in (2, 5) else dict(cmap='viridis')
        if i in (0, 7): kw = dict(cmap='viridis', vmin=(lo - 3000) / 1500, vmax=(hi - 3000) / 1500)
        ax.imshow(im, extent=ext_m, **kw); ax.set_title(f'3 input ch{i}: {n}', fontsize=9); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[2:4, 4:6]); strip = np.concatenate([raw[s] * (1 + np.arange(100) * 0.02)[:, None] for s in range(5)], 1); c = np.abs(strip).max() * 0.3
    ax.imshow(strip, aspect='auto', cmap='gray', vmin=-c, vmax=c); ax.set_title('3 input raw: the 5 gathers compressed\nto 100 x 35 each (then PCA 128 inside training)', fontsize=9); ax.set_xticks([]); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[2:4, 6:8]); ax.imshow(true, extent=ext_m, cmap='viridis', vmin=lo, vmax=hi); ax.set_title('3 target: true velocity\n(the network predicts a correction to ch0)', fontsize=9); ax.tick_params(labelsize=7)
    fig.text(0.01, 0.89, '1 INPUT', fontsize=14, weight='bold', rotation=90); fig.text(0.01, 0.66, '2 PROCESSING', fontsize=14, weight='bold', rotation=90); fig.text(0.01, 0.25, '3 SENT TO TRAINING', fontsize=14, weight='bold', rotation=90)
    out = f'openfwi/results/figures/pipeline_{a.family}_{a.model}.png'; plt.savefig(out, dpi=75, bbox_inches='tight'); print('wrote', out)

if __name__ == '__main__':
    main()

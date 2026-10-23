"""Step 2, processing: per-shot wave-equation images for every model, the same recipe for every family.

1. Calibrate one source wavelet for the simulator from 20 training models (2 per family)  -> work/source_wavelet.npy
2. Mean depth trend of velocity over all training models                                    -> work/depth_trend.npy
3. For every model: surface velocity from the direct wave; three backgrounds (slow, medium, fast: surface velocity +
   0.6 / 1.0 / 1.4 x depth trend); for each background and each shot, the reverse-time-migration image of what that
   background does not explain (direct wave muted, illumination corrected, Laplacian filtered). Each model's 15 images
   are scaled together by their 99th percentile magnitude, so relative amplitudes across shots and backgrounds are kept.
                                                                                            -> work/inputs/<family>.npz
   images (500, 3, 5, 70, 70) float16, surface_velocity (500,)

usage: uv run python -I openfwi/2_process.py [--families all] [--batch 8]
"""
import argparse, json, os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'physics'))
import gpu, background as BG
WORK = os.path.join(HERE, 'work')

def load_manifest():
    m = json.load(open(os.path.join(HERE, 'data', 'manifest.json'))); fam = m['families']
    G = {f: np.load(os.path.join(HERE, v['gathers']), mmap_mode='r') for f, v in fam.items()}
    M = {f: np.load(os.path.join(HERE, v['models']), mmap_mode='r') for f, v in fam.items()}
    return m, G, M

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--families', default='all'); ap.add_argument('--batch', type=int, default=8); a = ap.parse_args()
    man, G, M = load_manifest(); fams = list(G) if a.families == 'all' else a.families.split(','); ntr = man['n_train']
    os.makedirs(os.path.join(WORK, 'inputs'), exist_ok=True)
    wp, tp = os.path.join(WORK, 'source_wavelet.npy'), os.path.join(WORK, 'depth_trend.npy')
    if not os.path.exists(wp):
        np.save(wp, gpu.calibrate([np.array(M[f][k, 0], float) for f in G for k in (0, 1)], [np.array(G[f][k], float) for f in G for k in (0, 1)]))
    if not os.path.exists(tp):
        np.save(tp, BG.depth_trend(np.concatenate([np.asarray(M[f][:ntr, 0], np.float32) for f in G])))
    sim = gpu.Sim(np.load(wp)); trend = np.load(tp); nb = len(BG.RHOS)
    for f in fams:
        out = os.path.join(WORK, 'inputs', f'{f}.npz')
        if os.path.exists(out): print(f, 'done already', flush=True); continue
        n = man['n_total']; imgs = np.zeros((n, nb, 5, 70, 70), np.float16); v1s = np.zeros(n, np.float32); t0 = time.time()
        for i in range(0, n, a.batch):
            ks = list(range(i, min(i + a.batch, n))); obs = np.asarray(G[f][ks], np.float64)
            v1 = np.array([BG.surface_velocity(o) for o in obs]); bgs = np.concatenate([BG.backgrounds(v, trend) for v in v1])      # (len(ks) * nb, 70, 70)
            img, _ = sim.images(bgs, np.repeat(obs, nb, 0), v_mute=np.repeat(v1, nb))
            img = img.reshape(len(ks), nb, 5, 70, 70); img[..., :2, :] = 0
            sc = np.percentile(np.abs(img).reshape(len(ks), -1), 99, axis=1)[:, None, None, None, None] + 1e-30
            imgs[ks] = np.clip(img / sc, -4, 4).astype(np.float16); v1s[ks] = v1
            if i % 100 == 0: print(f'{f}: {i + len(ks)}/{n} models, {time.time() - t0:.0f}s', flush=True)
        np.savez(out, images=imgs, surface_velocity=v1s); print(f, 'saved', f'{time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()

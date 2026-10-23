"""Show what the network receives for one model: the truth, the three backgrounds, the stacked image per background, one
shot's image, and the disagreement between shots. Reads work/inputs/<family>.npz (run 2_process.py first).

usage: uv run python -I openfwi/tools/show_images.py --family curvefault-b --model 42
Writes results/figures/inputs_<family>_<model>.png
"""
import argparse, json, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'physics'))
import background as BG

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); ap.add_argument('--model', type=int, required=True); a = ap.parse_args()
    man = json.load(open(os.path.join(ROOT, 'data', 'manifest.json')))['families'][a.family]
    true = np.load(os.path.join(ROOT, man['models']), mmap_mode='r')[a.model, 0]
    inp = np.load(os.path.join(ROOT, 'work', 'inputs', f'{a.family}.npz')); img = inp['images'][a.model].astype(np.float32)   # (3, 5, 70, 70)
    bgs = BG.backgrounds(float(inp['surface_velocity'][a.model]), np.load(os.path.join(ROOT, 'work', 'depth_trend.npy')))
    fig, ax = plt.subplots(2, 5, figsize=(17, 7)); lo, hi = true.min(), true.max()
    ax[0, 0].imshow(true, cmap='viridis'); ax[0, 0].set_title('truth (target)')
    for i in range(3): ax[0, i + 1].imshow(bgs[i], cmap='viridis', vmin=lo, vmax=hi); ax[0, i + 1].set_title(f'background rho = {BG.RHOS[i]}')
    ax[0, 4].imshow(img[1, 0], cmap='gray', vmin=-2, vmax=2); ax[0, 4].set_title('shot 1 image, medium background')
    for i in range(3): ax[1, i].imshow(img[i].mean(0), cmap='gray', vmin=-1.5, vmax=1.5); ax[1, i].set_title(f'mean of 5 shot images, rho = {BG.RHOS[i]}')
    ax[1, 3].imshow(img[1].std(0), cmap='magma', vmin=0, vmax=1.5); ax[1, 3].set_title('disagreement between shots, medium')
    ax[1, 4].imshow(img[2].std(0) - img[0].std(0), cmap='RdBu', vmin=-1, vmax=1); ax[1, 4].set_title('disagreement: fast minus slow background')
    for x in ax.ravel(): x.axis('off')
    plt.suptitle(f'{a.family} model {a.model}: network inputs besides the raw gathers'); plt.tight_layout()
    out = os.path.join(ROOT, 'results', 'figures', f'inputs_{a.family}_{a.model}.png'); plt.savefig(out, dpi=80); print('wrote', out)

if __name__ == '__main__':
    main()

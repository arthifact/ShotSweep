"""Step 1, data: get and check the OpenFWI files, write data/manifest.json.

For each of the ten 2D families we use one official file pair: the first training file of velocity models and its
matching file of shot gathers (500 models, 5 shots x 1000 samples x 70 receivers). Models 0-451 are the training pool,
models 452-499 the held-out test set.

usage: uv run python -I openfwi/1_data.py            (check files, write the manifest)
       uv run python -I openfwi/1_data.py --download (first download any missing model or gather file, about 7 GB in
                                                      total, from the public OpenFWI Google Drive folders)
"""
import argparse, json, os, re, sys, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
N_TRAIN, N_TOTAL = 452, 500
# family, OpenFWI Google Drive folder, velocity-model file, gather file
FAMILIES = [
    ('flatvel-a', '1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb', 'model1.npy', 'data1.npy'),
    ('flatvel-b', '1XnQjke4vG8DpHvfpoCIKQaG0D7Jsi_8O', 'model1.npy', 'data1.npy'),
    ('curvevel-a', '1NGXnVG0gUFHfDcUvJxfozCgiI4WwquVk', 'model1.npy', 'data1.npy'),
    ('curvevel-b', '1d1N6DIISC0fBMwPxnOYOhFP3p2ZtCPmf', 'model1.npy', 'data1.npy'),
    ('flatfault-a', '1jOB6R_zewuFj5wZam7nDP7GixQnbnRLR', 'vel2_1_0.npy', 'seis2_1_0.npy'),
    ('flatfault-b', '1mfu5DCMTiEzrYoXbMfq9ILAM0RJ2SqNX', 'vel6_1_0.npy', 'seis6_1_0.npy'),
    ('curvefault-a', '1vqUHJ-iRwp3ozL-e4HhKGpdO0e7NQZE1', 'vel2_1_0.npy', 'seis2_1_0.npy'),
    ('curvefault-b', '1QTDno4_qed94mqhxJRzqWC19iYfANRqD', 'vel6_1_0.npy', 'seis6_1_0.npy'),
    ('style-a', '1CQceUL5ITTHV-PRqbIzkyZwCDdUk0XkU', 'model1.npy', 'data1.npy'),
    ('style-b', '1JbyhbEq8TIrJbH-FZJje9WOejktE0f0Y', 'model1.npy', 'data1.npy'),
]

def drive_listing(folder):
    html = urllib.request.urlopen(f'https://drive.google.com/embeddedfolderview?id={folder}').read().decode()
    ids = re.findall(r'id="entry-([^"]+)"', html); names = re.findall(r'flip-entry-title">([^<]*)', html)
    return dict(zip(names, ids))

def download(folder, name, out, subfolder):
    top = drive_listing(folder); sub = drive_listing(top[subfolder]) if subfolder in top else top
    url = f'https://drive.usercontent.google.com/download?id={sub[name]}&export=download&confirm=t'
    with urllib.request.urlopen(url) as r, open(out + '.part', 'wb') as f:
        while chunk := r.read(1 << 22): f.write(chunk)
    os.rename(out + '.part', out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--download', action='store_true'); a = ap.parse_args()
    manifest = {'n_train': N_TRAIN, 'n_total': N_TOTAL, 'families': {}}
    for fam, folder, mfile, gfile in FAMILIES:
        d = os.path.join(DATA, fam); os.makedirs(d, exist_ok=True); gp, mp = os.path.join(d, gfile), os.path.join(d, mfile)
        if not os.path.exists(mp) and a.download: print('downloading', fam, mfile, flush=True); download(folder, mfile, mp, 'model')
        if not os.path.exists(gp) and a.download: print('downloading', fam, gfile, flush=True); download(folder, gfile, gp, 'data')
        ok = os.path.exists(gp) and os.path.exists(mp)
        if ok:
            g = np.load(gp, mmap_mode='r'); m = np.load(mp, mmap_mode='r')
            ok = g.shape == (N_TOTAL, 5, 1000, 70) and m.shape[0] == N_TOTAL and m.shape[-2:] == (70, 70)
            vr = (float(np.min(m[:50])), float(np.max(m[:50])))
        manifest['families'][fam] = dict(gathers=os.path.relpath(gp, HERE), models=os.path.relpath(mp, HERE), ok=bool(ok), velocity_range_first50=vr if ok else None)
        print(f'{fam:13s} {"ok" if ok else "MISSING OR BAD"}', flush=True)
    json.dump(manifest, open(os.path.join(DATA, 'manifest.json'), 'w'), indent=1)
    if not all(v['ok'] for v in manifest['families'].values()): sys.exit(1)

if __name__ == '__main__':
    main()

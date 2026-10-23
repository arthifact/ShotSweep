"""Step 1, data: get and check the OpenFWI files, write data/manifest.json.

For each of the ten 2D families we use one official file pair: the first training file of velocity models and its
matching file of shot gathers (500 models, 5 shots x 1000 samples x 70 receivers). Models 0-399 are training,
400-451 validation, and 452-499 the held-out test set. This is a within-file pilot split.

usage: uv run python -I openfwi/1_data.py            (check files, write the manifest)
       uv run python -I openfwi/1_data.py --download (first download any missing model or gather file, about 7 GB in
                                                      total, from the public OpenFWI Google Drive folders)
"""
import argparse, hashlib, json, os, re, sys, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
N_TRAIN, N_VAL, N_TOTAL = 400, 52, 500
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
    html = urllib.request.urlopen(f'https://drive.google.com/embeddedfolderview?id={folder}', timeout=60).read().decode()
    ids = re.findall(r'id="entry-([^"]+)"', html); names = re.findall(r'flip-entry-title">([^<]*)', html)
    return dict(zip(names, ids))

def download(folder, name, out, subfolder):
    top = drive_listing(folder); sub = drive_listing(top[subfolder]) if subfolder in top else top
    url = f'https://drive.usercontent.google.com/download?id={sub[name]}&export=download&confirm=t'
    part = out + '.part'
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(part, 'wb') as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        # A quota/confirmation HTML page must never become a cached .npy file.
        np.load(part, mmap_mode='r', allow_pickle=False)
        os.replace(part, out)
    finally:
        if os.path.exists(part):
            os.remove(part)


def check_pair(gp, mp, n_total=N_TOTAL):
    """Check every sample in bounded chunks; return content hashes for reproducibility."""
    g = np.load(gp, mmap_mode='r', allow_pickle=False)
    m = np.load(mp, mmap_mode='r', allow_pickle=False)
    if g.shape != (n_total, 5, 1000, 70) or m.shape != (n_total, 1, 70, 70):
        raise ValueError(f'Unexpected shapes: gathers={g.shape}, models={m.shape}')
    hashes = {}
    for name, array in (('gathers', g), ('models', m)):
        for i in range(0, n_total, 8):
            chunk = np.asarray(array[i:i + 8])
            if not np.isfinite(chunk).all():
                raise ValueError(f'{name} contains nonfinite values at samples {i}:{i + 8}')
            if name == 'gathers' and np.any(np.max(np.abs(chunk), axis=(1, 2, 3)) == 0):
                raise ValueError(f'Zero recording at samples {i}:{i + 8}')
            if name == 'models' and (chunk.min() < 1500 or chunk.max() > 4500):
                raise ValueError('Velocity labels must be within 1500–4500 m/s')
        with open(gp if name == 'gathers' else mp, 'rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--families', default='all', help='Comma-separated families to check/download')
    ap.add_argument('--data-root', default=HERE, help='Directory containing data/ and work/')
    a = ap.parse_args()
    selected = {f[0] for f in FAMILIES} if a.families == 'all' else set(a.families.split(','))
    if not selected or selected - {f[0] for f in FAMILIES}:
        ap.error('Unknown or empty family selection')
    root = os.path.abspath(a.data_root)
    manifest = {'schema_version': 2, 'n_train': N_TRAIN, 'n_val': N_VAL,
                'n_total': N_TOTAL, 'protocol': 'within-file-pilot-v2', 'families': {}}
    for fam, folder, mfile, gfile in FAMILIES:
        if fam not in selected:
            continue
        d = os.path.join(root, 'data', fam); os.makedirs(d, exist_ok=True); gp, mp = os.path.join(d, gfile), os.path.join(d, mfile)
        if not os.path.exists(mp) and a.download: print('downloading', fam, mfile, flush=True); download(folder, mfile, mp, 'model')
        if not os.path.exists(gp) and a.download: print('downloading', fam, gfile, flush=True); download(folder, gfile, gp, 'data')
        try:
            hashes = check_pair(gp, mp)
        except (OSError, ValueError) as exc:
            raise SystemExit(f'{fam}: {exc}. Existing manifest was not replaced.') from exc
        manifest['families'][fam] = dict(gathers=os.path.relpath(gp, root), models=os.path.relpath(mp, root), ok=True, sha256=hashes)
        print(f'{fam:13s} ok', flush=True)
    path = os.path.join(root, 'data', 'manifest.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(path + '.tmp', path)

if __name__ == '__main__':
    main()

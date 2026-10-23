"""Create an isolated small dataset, preserving train/validation/test membership.

Use --source-root for real OpenFWI data, or --synthetic for a software rehearsal.
Synthetic results measure software behavior, never inversion accuracy on OpenFWI.
The output directory must not exist; original data and work caches are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'physics'))


def subset_indices(manifest, n_train, n_val, n_test):
    nt, nv, n = (int(manifest[k]) for k in ('n_train', 'n_val', 'n_total'))
    if min(n_train, n_val, n_test) < 1 or nt < 2 or nv < 1 or nt + nv >= n:
        raise ValueError('Nonempty, explicit training, validation, and test partitions are required')
    if n_train > nt or n_val > nv or n_test > n - nt - nv:
        raise ValueError('Requested pilot exceeds a source partition')
    return np.r_[np.arange(n_train), np.arange(nt, nt + n_val),
                 np.arange(nt + nv, nt + nv + n_test)]


def prepare(output, source_root=None, families='all', synthetic=False,
            n_train=8, n_val=2, n_test=2, device='cpu'):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output must be a new directory: {output}')
    if (source_root is None) == (not synthetic):
        raise ValueError('Choose exactly one of source_root and synthetic')
    if n_train < 2 or min(n_val, n_test) < 1:
        raise ValueError('Need at least two training samples and nonempty validation/test sets')
    n = n_train + n_val + n_test
    manifest = dict(schema_version=2, n_train=n_train, n_val=n_val, n_total=n,
                    protocol='synthetic-software-rehearsal' if synthetic else 'openfwi-small-pilot',
                    synthetic=synthetic, families={})
    if synthetic:
        import torch
        import gpu
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS was requested but is unavailable')
        gpu.DEV = torch.device(device)
        models = np.empty((n, 1, 70, 70), dtype=np.float32)
        for i in range(n):
            models[i, 0] = 1800 + 80 * (i % 6)
            models[i, 0, 18 + (i % 5):] = 2800 + 80 * (i % 4)
        gathers = gpu.Sim().gathers(models[:, 0], chunk=1)
        arrays = [('synthetic-layered', gathers, models, list(range(n)))]
    else:
        source_root = Path(source_root).resolve()
        source = json.loads((source_root / 'data/manifest.json').read_text())
        ids = subset_indices(source, n_train, n_val, n_test)
        names = list(source['families']) if families == 'all' else families.split(',')
        if not names or len(set(names)) != len(names) or set(names) - source['families'].keys():
            raise ValueError('Unknown or duplicate family')
        arrays = []
        for name in names:
            paths = source['families'][name]
            g = np.load(source_root / paths['gathers'], mmap_mode='r', allow_pickle=False)[ids]
            m = np.load(source_root / paths['models'], mmap_mode='r', allow_pickle=False)[ids]
            original_ids = paths.get('source_indices', list(range(source['n_total'])))
            if len(original_ids) != source['n_total']:
                raise ValueError(f'{name}: source_indices does not cover every source sample')
            arrays.append((name, g, m, [original_ids[i] for i in ids]))
        manifest['source_protocol'] = source.get('protocol')
        with (source_root / 'data/manifest.json').open('rb') as stream:
            manifest['source_manifest_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    # Validate all selected records before creating an output directory.
    for name, g, m, ids in arrays:
        if g.shape != (n, 5, 1000, 70) or m.shape != (n, 1, 70, 70):
            raise ValueError(f'{name}: unexpected shapes')
        if not np.isfinite(g).all() or not np.isfinite(m).all():
            raise ValueError(f'{name}: nonfinite input')
        if m.min() < 1500 or m.max() > 4500 or np.any(np.max(np.abs(g), axis=(1, 2, 3)) == 0):
            raise ValueError(f'{name}: invalid velocity range or empty recording')
    output.mkdir(parents=True)
    for name, g, m, ids in arrays:
        folder = output / 'data' / name
        folder.mkdir(parents=True)
        np.save(folder / 'gathers.npy', np.asarray(g, np.float32))
        np.save(folder / 'models.npy', np.asarray(m, np.float32))
        hashes = {}
        for key in ('gathers', 'models'):
            with (folder / f'{key}.npy').open('rb') as stream:
                hashes[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['families'][name] = dict(gathers=f'data/{name}/gathers.npy',
                                         models=f'data/{name}/models.npy',
                                         source_indices=ids, sha256=hashes, ok=True)
    (output / 'data/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-root', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', default='all')
    parser.add_argument('--n-train', type=int, default=8)
    parser.add_argument('--n-val', type=int, default=2)
    parser.add_argument('--n-test', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    manifest = prepare(args.output, args.source_root, args.families, args.synthetic,
                       args.n_train, args.n_val, args.n_test, args.device)
    print(json.dumps(dict(output=str(args.output.resolve()), protocol=manifest['protocol'],
                          families=list(manifest['families']), n_train=args.n_train,
                          n_val=args.n_val, n_test=args.n_test), indent=2))


if __name__ == '__main__':
    main()

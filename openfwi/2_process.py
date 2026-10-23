"""Build wave-equation images using training-only calibration and validated caches.

Usage: python -I openfwi/2_process.py [--families all] [--batch 1]
       [--data-root DIRECTORY] [--max-memory-gb 2]
DIRECTORY contains data/manifest.json and receives work/. Existing caches are
reused only when their data, split, physics implementation, and calibration
fingerprints match. Family outputs are written atomically after all models pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'physics'))
import gpu
import background as BG
import constants

CACHE_SCHEMA = 2


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npy(path, values):
    _atomic_write(path, lambda stream: np.save(stream, values, allow_pickle=False))


def atomic_npz(path, **values):
    _atomic_write(path, lambda stream: np.savez(stream, **values))


def atomic_json(path, value):
    _atomic_write(path, lambda stream: stream.write(json.dumps(value, sort_keys=True, indent=2).encode()))


def load_manifest(root=HERE):
    root = Path(root)
    with open(root / 'data' / 'manifest.json') as stream:
        manifest = json.load(stream)
    ntr = manifest['n_train']
    nval = manifest.get('n_val', 0)
    if ntr < 1 or nval < 1 or ntr + nval >= manifest['n_total']:
        raise ValueError('manifest must define nonempty train, validation, and test partitions; run 1_data.py')
    gathers = {f: np.load(root / entry['gathers'], mmap_mode='r', allow_pickle=False)
               for f, entry in manifest['families'].items()}
    models = {f: np.load(root / entry['models'], mmap_mode='r', allow_pickle=False)
              for f, entry in manifest['families'].items()}
    for family in gathers:
        if len(gathers[family]) != manifest['n_total'] or len(models[family]) != manifest['n_total']:
            raise ValueError(f'{family}: data length does not match manifest n_total')
    return manifest, gathers, models


def calibration_valid(work, expected_fingerprint):
    work = Path(work)
    try:
        with open(work / 'calibration_cache.json') as stream:
            metadata = json.load(stream)
        return (metadata['fingerprint'] == expected_fingerprint
                and metadata['wavelet_sha256'] == file_digest(work / 'source_wavelet.npy')
                and metadata['trend_sha256'] == file_digest(work / 'depth_trend.npy'))
    except (OSError, ValueError, KeyError):
        return False


def inputs_valid(path, expected_fingerprint, n_total):
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache['cache_provenance'].item()))
            return (metadata['fingerprint'] == expected_fingerprint
                    and cache['images'].shape == (n_total, len(BG.RHOS), 5, 70, 70)
                    and cache['surface_velocity'].shape == (n_total,)
                    and cache['surface_confidence'].shape == (n_total,)
                    and np.isfinite(cache['images']).all()
                    and np.isfinite(cache['surface_velocity']).all()
                    and np.isfinite(cache['surface_confidence']).all())
    except (OSError, ValueError, KeyError, EOFError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--families', default='all')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--data-root', type=Path, default=HERE)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-memory-gb', type=float, default=2.0,
                        help='estimated imaging tensor-storage budget; runtime overhead is additional')
    args = parser.parse_args()
    if args.batch < 1 or args.max_memory_gb <= 0 or args.threads < 1:
        parser.error('--batch, --threads and --max-memory-gb must be positive')
    gpu.DEV = gpu.resolve_device(args.device)
    gpu.torch.set_num_threads(args.threads)
    root = args.data_root.resolve()
    work = root / 'work'
    manifest, gathers, models = load_manifest(root)
    families = list(gathers) if args.families == 'all' else args.families.split(',')
    if not set(families) <= set(gathers):
        parser.error('unknown family requested')
    ntr = manifest['n_train']
    (work / 'inputs').mkdir(parents=True, exist_ok=True)
    implementation = {Path(path).name: file_digest(path)
                      for path in (__file__, gpu.__file__, BG.__file__, constants.__file__)}
    data_identity = {
        family: {key: file_digest(root / entry[key]) for key in ('gathers', 'models')}
        for family, entry in manifest['families'].items()
    }
    provenance = {
        'schema': CACHE_SCHEMA, 'implementation': implementation,
        'manifest': manifest, 'data': data_identity,
        'training_indices': list(range(ntr)), 'calibration_indices': list(range(min(2, ntr))),
        'rhos': list(BG.RHOS),
    }
    calibration_fingerprint = fingerprint(provenance)
    wavelet_path, trend_path = work / 'source_wavelet.npy', work / 'depth_trend.npy'
    if not calibration_valid(work, calibration_fingerprint):
        print('Calibrating source and depth trend from training indices only; old caches are invalid.', flush=True)
        indices = range(min(2, ntr))
        wavelet = gpu.calibrate([np.array(models[f][k, 0], float) for f in gathers for k in indices],
                               [np.array(gathers[f][k], float) for f in gathers for k in indices])
        trend = BG.depth_trend(np.concatenate([np.asarray(models[f][:ntr, 0], np.float32) for f in gathers]))
        if not np.isfinite(wavelet).all() or not np.any(wavelet) or not np.isfinite(trend).all():
            raise ValueError('calibration produced invalid source wavelet or depth trend')
        atomic_npy(wavelet_path, wavelet)
        atomic_npy(trend_path, trend)
        atomic_json(work / 'calibration_cache.json', {
            'fingerprint': calibration_fingerprint, 'provenance': provenance,
            'wavelet_sha256': file_digest(wavelet_path), 'trend_sha256': file_digest(trend_path),
        })
    with open(work / 'calibration_cache.json') as stream:
        calibration_metadata = json.load(stream)
    sim = gpu.Sim(np.load(wavelet_path, allow_pickle=False))
    trend = np.load(trend_path, allow_pickle=False)
    nb = len(BG.RHOS)
    per_sample_bytes = gpu.imaging_memory_bytes(nb)
    max_batch = int(args.max_memory_gb * 1024 ** 3 // per_sample_bytes)
    if max_batch < 1:
        parser.error(f'memory budget cannot fit one model; estimated tensor storage is {per_sample_bytes / 1024 ** 3:.2f} GiB')
    batch_size = min(args.batch, max_batch)
    print(f'Processing batch {batch_size}; estimated imaging tensor storage {batch_size * per_sample_bytes / 1024 ** 3:.2f} GiB.', flush=True)
    n = manifest['n_total']
    for family in families:
        out = work / 'inputs' / f'{family}.npz'
        input_provenance = {
            'schema': CACHE_SCHEMA, 'calibration_fingerprint': calibration_fingerprint,
            'calibration_wavelet_sha256': calibration_metadata['wavelet_sha256'],
            'calibration_trend_sha256': calibration_metadata['trend_sha256'],
            'family': family, 'data': data_identity[family], 'n_total': n,
            'n_train': ntr, 'n_val': manifest['n_val'],
        }
        input_provenance['fingerprint'] = fingerprint(input_provenance)
        if inputs_valid(out, input_provenance['fingerprint'], n):
            print(f'{family}: matching validated cache already exists', flush=True)
            continue
        imgs = np.zeros((n, nb, 5, 70, 70), np.float16)
        surface = np.zeros(n, np.float32)
        confidence = np.zeros(n, np.float32)
        fit_rmse = np.zeros(n, np.float32)
        refraction = np.zeros(n, bool)
        started = time.time()
        for i in range(0, n, batch_size):
            ks = list(range(i, min(i + batch_size, n)))
            obs = np.asarray(gathers[family][ks], np.float64)
            estimates = []
            for k, observed in zip(ks, obs):
                try:
                    estimates.append(BG.estimate_surface_velocity(observed))
                except ValueError as error:
                    raise ValueError(f'{family} model {k}: {error}; no replacement family cache saved') from error
            v1 = np.array([estimate['velocity'] for estimate in estimates])
            bgs = np.concatenate([BG.backgrounds(v, trend) for v in v1])
            img, _ = sim.images(bgs, np.repeat(obs, nb, 0), v_mute=np.repeat(v1, nb))
            img = img.reshape(len(ks), nb, 5, 70, 70)
            img[..., :2, :] = 0
            scale = np.percentile(np.abs(img).reshape(len(ks), -1), 99, axis=1)[:, None, None, None, None]
            if not np.isfinite(img).all() or np.any(scale <= 0):
                raise ValueError(f'{family} models {ks}: nonfinite or zero-energy images; no replacement cache saved')
            imgs[ks] = np.clip(img / scale, -4, 4).astype(np.float16)
            surface[ks] = v1
            confidence[ks] = [estimate['confidence'] for estimate in estimates]
            fit_rmse[ks] = [estimate['fit_rmse_ms'] for estimate in estimates]
            refraction[ks] = [estimate['possible_refraction'] for estimate in estimates]
            print(f'{family}: {i + len(ks)}/{n} models, {time.time() - started:.0f}s', flush=True)
        atomic_npz(out, images=imgs, surface_velocity=surface, surface_confidence=confidence,
                   surface_fit_rmse_ms=fit_rmse, surface_possible_refraction=refraction,
                   cache_provenance=np.array(json.dumps(input_provenance, sort_keys=True)))
        print(f'{family}: saved validated cache in {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()

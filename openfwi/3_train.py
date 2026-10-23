"""Step 3: train with a separate validation partition and resumable checkpoints.

The l1ssim objective is L1 plus 0.5 times one minus mean SSIM at three
resolutions. It is deliberately not the conventional product MS-SSIM.
SSIM sees [0, 1]; pixel losses see [-1, 1]. Left-right augmentation is disabled
because the middle acquisition source is not mirror symmetric.

Example: python openfwi/3_train.py --model ours --steps 8000 --batch 64
Pilot: add --limit-per-family 8 --val-limit-per-family 4 --steps 10 --batch 2
       --val-every 5 --save-every 5 --experiment pilot --device cpu
Resume: --resume <work/checkpoints/run.pt>; saved training settings are restored.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'physics'))
from metrics import torch_ssim


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'Unsupported device: {name}')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable in this PyTorch installation')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable in this PyTorch installation')
    return device


DEV = resolve_device()


def read_manifest(data_root=None):
    root = Path(data_root or HERE)
    with (root / 'data' / 'manifest.json').open() as stream:
        manifest = json.load(stream)
    if not all(k in manifest for k in ('n_train', 'n_val', 'n_total')):
        raise ValueError('Manifest must explicitly define n_train, n_val, n_total; regenerate data/manifest.json')
    ntr, nv, nt = (manifest[k] for k in ('n_train', 'n_val', 'n_total'))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (ntr, nv, nt)) or min(ntr, nv) < 1 or ntr + nv >= nt:
        raise ValueError('Manifest requires nonempty, disjoint train, validation and test partitions')
    if not manifest.get('families'):
        raise ValueError('Manifest contains no families')
    return manifest


def _finite(array, label):
    if not np.isfinite(array).all():
        raise ValueError(f'{label} contains NaN or infinite values')


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_data_files(manifest, data_root=None, families='all', require_hashes=True):
    """Verify the manifest's complete NPY file identities before using a run."""
    root = Path(data_root or HERE)
    selected = list(manifest['families']) if families == 'all' else families.split(',')
    for family in selected:
        if family not in manifest['families']:
            raise ValueError(f'Unknown family: {family}')
        entry = manifest['families'][family]
        identities = entry.get('sha256')
        if identities is None and not require_hashes:
            continue  # Small programmatic fixtures may omit complete-file identities.
        if not isinstance(identities, dict) or any(key not in identities for key in ('gathers', 'models')):
            raise ValueError(f'{family}: manifest requires gathers/models SHA-256 identities; rerun 1_data.py')
        for key in ('gathers', 'models'):
            if identities[key] != _file_digest(root / entry[key]):
                raise ValueError(f'{family}: {key} file no longer matches the manifest SHA-256 identity')


def _calibration_metadata(root, manifest):
    """Fail closed on legacy/stale physics inputs, including a changed split."""
    try:
        with (root / 'work' / 'calibration_cache.json').open() as stream:
            metadata = json.load(stream)
        provenance = metadata['provenance']
        expected = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        valid = (metadata['fingerprint'] == expected and provenance['schema'] == 2
                 and manifest_fingerprint(provenance['manifest']) == manifest_fingerprint(manifest)
                 and provenance['training_indices'] == list(range(manifest['n_train']))
                 and metadata['wavelet_sha256'] == _file_digest(root / 'work' / 'source_wavelet.npy')
                 and metadata['trend_sha256'] == _file_digest(root / 'work' / 'depth_trend.npy'))
        for filename, digest in provenance['implementation'].items():
            path = Path(HERE) / filename if filename == '2_process.py' else Path(HERE) / 'physics' / filename
            valid = valid and digest == _file_digest(path)
        for family, identities in provenance['data'].items():
            for key, digest in identities.items():
                valid = valid and digest == _file_digest(root / manifest['families'][family][key])
        if not valid:
            raise ValueError('stale calibration')
        return metadata
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError('Missing or stale physics calibration; rerun 2_process.py for this manifest') from error


def load_data(split, families='all', images=True, data_root=None, limit_per_family=None):
    """Load train/val/test, never using validation or test data to fit a scale.

    Partitions are [0,n_train), [n_train,n_train+n_val), and the remainder.
    Pilot limits select the first examples *within* each partition.
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown split: {split}')
    if limit_per_family is not None and limit_per_family < 1:
        raise ValueError('limit_per_family must be positive')
    root = Path(data_root or HERE)
    man = read_manifest(root)
    requested = list(man['families']) if families == 'all' else families.split(',')
    if not requested or len(requested) != len(set(requested)) or any(f not in man['families'] for f in requested):
        raise ValueError(f'Unknown or repeated families: {families}')
    ntr, nv, total = man['n_train'], man['n_val'], man['n_total']
    start, stop = {'train': (0, ntr), 'val': (ntr, ntr + nv), 'test': (ntr + nv, total)}[split]
    if limit_per_family is not None:
        stop = min(stop, start + limit_per_family)
    idx = np.arange(start, stop)
    trend = None
    if images:
        calibration = _calibration_metadata(root, man)
        trend = np.asarray(np.load(root / 'work' / 'depth_trend.npy'), np.float32)
        if trend.shape != (70,):
            raise ValueError('depth_trend.npy must have shape (70,)')
        _finite(trend, 'Depth trend')
    out = dict(gathers=[], images=[], vref=[], y=[], family=[], index=[])
    for family in requested:
        entry = man['families'][family]
        raw_g = np.load(root / entry['gathers'], mmap_mode='r')
        raw_y = np.load(root / entry['models'], mmap_mode='r')
        if raw_g.shape[0] < total or raw_g.shape[1:] != (5, 1000, 70):
            raise ValueError(f'{family}: gathers must cover n_total and have shape (N,5,1000,70)')
        if raw_y.shape[0] < total or raw_y.shape[1:] != (1, 70, 70):
            raise ValueError(f'{family}: models must cover n_total and have shape (N,1,70,70)')
        g = np.asarray(raw_g[idx], np.float32)
        y = np.asarray(raw_y[idx, 0], np.float32)
        _finite(g, f'{family} {split} gathers')
        _finite(y, f'{family} {split} models')
        if np.any(y < 1500 - 1e-3) or np.any(y > 4500 + 1e-3):
            raise ValueError(f'{family}: velocities must be in [1500,4500] m/s')
        out['gathers'].append((np.sign(g) * np.log1p(np.abs(g))).astype(np.float16))
        out['y'].append((y - 3000) / 1500)
        if images:
            with np.load(root / 'work' / 'inputs' / f'{family}.npz') as inp:
                try:
                    metadata = json.loads(str(inp['cache_provenance'].item()))
                    expected = dict(metadata)
                    fingerprint = expected.pop('fingerprint')
                    valid = (metadata['schema'] == 2 and metadata['family'] == family
                             and all(metadata[key] == man[key] for key in ('n_train', 'n_val', 'n_total'))
                             and metadata['calibration_fingerprint'] == calibration['fingerprint']
                             and metadata['calibration_wavelet_sha256'] == calibration['wavelet_sha256']
                             and metadata['calibration_trend_sha256'] == calibration['trend_sha256']
                             and fingerprint == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
                    if not valid:
                        raise ValueError('stale inputs')
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError(f'{family}: missing or stale input provenance; rerun 2_process.py') from error
                raw_im, raw_v1 = inp['images'], inp['surface_velocity']
                if raw_im.shape != (total, 3, 5, 70, 70) or raw_v1.shape != (total,):
                    raise ValueError(f'{family}: processed input shapes do not match the manifest')
                im, v1 = raw_im[idx], np.asarray(raw_v1[idx], np.float32)
            _finite(im, f'{family} {split} images')
            _finite(v1, f'{family} {split} surface velocity')
            out['images'].append(im)
            vr = np.broadcast_to(v1[:, None, None] + trend[None, :, None], (len(idx), 70, 70))
            out['vref'].append(np.clip((vr - 3000) / 1500, -1, 1).astype(np.float32))
        else:
            out['images'].append(np.zeros((len(idx), 1, 1, 1, 1), np.float16))
            out['vref'].append(np.zeros((len(idx), 70, 70), np.float32))
        out['family'].extend([family] * len(idx))
        out['index'].extend(idx.tolist())
    return {key: np.array(value) if key in ('family', 'index') else np.concatenate(value) for key, value in out.items()}


def batch(d, ids, scale, mirror=None, drop=None, device=None):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Gather normalization scale must be finite and positive')
    if mirror is not None and np.any(mirror):
        raise ValueError('Naive mirroring is invalid for the central source; regenerate mirrored acquisitions first')
    g = torch.from_numpy(d['gathers'][ids].astype(np.float32) / scale)
    im = torch.from_numpy(d['images'][ids].astype(np.float32))
    vr = torch.from_numpy(d['vref'][ids].astype(np.float32))
    y = torch.from_numpy(d['y'][ids].astype(np.float32))
    if drop is not None:
        drop = np.asarray(drop)
        if drop.shape != (len(ids),) or not np.isin(drop, (0, 1)).all():
            raise ValueError('Modality dropout mask must contain one 0/1 value per sample')
        im *= torch.tensor(1.0 - drop, dtype=torch.float32)[:, None, None, None, None]
    return tuple(t.to(device or DEV) for t in (g, im, vr, y))


# ---------------------------------------------------------------- networks
def cb(i, o, k=(3, 3), s=(1, 1), p=(1, 1)): return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
def db(i, o, k, s=1, p=0): return nn.Sequential(nn.ConvTranspose2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))

class InvEncoder(nn.Module):
    """InversionNet encoder: (B, 5, 1000, 70) -> (B, 512, 1, 1)."""
    def __init__(self, d=(32, 64, 128, 256, 512)):
        super().__init__(); d1, d2, d3, d4, d5 = d
        self.net = nn.Sequential(cb(5, d1, (7, 1), (2, 1), (3, 0)), cb(d1, d2, (3, 1), (2, 1), (1, 0)), cb(d2, d2, (3, 1), (1, 1), (1, 0)),
                                 cb(d2, d2, (3, 1), (2, 1), (1, 0)), cb(d2, d2, (3, 1), (1, 1), (1, 0)), cb(d2, d3, (3, 1), (2, 1), (1, 0)),
                                 cb(d3, d3, (3, 1), (1, 1), (1, 0)), cb(d3, d3, s=(2, 2)), cb(d3, d3), cb(d3, d4, s=(2, 2)), cb(d4, d4),
                                 cb(d4, d4, s=(2, 2)), cb(d4, d4), cb(d4, d5, (8, 9), (1, 1), (0, 0)))
    def forward(self, x): return self.net(x)

class InvDecoder(nn.Module):
    """InversionNet decoder: (B, 512, 1, 1) -> (B, 70, 70) in [-1, 1]."""
    def __init__(self, d=(32, 64, 128, 256, 512)):
        super().__init__(); d1, d2, d3, d4, d5 = d
        self.net = nn.Sequential(db(d5, d5, 5), cb(d5, d5), db(d5, d4, 4, 2, 1), cb(d4, d4), db(d4, d3, 4, 2, 1), cb(d3, d3),
                                 db(d3, d2, 4, 2, 1), cb(d2, d2), db(d2, d1, 4, 2, 1), cb(d1, d1))
        self.head = nn.Sequential(nn.Conv2d(d1, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.Tanh())
    def forward(self, z): return self.head(self.net(z)[:, :, 5:-5, 5:-5])[:, 0]

class InversionNet(nn.Module):
    def __init__(self): super().__init__(); self.enc = InvEncoder(); self.dec = InvDecoder()
    def forward(self, g, im=None, vr=None): return self.dec(self.enc(g))

class Ours(nn.Module):
    def __init__(self, k=3, use_raw=True, w=(64, 128, 256, 512), raw_only=False):
        super().__init__(); self.k = k; self.use_raw = use_raw; self.raw_only = raw_only
        self.enc = InvEncoder() if use_raw else None
        self.shot = nn.Sequential(cb(k, 32), cb(32, 32))                                   # shared by all five shots
        self.d0 = nn.Sequential(cb(65, w[0]), cb(w[0], w[0]))                              # 70 x 70: mean + std over shots + background
        self.d1 = nn.Sequential(cb(w[0], w[1], s=(2, 2)), cb(w[1], w[1]))                  # 35
        self.d2 = nn.Sequential(cb(w[1], w[2], s=(2, 2)), cb(w[2], w[2]))                  # 18
        self.d3 = nn.Sequential(cb(w[2], w[3], s=(2, 2)), cb(w[3], w[3]))                  # 9
        self.mid = cb(w[3] + (512 if use_raw else 0), w[3])
        self.u2 = nn.Sequential(cb(w[3] + w[2], w[2]), cb(w[2], w[2])); self.u1 = nn.Sequential(cb(w[2] + w[1], w[1]), cb(w[1], w[1]))
        self.u0 = nn.Sequential(cb(w[1] + w[0], w[0]), cb(w[0], w[0])); self.head = nn.Sequential(nn.Conv2d(w[0], 1, 1), nn.Tanh())
    def forward(self, g, im, vr):
        if self.raw_only:
            im = g.new_zeros((g.shape[0], 3, 5, 70, 70))
            vr = g.new_zeros((g.shape[0], 70, 70))
        B = im.shape[0]; im = im if self.k == 3 else im[:, 1:2]                            # im: (B, K, 5, 70, 70)
        f = self.shot(im.permute(0, 2, 1, 3, 4).reshape(B * 5, self.k, 70, 70)).reshape(B, 5, 32, 70, 70)
        x0 = self.d0(torch.cat([f.mean(1), f.std(1), vr[:, None]], 1)); x1 = self.d1(x0); x2 = self.d2(x1); x3 = self.d3(x2)
        if self.use_raw: x3 = torch.cat([x3, self.enc(g).expand(-1, -1, x3.shape[2], x3.shape[3])], 1)
        x = self.mid(x3)
        x = self.u2(torch.cat([F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False), x2], 1))
        x = self.u1(torch.cat([F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False), x1], 1))
        x = self.u0(torch.cat([F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False), x0], 1))
        return self.head(x)[:, 0]

def build(name):
    return {'inversionnet': InversionNet, 'ours': lambda: Ours(3, True), 'ours_k1': lambda: Ours(1, True),
            'ours_noraw': lambda: Ours(3, False), 'ours_raw': lambda: Ours(3, True, raw_only=True)}[name]()

# ---------------------------------------------------------------- objective

def mean_multires_ssim(a, b, scales=3):
    """Arithmetic mean of SSIM on [0,1] at three resolutions, not product MS-SSIM."""
    if scales < 1:
        raise ValueError('SSIM scales must be positive')
    values = []
    for level in range(scales):
        values.append(torch_ssim(a, b))
        if level < scales - 1:
            a = F.avg_pool2d(a[:, None], 2)[:, 0]
            b = F.avg_pool2d(b[:, None], 2)[:, 0]
    return torch.stack(values).mean()


def loss_fn(prediction, target, kind):
    if kind not in ('l1', 'l1ssim'):
        raise ValueError(f'Unknown loss: {kind}')
    pixel = (prediction - target).abs().mean()
    if kind == 'l1':
        return pixel
    return pixel + 0.5 * (1 - mean_multires_ssim((prediction + 1) / 2, (target + 1) / 2))


# ---------------------------------------------------------------- reproducibility

def data_fingerprint(d):
    """Hash the exact common inputs/labels loaded, independently of model variant."""
    digest = hashlib.sha256()
    for key in ('gathers', 'y', 'family', 'index'):
        value = np.ascontiguousarray(d[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def physics_fingerprint(d):
    digest = hashlib.sha256()
    for key in ('images', 'vref'):
        value = np.ascontiguousarray(d[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def physics_artifact_fingerprint(data_root=None, families='all'):
    root = Path(data_root or HERE)
    manifest = read_manifest(root)
    selected = list(manifest['families']) if families == 'all' else families.split(',')
    paths = ['work/depth_trend.npy', 'work/source_wavelet.npy', 'work/calibration_cache.json']
    paths.extend(f'work/inputs/{family}.npz' for family in sorted(selected))
    signatures = {path: _file_digest(root / path) for path in paths}
    return hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()


def manifest_fingerprint(manifest):
    # Protocol and source-index mappings are scientific provenance, not labels:
    # changing them must invalidate a saved run even if paths stay the same.
    shared = {key: value for key, value in manifest.items() if key != 'families'}
    shared['families'] = {name: {key: item for key, item in value.items()
                               if key not in ('ok', 'velocity_range_first50')}
                          for name, value in manifest['families'].items()}
    return hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng(rng):
    state = {'numpy': copy.deepcopy(rng.bit_generator.state), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, rng):
    rng.bit_generator.state = state['numpy']
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'].cpu())


def _ids_in_batches(count, size, require_two=False):
    if count < 1 or size < 1:
        raise ValueError('Batching requires positive data and batch sizes')
    if require_two and count == 1:
        yield np.array([0, 0])
        return
    start = 0
    while start < count:
        stop = min(start + max(size, 2 if require_two else 1), count)
        if require_two and count - stop == 1:
            stop = count
        yield np.arange(start, stop)
        start = stop


@torch.no_grad()
def calibrated_ema(ema, training, scale, batch_size=16, max_batches=None, device=None):
    """Recompute BN statistics with averaged weights and training inputs only.

    Work on a copy so validation/checkpoint cadence cannot perturb resume state.
    Modality dropout is off. A final singleton joins the preceding batch because
    InversionNet's last encoder BatchNorm sees a 1x1 feature map.
    """
    calibrated = copy.deepcopy(ema)
    calibrated.eval()
    norms = [module for module in calibrated.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    momenta = [module.momentum for module in norms]
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    for number, ids in enumerate(_ids_in_batches(len(training['y']), batch_size, require_two=True)):
        if max_batches is not None and number >= max_batches:
            break
        g, im, vr, _ = batch(training, ids, scale, device=device)
        prediction = calibrated(g, im, vr)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('EMA recalibration produced non-finite predictions')
    for module, momentum in zip(norms, momenta):
        module.momentum = momentum
    calibrated.eval()
    return calibrated


@torch.no_grad()
def validate(net, validation, scale, kind, batch_size=16, device=None):
    net.eval()
    totals = dict(loss=0.0, mae=0.0, rmse_squared=0.0, ssim=0.0)
    count = len(validation['y'])
    for ids in _ids_in_batches(count, batch_size):
        g, im, vr, y = batch(validation, ids, scale, device=device)
        prediction = net(g, im, vr)
        value = loss_fn(prediction, y, kind)
        if not torch.isfinite(prediction).all() or not torch.isfinite(value):
            raise FloatingPointError('Validation produced non-finite predictions/loss')
        totals['loss'] += len(ids) * float(value)
        totals['mae'] += len(ids) * float((prediction - y).abs().mean())
        totals['rmse_squared'] += len(ids) * float((prediction - y).square().mean())
        totals['ssim'] += len(ids) * float(torch_ssim((prediction + 1) / 2, (y + 1) / 2))
    return dict(loss=totals['loss'] / count, mae=totals['mae'] / count,
                rmse=math.sqrt(totals['rmse_squared'] / count), ssim=totals['ssim'] / count, n=count)


DEFAULTS = dict(model=None, seed=0, steps=8000, batch=64, lr=5e-4, loss='l1ssim',
                drop=0.3, families='all', train_limit_per_family=None,
                val_limit_per_family=None, experiment='default', ema_decay=0.999,
                bn_batches=None, eval_batch=16)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['inversionnet', 'ours', 'ours_k1', 'ours_noraw', 'ours_raw'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--loss', choices=['l1ssim', 'l1'])
    parser.add_argument('--drop', type=float)
    parser.add_argument('--families')
    parser.add_argument('--limit-per-family', '--train-limit-per-family', dest='train_limit_per_family', type=int)
    parser.add_argument('--val-limit-per-family', type=int)
    parser.add_argument('--experiment')
    parser.add_argument('--ema-decay', type=float)
    parser.add_argument('--bn-batches', type=int, help='Optional training batch cap for EMA recalibration; default all training examples')
    parser.add_argument('--data-root', default=HERE, help='Root containing data/, work/ and results/')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--eval-batch', type=int)
    parser.add_argument('--tag')
    parser.add_argument('--resume', help='Training checkpoint; explicit changes to the saved recipe are rejected')
    parser.add_argument('--stop-after', type=int, help='Stop cleanly after this total step without changing the schedule horizon')
    return parser.parse_args(argv)


def _resolve_config(args, checkpoint):
    saved = checkpoint['config'] if checkpoint else {}
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        if value is None:
            value = saved.get(key, default)
        elif checkpoint and value != saved.get(key):
            raise ValueError(f'Resume cannot change {key}: saved={saved.get(key)!r}, requested={value!r}')
        setattr(args, key, value)
    saved_tag = checkpoint.get('tag', '') if checkpoint else ''
    if args.tag is None:
        args.tag = saved_tag
    elif checkpoint and args.tag != saved_tag:
        raise ValueError('Resume cannot change the saved run tag')
    if args.model is None:
        raise ValueError('--model is required for a new run')
    if args.steps < 1 or args.batch < 2 or args.eval_batch < 1:
        raise ValueError('steps must be positive, training batch at least 2, and eval-batch positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.drop <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError('Invalid learning rate, modality dropout or EMA decay')
    if min(args.val_every, args.save_every) < 1:
        raise ValueError('val-every and save-every must be positive')
    if any(v is not None and v < 1 for v in (args.train_limit_per_family, args.val_limit_per_family, args.bn_batches, args.stop_after)):
        raise ValueError('Limits must be positive when specified')
    if not args.experiment or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.experiment):
        raise ValueError('experiment must use letters, digits, dots, underscores or hyphens')
    if any(c in args.tag for c in '/\\'):
        raise ValueError('tag must not contain path separators')


def main(argv=None):
    args = parse_args(argv)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint and (checkpoint.get('schema_version') != 2 or 'training_state' not in checkpoint):
        raise ValueError('Resume requires a version-2 training checkpoint')
    _resolve_config(args, checkpoint)
    device = resolve_device(args.device)
    root = Path(args.data_root)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    manifest = read_manifest(root)
    verify_data_files(manifest, root, args.families, require_hashes=False)
    needs_images = args.model not in ('inversionnet', 'ours_raw')
    training = load_data('train', args.families, needs_images, root, args.train_limit_per_family)
    validation = load_data('val', args.families, needs_images, root, args.val_limit_per_family)
    scale = float(np.abs(training['gathers']).max())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Training gathers must have a positive finite normalization scale')
    config = {key: getattr(args, key) for key in DEFAULTS}
    config.update(recipe_version=2, split={key: manifest[key] for key in ('n_train', 'n_val', 'n_total')},
                  manifest_fingerprint=manifest_fingerprint(manifest),
                  data_fingerprints={'train': data_fingerprint(training), 'val': data_fingerprint(validation)},
                  physics_fingerprints={'train': physics_fingerprint(training), 'val': physics_fingerprint(validation)} if needs_images else None,
                  physics_artifact_fingerprint=physics_artifact_fingerprint(root, args.families) if needs_images else None,
                  normalization_scale=scale)
    if checkpoint and config != checkpoint['config']:
        raise ValueError('Resume data, split, normalization or recipe differs from the saved checkpoint')
    net = build(args.model).to(device)
    ema = copy.deepcopy(net)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(500, max(1, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) *
        0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
    completed = 0
    history = []
    elapsed_before = 0.0
    if checkpoint:
        state = checkpoint['training_state']
        net.load_state_dict(state['net'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        completed = checkpoint['completed_steps']
        history = checkpoint.get('validation_history', [])
        elapsed_before = checkpoint.get('seconds', 0.0)
        restore_rng(state['rng'], rng)
    if completed > args.steps:
        raise ValueError('Checkpoint completed_steps exceeds its planned training steps')
    if args.stop_after is not None and args.stop_after <= completed:
        raise ValueError('stop-after must be greater than completed_steps')
    count = len(training['y'])
    nparam = sum(parameter.numel() for parameter in net.parameters())
    stem = f'{args.model}_{args.loss}_s{args.seed}_{args.experiment}{args.tag}'
    run_id = checkpoint['run_id'] if checkpoint else uuid.uuid4().hex
    resume_path = root / 'work' / 'checkpoints' / f'{stem}.pt'
    final_path = root / 'work' / 'models' / f'{stem}.pt'
    if checkpoint is None and (resume_path.exists() or final_path.exists()):
        raise ValueError('Run already exists; use --resume or a different --experiment/--tag')
    if checkpoint:
        for path in (resume_path, final_path):
            if path.exists():
                existing = checkpoint if path.resolve() == Path(args.resume).resolve() else torch.load(path, map_location='cpu', weights_only=False)
                if existing.get('run_id') != run_id:
                    raise ValueError(f'Resume destination belongs to another run: {path}')
    print(f'{args.model}: {count} train / {len(validation["y"])} validation, '
          f'{nparam / 1e6:.1f}M parameters, device={device}, completed={completed}/{args.steps}', flush=True)

    def snapshot(kind, status, inference_state=None):
        return dict(schema_version=2, kind=kind, status=status, experiment=args.experiment,
                    run_id=run_id, tag=args.tag,
                    families=args.families, model=args.model, loss=args.loss, seed=args.seed,
                    steps=args.steps, completed_steps=completed, scale=scale, params=nparam,
                    config=config, state=inference_state, validation_history=history,
                    seconds=elapsed_before + time.time() - started,
                    training_state={'net': net.state_dict(), 'ema': ema.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                    'rng': capture_rng(rng)})

    final_net = None
    try:
        for step in range(completed, args.steps):
            ids = rng.integers(0, count, args.batch)
            drop = (rng.random(args.batch) < args.drop).astype(np.float32) if args.model in ('ours', 'ours_k1') else None
            g, im, vr, y = batch(training, ids, scale, drop=drop, device=device)
            net.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = net(g, im, vr)
            loss = loss_fn(prediction, y, args.loss)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite prediction/loss at step {step + 1}; last valid checkpoint retained')
            loss.backward()
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in net.parameters() if parameter.grad is not None]
            if finite_gradients and not torch.stack(finite_gradients).all():
                raise FloatingPointError(f'Non-finite gradient at step {step + 1}; last valid checkpoint retained')
            optimizer.step()
            if not torch.stack([torch.isfinite(parameter).all() for parameter in net.parameters()]).all():
                raise FloatingPointError(f'Non-finite weights at step {step + 1}; last valid checkpoint retained')
            scheduler.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), net.parameters()):
                    average.mul_(args.ema_decay).add_(current, alpha=1 - args.ema_decay)
                # These are resume buffers only; inference copies are recalibrated below.
                for average, current in zip(ema.buffers(), net.buffers()):
                    average.copy_(current)
            completed = step + 1
            stopping = args.stop_after is not None and completed >= args.stop_after
            if completed % args.val_every == 0 or completed == args.steps:
                final_net = calibrated_ema(ema, training, scale, args.eval_batch, args.bn_batches, device)
                scores = validate(final_net, validation, scale, args.loss, args.eval_batch, device)
                history.append(dict(step=completed, **scores))
                print(f'step {completed}: train loss={loss.detach().item():.6f}; '
                      f'validation MAE={scores["mae"]:.6f}, SSIM={scores["ssim"]:.6f}', flush=True)
            elif completed == 1 or stopping:
                print(f'step {completed}: train loss={loss.detach().item():.6f}', flush=True)
            if completed % args.save_every == 0 or completed == args.steps or stopping:
                atomic_torch_save(snapshot('resume', 'complete' if completed == args.steps else 'in_progress'), resume_path)
            if stopping and completed < args.steps:
                print(f'Stopped at step {completed}; resume from {resume_path}', flush=True)
                return {'status': 'interrupted', 'checkpoint': str(resume_path), 'completed_steps': completed}
    except KeyboardInterrupt:
        # Only committed optimizer steps are resumable. The periodic checkpoint is
        # retained because an interrupt can occur halfway through optimizer.step.
        print(f'Interrupted; latest atomic checkpoint: {resume_path if resume_path.exists() else "none yet"}', flush=True)
        raise
    if final_net is None:
        final_net = calibrated_ema(ema, training, scale, args.eval_batch, args.bn_batches, device)
    atomic_torch_save(snapshot('final', 'complete', final_net.state_dict()), final_path)
    print(f'Saved calibrated model: {final_path}', flush=True)
    return {'status': 'complete', 'checkpoint': str(final_path), 'completed_steps': completed}


if __name__ == '__main__':
    main()

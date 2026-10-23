"""Evaluate compatible completed runs on validation or held-out test models.

OpenFWI metrics: MAE/RMSE on [-1, 1], Gaussian-window SSIM on [0, 1].
Use --experiment to select one experiment when the model directory contains
several. Incomplete runs and legacy checkpoints without provenance are skipped.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'physics'))
from metrics import metrics

spec = importlib.util.spec_from_file_location('train', os.path.join(HERE, '3_train.py'))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
# These are the intentional comparison axes; all other recipe/data settings
# must agree. A different budget, training population or preprocessing is not
# another seed of the same experiment.
RAW_MODELS = frozenset(('inversionnet', 'ours_raw'))
PHYSICS_FIELDS = ('physics_fingerprints', 'physics_artifact_fingerprint')
COMPARISON_AXES = frozenset(('model', 'loss', 'seed', 'experiment', *PHYSICS_FIELDS))


def comparison_config(checkpoint):
    config = checkpoint.get('config')
    if not isinstance(config, dict) or not config:
        raise ValueError('Final checkpoint has no experiment configuration')
    return {key: value for key, value in config.items() if key not in COMPARISON_AXES}


def select_checkpoints(paths, experiment=None):
    """Load and validate final runs before any predictions/results are written."""
    if not paths:
        raise ValueError('No checkpoints matched; train a run or check --checkpoints')
    candidates, skipped = [], []
    for path in sorted(paths):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f'{path}: checkpoint must be a dictionary')
        if experiment is not None and checkpoint.get('experiment') != experiment:
            continue
        if (checkpoint.get('kind') != 'final' or checkpoint.get('status') != 'complete'
                or checkpoint.get('steps', 0) <= 0
                or checkpoint.get('completed_steps') != checkpoint.get('steps')):
            skipped.append(f'{path}: not a completed final checkpoint')
            continue
        required = ('model', 'loss', 'seed', 'params', 'scale', 'state', 'experiment')
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise ValueError(f'{path}: final checkpoint missing {", ".join(missing)}')
        if not np.isfinite(checkpoint['scale']) or checkpoint['scale'] <= 0:
            raise ValueError(f'{path}: invalid gather normalization scale')
        comparison_config(checkpoint)
        # Final files can also contain optimizer/RNG state for recovery. It is
        # unnecessary for scoring and otherwise accumulates across every seed.
        checkpoint.pop('training_state', None)
        candidates.append((path, checkpoint))
    if not candidates:
        detail = f' for experiment {experiment!r}' if experiment is not None else ''
        raise ValueError(f'No completed final checkpoints{detail}; incomplete and legacy runs cannot be scored')
    experiments = {checkpoint['experiment'] for _, checkpoint in candidates}
    if len(experiments) != 1:
        raise ValueError('Multiple experiments found; select one with --experiment: '
                         + ', '.join(sorted(map(str, experiments))))
    expected = comparison_config(candidates[0][1])
    seen = set()
    names = set()
    expected_physics = None
    for path, checkpoint in candidates:
        current = comparison_config(checkpoint)
        changed = sorted(key for key in expected.keys() | current.keys()
                         if key not in expected or key not in current or expected[key] != current[key])
        if changed:
            raise ValueError(f'{path}: incompatible experiment configuration: {", ".join(changed)}')
        physics = {key: checkpoint['config'].get(key) for key in PHYSICS_FIELDS}
        if checkpoint['model'] in RAW_MODELS:
            if any(value is not None for value in physics.values()):
                raise ValueError(f'{path}: raw-only model unexpectedly declares physics inputs')
        else:
            if (not isinstance(physics['physics_fingerprints'], dict)
                    or set(physics['physics_fingerprints']) != {'train', 'val'}
                    or not all(physics['physics_fingerprints'].values())
                    or not physics['physics_artifact_fingerprint']):
                raise ValueError(f'{path}: image model has incomplete physics provenance')
            if expected_physics is not None and physics != expected_physics:
                raise ValueError(f'{path}: incompatible physics inputs across image models')
            expected_physics = physics
        identity = (checkpoint['model'], checkpoint['loss'], checkpoint['seed'])
        if identity in seen:
            raise ValueError(f'{path}: duplicate model/loss/seed {identity}; select one run per seed')
        seen.add(identity)
        if Path(path).stem in names:
            raise ValueError(f'{path}: duplicate checkpoint filename would overwrite predictions')
        names.add(Path(path).stem)
    return candidates, skipped


def validate_dataset_provenance(checkpoint, data_root):
    """Bind the saved split and manifest to actual raw file bytes, including headers."""
    manifest = T.read_manifest(data_root)
    config = checkpoint['config']
    split = {key: manifest[key] for key in ('n_train', 'n_val', 'n_total')}
    if config.get('split') != split:
        raise ValueError('Evaluation manifest split differs from the checkpoint')
    if config.get('manifest_fingerprint') != T.manifest_fingerprint(manifest):
        raise ValueError('Evaluation manifest fingerprint differs from the checkpoint')
    root = Path(data_root)
    for family, entry in manifest['families'].items():
        checksums = entry.get('sha256', {})
        for kind in ('gathers', 'models'):
            expected = checksums.get(kind)
            if not expected:
                raise ValueError(f'{family}: missing {kind} file checksum; regenerate the data manifest')
            path = root / entry[kind]
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f'{family}: {kind} file checksum changed since the recorded dataset')


def validate_physics_provenance(checkpoints, data_root):
    """Compare exact cached artifacts once; image models already agree on scope."""
    for _, checkpoint in checkpoints:
        if checkpoint['model'] not in RAW_MODELS:
            expected = checkpoint['config']['physics_artifact_fingerprint']
            actual = T.physics_artifact_fingerprint(data_root, checkpoint['config']['families'])
            if actual != expected:
                raise ValueError('Physics inputs changed since training; regenerate/retrain this experiment')
            return


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Device must be auto, cpu, cuda, cuda:N, or mps')
    return device


@torch.no_grad()
def predict(checkpoint, data, batch_size=48):
    if batch_size < 1:
        raise ValueError('Evaluation batch size must be positive')
    if len(data['y']) == 0:
        raise ValueError('Evaluation split is empty')
    net = T.build(checkpoint['model']).to(T.DEV)
    net.load_state_dict(checkpoint['state'])
    net.eval()
    output = []
    for start in range(0, len(data['y']), batch_size):
        ids = np.arange(start, min(start + batch_size, len(data['y'])))
        g, im, vr, _ = T.batch(data, ids, checkpoint['scale'])
        prediction = net(g, im, vr)
        if not torch.isfinite(prediction).all():
            raise ValueError('Network produced nonfinite predictions')
        output.append(prediction.cpu().numpy())
    return np.concatenate(output).astype(np.float64) * 1500 + 3000


def score_predictions(prediction, truth, families):
    """Aggregate squared errors before the square root for family RMSE."""
    if prediction.shape != truth.shape or prediction.ndim != 3 or len(families) != len(truth):
        raise ValueError('Predictions, truth and family labels must have matching image counts/shapes')
    if len(truth) == 0:
        raise ValueError('Cannot score an empty split')
    per_image = [metrics(pred, true) for pred, true in zip(prediction, truth)]
    result = {}
    for family in dict.fromkeys(families):
        selected = [score for score, label in zip(per_image, families) if label == family]
        result[str(family)] = dict(
            n=len(selected), mae=float(np.mean([score['mae'] for score in selected])),
            rmse=float(np.sqrt(np.mean([score['rmse'] ** 2 for score in selected]))),
            ssim=float(np.mean([score['ssim'] for score in selected])))
    return result


def write_table(rows, families, destination):
    keys = sorted({(row['model'], row['loss']) for row in rows})
    lines = []
    for metric in ('mae', 'rmse', 'ssim'):
        direction = 'higher' if metric == 'ssim' else 'lower'
        lines.extend([f'\n### {metric.upper()} (mean over seeds; {direction} is better)\n',
                      '| family | ' + ' | '.join(f'{model} ({loss})' for model, loss in keys) + ' |',
                      '|---' * (len(keys) + 1) + '|'])
        for family in list(families) + ['mean']:
            cells = []
            for model, loss in keys:
                values = [row[metric] for row in rows
                          if row['model'] == model and row['loss'] == loss
                          and (row['family'] == family or family == 'mean')]
                cells.append(f'{np.mean(values):.4f}')
            lines.append(f'| {family} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    Path(destination).write_text(text)
    return text


def plot_examples(predictions, truth, data, families, destination):
    # Choose the same seed for both displayed methods; do not combine seeds.
    pairs = {}
    for name, checkpoint, prediction in predictions:
        if checkpoint['loss'] == 'l1ssim' and checkpoint['model'] in ('inversionnet', 'ours'):
            pairs.setdefault(checkpoint['seed'], {})[checkpoint['model']] = prediction
    shared_seeds = sorted(seed for seed, pair in pairs.items() if len(pair) == 2)
    if not shared_seeds:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed = shared_seeds[0]
    base, ours = pairs[seed]['inversionnet'], pairs[seed]['ours']
    fig, axes = plt.subplots(len(families), 3, figsize=(9, 3 * len(families)), squeeze=False)
    for row, family in enumerate(families):
        ids = np.where(data['family'] == family)[0]
        # Fixed midpoint index, independent of either model's performance.
        index = ids[len(ids) // 2]
        lo, hi = truth[index].min(), truth[index].max()
        views = [(truth[index], f'{family} model {data["index"][index]}: truth'),
                 (base[index], f'InversionNet seed {seed}, MAE {metrics(base[index], truth[index])["mae"]:.3f}'),
                 (ours[index], f'ours seed {seed}, MAE {metrics(ours[index], truth[index])["mae"]:.3f}')]
        for column, (image, title) in enumerate(views):
            axes[row, column].imshow(image, cmap='viridis', vmin=lo, vmax=hi)
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis('off')
    plt.tight_layout()
    plt.savefig(destination, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', help='Quoted glob; defaults to DATA_ROOT/work/models/*.pt')
    parser.add_argument('--data-root', default=HERE, help='Directory containing data/, work/, and results/')
    parser.add_argument('--families', default='all')
    parser.add_argument('--experiment', help='Exact experiment label saved during training')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--limit-per-family', type=int, help='Optional pilot cap within the evaluation partition')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--output-dir', help='Defaults to DATA_ROOT/results/SPLIT')
    args = parser.parse_args()
    try:
        if args.batch < 1:
            raise ValueError('--batch must be positive')
        if args.limit_per_family is not None and args.limit_per_family < 1:
            raise ValueError('--limit-per-family must be positive')
        T.DEV = resolve_device(args.device)
        pattern = args.checkpoints or os.path.join(args.data_root, 'work', 'models', '*.pt')
        checkpoints, skipped = select_checkpoints(glob.glob(pattern), args.experiment)
        for reason in skipped:
            print('Skipping', reason, flush=True)
        validate_dataset_provenance(checkpoints[0][1], args.data_root)
        validate_physics_provenance(checkpoints, args.data_root)
        needs_images = any(checkpoint['model'] not in RAW_MODELS for _, checkpoint in checkpoints)
        data = T.load_data(args.split, args.families, images=needs_images, data_root=args.data_root,
                           limit_per_family=args.limit_per_family)
        if not len(data['y']):
            raise ValueError('Evaluation split contains no models')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    result_dir = Path(args.output_dir or os.path.join(args.data_root, 'results', args.split))
    (result_dir / 'preds').mkdir(parents=True, exist_ok=True)
    (result_dir / 'figures').mkdir(exist_ok=True)
    truth = data['y'].astype(np.float64) * 1500 + 3000
    rows, predictions = [], []
    print(f'Evaluating {len(checkpoints)} completed runs on {args.split}, device={T.DEV}', flush=True)
    for path, checkpoint in checkpoints:
        name = Path(path).stem
        prediction = predict(checkpoint, data, args.batch)
        predictions.append((name, checkpoint, prediction))
        np.savez_compressed(result_dir / 'preds' / f'{name}.npz', pred=prediction.astype(np.float32),
                            family=data['family'], index=data['index'])
        scores = score_predictions(prediction, truth, data['family'])
        for family, score in scores.items():
            rows.append(dict(network=name, model=checkpoint['model'], loss=checkpoint['loss'],
                             seed=checkpoint['seed'], params=checkpoint['params'], family=family,
                             experiment=checkpoint['experiment'], split=args.split,
                             steps=checkpoint['completed_steps'], **score))
        print(f'{name}: mean MAE {np.mean([score["mae"] for score in scores.values()]):.4f}', flush=True)
    with (result_dir / 'scores.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    (result_dir / 'experiment.json').write_text(json.dumps(dict(
        experiment=checkpoints[0][1]['experiment'], config=comparison_config(checkpoints[0][1]),
        checkpoints=[str(Path(path).resolve()) for path, _ in checkpoints],
        split=args.split, families=args.families, limit_per_family=args.limit_per_family), indent=2) + '\n')
    families = list(dict.fromkeys(data['family']))
    print(write_table(rows, families, result_dir / 'table.md'))
    plot_examples(predictions, truth, data, families, result_dir / 'figures' / 'examples.png')
    print('Saved evaluation to', result_dir, flush=True)


if __name__ == '__main__':
    main()

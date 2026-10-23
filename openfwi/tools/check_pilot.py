"""Check a selected device and one training batch before launching a pilot.

This does not establish scientific accuracy. It checks data separation, finite
loss/gradients, output dimensions, and a parameter update on the requested device.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_training', HERE / '3_train.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def check(root, model, device, batch_size=2):
    device = T.resolve_device(device)
    if batch_size < 2:
        raise ValueError('Training BatchNorm requires --batch >= 2')
    T.verify_data_files(T.read_manifest(root), root, require_hashes=True)
    images = model not in ('inversionnet', 'ours_raw')
    train = T.load_data('train', images=images, data_root=root, limit_per_family=batch_size)
    val = T.load_data('val', images=images, data_root=root, limit_per_family=1)
    train_keys = set(zip(train['family'], train['index']))
    val_keys = set(zip(val['family'], val['index']))
    if train_keys & val_keys:
        raise ValueError('Training and validation samples overlap')
    scale = float(np.abs(train['gathers']).max())
    if len(train['y']) < batch_size:
        raise ValueError('Not enough training samples for the requested batch')
    data = T.batch(train, np.arange(batch_size), scale, device=device)
    net = T.build(model).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    prediction = net(*data[:3])
    if prediction.shape != data[3].shape or not torch.isfinite(prediction).all():
        raise ValueError('Wrong output shape or nonfinite model output')
    loss = T.loss_fn(prediction, data[3], 'l1ssim')
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or nonfinite gradients')
    param = next(p for p in net.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
    before = param.detach().clone()
    opt.step()
    if torch.equal(before, param):
        raise ValueError('Optimizer did not change parameters')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return dict(status='passed', model=model, device=str(device), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, batch=batch_size,
                loss=float(loss.detach().cpu()), parameters=sum(p.numel() for p in net.parameters()),
                training_examples=len(train['y']), validation_examples=len(val['y']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=HERE)
    parser.add_argument('--model', default='ours', choices=['ours', 'ours_raw', 'ours_k1', 'ours_noraw', 'inversionnet'])
    parser.add_argument('--device', required=True, choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(json.dumps(check(args.data_root, args.model, args.device, args.batch), indent=2))


if __name__ == '__main__':
    main()

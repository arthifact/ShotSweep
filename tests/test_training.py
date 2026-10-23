"""Regression checks for the training objective, partitions and resume contract."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shotsweep_training_tests', ROOT / 'openfwi' / '3_train.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
torch.set_num_threads(2)


def make_fixture(root):
    root = Path(root)
    (root / 'data').mkdir(parents=True)
    manifest = dict(n_train=4, n_val=2, n_total=8,
                    families={'fixture': dict(gathers='data/g.npy', models='data/y.npy')})
    (root / 'data' / 'manifest.json').write_text(json.dumps(manifest))
    g = np.empty((8, 5, 1000, 70), np.float32)
    y = np.empty((8, 1, 70, 70), np.float32)
    for index in range(8):
        g[index].fill(index + 1)
        y[index].fill(2500 + index * 50)
    np.save(root / 'data' / 'g.npy', g)
    np.save(root / 'data' / 'y.npy', y)
    return manifest


class TinyModel(nn.Module):
    """Has BN and stochastic training to exercise exact resume/recalibration."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), nn.Dropout(0.2),
                                 nn.Conv2d(2, 1, 1), nn.Tanh())

    def forward(self, g, im, vr):
        value = g.mean(dim=(1, 2, 3))[:, None, None, None].expand(-1, 1, 70, 70)
        return self.net(value)[:, 0]


class LossAndArchitectureTests(unittest.TestCase):
    def test_structural_loss_pushes_low_prediction_toward_truth(self):
        prediction = torch.full((1, 70, 70), -0.07, requires_grad=True)
        target = torch.full_like(prediction, 0.03)
        loss = T.loss_fn(prediction, target, 'l1ssim')
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess(float(prediction.grad.sum()), 0)
        self.assertAlmostEqual(float(T.loss_fn(target, target, 'l1ssim')), 0, places=6)

    def test_final_baseline_head_matches_reference_order(self):
        decoder = T.InvDecoder(d=(2, 3, 4, 5, 6))
        self.assertIsInstance(decoder.head[0], nn.Conv2d)
        self.assertIsInstance(decoder.head[1], nn.BatchNorm2d)
        self.assertIsInstance(decoder.head[2], nn.Tanh)
        seen = []
        hook = decoder.head.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
        decoder.eval()
        with torch.no_grad():
            output = decoder(torch.randn(2, 6, 1, 1))
        hook.remove()
        self.assertEqual(seen[0][-2:], (70, 70))
        self.assertEqual(output.shape, (2, 70, 70))

    def test_raw_control_ignores_physics_and_receives_raw_gradients(self):
        model = T.Ours(w=(4, 8, 16, 32), raw_only=True)
        # A cheap encoder keeps this focused on the control's information flow.
        model.enc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(5, 512, 1))
        model.eval()
        g = torch.randn(2, 5, 16, 10, requires_grad=True)
        dummy = torch.zeros(2, 1, 1, 1, 1)
        first = model(g, dummy, torch.zeros(2, 70, 70))
        second = model(g, torch.randn(2, 3, 5, 70, 70), torch.randn(2, 70, 70))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        first.mean().backward()
        self.assertGreater(float(g.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(g.grad).all())
        standard = T.Ours(w=(4, 8, 16, 32))
        control = T.Ours(w=(4, 8, 16, 32), raw_only=True)
        self.assertEqual(sum(p.numel() for p in standard.parameters()), sum(p.numel() for p in control.parameters()))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = make_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_disjoint_explicit_splits_and_train_only_normalization(self):
        train = T.load_data('train', images=False, data_root=self.root)
        val = T.load_data('val', images=False, data_root=self.root, limit_per_family=1)
        test = T.load_data('test', images=False, data_root=self.root)
        self.assertEqual(train['index'].tolist(), [0, 1, 2, 3])
        self.assertEqual(val['index'].tolist(), [4])
        self.assertEqual(test['index'].tolist(), [6, 7])
        scale = float(np.abs(train['gathers']).max())
        g, *_ = T.batch(val, [0], scale, device='cpu')
        self.assertGreater(float(g.max()), 1)  # no refitting on validation amplitude
        self.assertAlmostEqual(scale, float(np.float16(np.log1p(4))), places=5)

    def test_legacy_manifest_and_nonfinite_arrays_rejected(self):
        legacy = dict(self.manifest)
        del legacy['n_val']
        path = self.root / 'data' / 'manifest.json'
        path.write_text(json.dumps(legacy))
        with self.assertRaisesRegex(ValueError, 'explicitly define'):
            T.load_data('train', images=False, data_root=self.root)
        path.write_text(json.dumps(self.manifest))
        g = np.load(self.root / 'data' / 'g.npy', mmap_mode='r+')
        g[0, 0, 0, 0] = np.nan
        g.flush()
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            T.load_data('train', images=False, data_root=self.root)

    def test_invalid_mirror_and_normalization_rejected(self):
        data = T.load_data('train', images=False, data_root=self.root)
        with self.assertRaisesRegex(ValueError, 'central source'):
            T.batch(data, [0], 1, mirror=[True], device='cpu')
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            T.batch(data, [0], 0, device='cpu')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            T.load_data('train', families='unknown', images=False, data_root=self.root)

    def test_images_require_current_calibration(self):
        with self.assertRaisesRegex(ValueError, 'physics calibration'):
            T.load_data('train', images=True, data_root=self.root)

    def test_physics_digest_changes_independently_of_common_data(self):
        data = T.load_data('train', images=False, data_root=self.root)
        changed = dict(data)
        changed['images'] = data['images'] + 1
        self.assertEqual(T.data_fingerprint(data), T.data_fingerprint(changed))
        self.assertNotEqual(T.physics_fingerprint(data), T.physics_fingerprint(changed))

    def test_manifest_digest_includes_source_mapping_and_protocol(self):
        original = T.manifest_fingerprint(self.manifest)
        changed = copy.deepcopy(self.manifest)
        changed['families']['fixture']['source_indices'] = [0, 1, 2, 3, 400, 401, 452, 453]
        self.assertNotEqual(original, T.manifest_fingerprint(changed))
        changed = copy.deepcopy(self.manifest)
        changed['synthetic'] = True
        self.assertNotEqual(original, T.manifest_fingerprint(changed))

    def test_actual_data_hash_is_checked_before_training(self):
        self.manifest['families']['fixture']['sha256'] = {
            'gathers': T._file_digest(self.root / 'data' / 'g.npy'),
            'models': T._file_digest(self.root / 'data' / 'y.npy')}
        T.verify_data_files(self.manifest, self.root)
        g = np.load(self.root / 'data' / 'g.npy', mmap_mode='r+')
        g[7, 0, 0, 0] += 1  # Even unchanged train/val data cannot hide a changed held-out file.
        g.flush()
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            T.verify_data_files(self.manifest, self.root)
        (self.root / 'data' / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(T, 'build') as builder:
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                T.main(['--model', 'inversionnet', '--steps', '1', '--batch', '2',
                        '--device', 'cpu', '--data-root', str(self.root)])
            builder.assert_not_called()


class CheckpointTests(unittest.TestCase):
    def test_atomic_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.pt'
            T.atomic_torch_save({'step': 1}, path)
            original = path.read_bytes()
            with patch.object(T.torch, 'save', side_effect=RuntimeError('disk failure')):
                with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                    T.atomic_torch_save({'step': 2}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_resume_exactly_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            make_fixture(first)
            make_fixture(second)
            common = ['--model', 'inversionnet', '--steps', '6', '--batch', '2', '--lr', '.01',
                      '--seed', '9', '--device', 'cpu', '--val-every', '2', '--save-every', '2', '--eval-batch', '2', '--tag', '_named']
            with patch.object(T, 'build', side_effect=lambda name: TinyModel()), contextlib.redirect_stdout(io.StringIO()):
                full_result = T.main(common + ['--data-root', first])
                interrupted = T.main(common + ['--data-root', second, '--stop-after', '3'])
                resumed_result = T.main(['--resume', interrupted['checkpoint'], '--data-root', second,
                                         '--device', 'cpu', '--val-every', '2', '--save-every', '2'])
            full = torch.load(full_result['checkpoint'], weights_only=False)
            resumed = torch.load(resumed_result['checkpoint'], weights_only=False)
            self.assertEqual(resumed['completed_steps'], 6)
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(resumed['kind'], 'final')
            self.assertTrue(resumed_result['checkpoint'].endswith('_named.pt'))
            self.assertEqual(resumed['config']['eval_batch'], 2)
            self.assertEqual(full['config'], resumed['config'])
            self.assertEqual(full['validation_history'], resumed['validation_history'])
            for key in ('net', 'ema'):
                for name, value in full['training_state'][key].items():
                    torch.testing.assert_close(value, resumed['training_state'][key][name], atol=0, rtol=0)
            for name, value in full['state'].items():
                torch.testing.assert_close(value, resumed['state'][name], atol=0, rtol=0)
            torch.testing.assert_close(full['training_state']['rng']['torch'], resumed['training_state']['rng']['torch'], atol=0, rtol=0)
            self.assertEqual(full['training_state']['scheduler'], resumed['training_state']['scheduler'])
            # A changed training recipe cannot silently resume under the old run name.
            with patch.object(T, 'build', side_effect=lambda name: TinyModel()):
                with self.assertRaisesRegex(ValueError, 'Resume cannot change batch'):
                    T.main(['--resume', interrupted['checkpoint'], '--data-root', second, '--batch', '3'])

    def test_ema_calibration_uses_averaged_weights_without_mutating_resume_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_fixture(temporary)
            data = T.load_data('train', images=False, data_root=temporary)
            model = TinyModel()
            original = copy.deepcopy(model.state_dict())
            calibrated = T.calibrated_ema(model, data, 2, batch_size=2, device='cpu')
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)
            self.assertFalse(calibrated.training)
            self.assertEqual(int(calibrated.net[1].num_batches_tracked), 2)
            self.assertFalse(torch.equal(calibrated.net[1].running_mean, original['net.1.running_mean']))

    def test_nonfinite_training_loss_never_saves_a_final_model(self):
        class BadModel(TinyModel):
            def forward(self, g, im, vr):
                return super().forward(g, im, vr) * float('nan')
        with tempfile.TemporaryDirectory() as temporary:
            make_fixture(temporary)
            with patch.object(T, 'build', side_effect=lambda name: BadModel()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(FloatingPointError, 'Non-finite prediction/loss'):
                    T.main(['--model', 'inversionnet', '--steps', '1', '--batch', '2', '--device', 'cpu', '--data-root', temporary])
            self.assertFalse((Path(temporary) / 'work' / 'models').exists())


if __name__ == '__main__':
    unittest.main()

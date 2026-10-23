"""Physics and processing regressions; run with python -m unittest discover -s tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'openfwi' / 'physics'))
import gpu
import background as BG

spec = importlib.util.spec_from_file_location('process_physics_tests', ROOT / 'openfwi' / '2_process.py')
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)


class AdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sim = gpu.Sim(device='cpu', dtype=torch.float64)
        cls.velocity = np.full((1, 70, 70), 2000.0)
        truth = cls.velocity.copy()
        truth[:, 8:] = 2400.0
        cls.observed = cls.sim.gathers(truth)

    def test_laplacian_transpose_including_boundary(self):
        generator = torch.Generator().manual_seed(8)
        x = torch.randn(2, 9, 11, generator=generator, dtype=torch.float64)
        y = torch.randn(2, 9, 11, generator=generator, dtype=torch.float64)
        self.assertAlmostEqual(float((gpu.lap(x) * y).sum()), float((x * gpu.lap_adjoint(y)).sum()), places=10)

    def test_source_and_padding_derivatives_match_finite_differences(self):
        gradient, _ = self.sim.gradient(self.velocity, self.observed)
        gradient = gradient.sum(1)
        directions = {
            'interior': (slice(10, 30), 35),
            'left_boundary': (slice(10, 30), 0),
            'bottom_boundary': (69, slice(25, 45)),
            'top_boundary': (0, slice(25, 45)),
            'velocity_dependent_source': (1, 34),
        }
        for name, location in directions.items():
            with self.subTest(name=name):
                direction = np.zeros_like(self.velocity)
                direction[(0,) + location] = 1
                epsilon = 0.1
                def loss(velocity):
                    residual = self.sim.gathers(velocity) - self.observed
                    return 0.5 * np.sum(residual * residual)
                numerical = (loss(self.velocity + epsilon * direction) - loss(self.velocity - epsilon * direction)) / (2 * epsilon)
                adjoint = np.sum(gradient * direction)
                self.assertLess(abs(numerical - adjoint) / max(abs(numerical), 1.0), 1e-4)

    def test_muted_gradient_and_checkpoint_recomputation(self):
        mute = np.array([2000.0])
        gradient, illumination = self.sim.gradient(self.velocity, self.observed, v_mute=mute)
        direction = np.zeros_like(self.velocity)
        direction[:, 69, 25:45] = 1
        weights = self.sim.mute_weights(mute).cpu().numpy()
        def loss(velocity):
            residual = self.sim.gathers(velocity) - self.observed
            return 0.5 * np.sum(weights * residual ** 2)
        epsilon = 0.1
        numerical = (loss(self.velocity + epsilon * direction) - loss(self.velocity - epsilon * direction)) / (2 * epsilon)
        adjoint = (gradient.sum(1) * direction).sum()
        self.assertLess(abs(numerical - adjoint) / abs(numerical), 1e-4)
        other = gpu.Sim(device='cpu', dtype=torch.float64, checkpoint_interval=73)
        other_gradient, other_illumination = other.gradient(self.velocity, self.observed, v_mute=mute)
        np.testing.assert_allclose(other_gradient, gradient, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(other_illumination, illumination, rtol=1e-12, atol=1e-6)

    def test_checkpoint_memory_is_bounded(self):
        full_history = 24 * 5 * 1000 * 150 * 150 * 4
        self.assertLess(gpu.imaging_memory_bytes(24), full_history / 5)

    def test_unavailable_explicit_device_fails(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                gpu.resolve_device('cuda')


class SurfaceVelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        models = [np.full((70, 70), v, dtype=float) for v in (1500, 2500, 4500)]
        for depth in (5, 2):
            model = np.full((70, 70), 1500.0)
            model[depth:] = 4500.0
            models.append(model)
        cls.observed = gpu.Sim(device='cpu').gathers(np.array(models), chunk=3)

    def test_homogeneous_velocities_and_quality(self):
        for index, truth in enumerate((1500, 2500, 4500)):
            estimate = BG.estimate_surface_velocity(self.observed[index])
            self.assertLess(abs(estimate['velocity'] - truth) / truth, 0.03)
            self.assertTrue(0 < estimate['confidence'] <= 1)
            self.assertFalse(estimate['possible_refraction'])

    def test_shallow_refractor_uses_short_offsets(self):
        estimate = BG.estimate_surface_velocity(self.observed[3])
        self.assertLess(abs(estimate['velocity'] - 1500), 100)
        self.assertTrue(estimate['possible_refraction'])
        self.assertLess(estimate['confidence'], 0.6)

    def test_unresolved_shallow_refractor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unreliable'):
            BG.estimate_surface_velocity(self.observed[4])

    def test_zero_and_nonfinite_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no nonzero'):
            BG.estimate_surface_velocity(np.zeros((5, 1000, 70)))
        observed = self.observed[0].copy()
        observed[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            BG.estimate_surface_velocity(observed)


class ProcessingTests(unittest.TestCase):
    def test_atomic_write_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'values.npy'
            process.atomic_npy(path, np.array([1]))
            def failed_write(stream):
                stream.write(b'broken')
                raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):
                process._atomic_write(path, failed_write)
            np.testing.assert_array_equal(np.load(path), [1])
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['values.npy'])

    def test_training_only_calibration_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            models = np.stack([np.full((1, 70, 70), v) for v in (1800., 2200., 3200., 4400.)])
            np.save(root / 'data/models.npy', models)
            np.save(root / 'data/gathers.npy', np.zeros((4, 5, 1000, 70), dtype=np.float32))
            manifest = {'schema': 2, 'n_train': 2, 'n_val': 1, 'n_total': 4,
                        'families': {'synthetic': {'models': 'data/models.npy', 'gathers': 'data/gathers.npy'}}}
            process.atomic_json(root / 'data/manifest.json', manifest)
            seen = []
            def calibrate(training_models, training_gathers):
                seen.append([float(model[0, 0]) for model in training_models])
                return gpu.RICKER
            def images(backgrounds, observations, **kwargs):
                values = np.ones((len(backgrounds), 5, 70, 70), np.float32)
                return values, values
            estimate = {'velocity': 2000., 'confidence': 1., 'fit_rmse_ms': 0., 'possible_refraction': False}
            argv = ['2_process.py', '--data-root', str(root), '--device', 'cpu', '--batch', '1']
            with mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(gpu, 'calibrate', side_effect=calibrate), \
                 mock.patch.object(gpu.Sim, 'images', side_effect=images), \
                 mock.patch.object(BG, 'estimate_surface_velocity', return_value=estimate), \
                 contextlib.redirect_stdout(io.StringIO()):
                process.main()
                cache = root / 'work/inputs/synthetic.npz'
                with np.load(cache) as data:
                    first = json.loads(str(data['cache_provenance'].item()))
                self.assertEqual(seen, [[1800., 2200.]])
                self.assertTrue(process.inputs_valid(cache, first['fingerprint'], 4))
                process.main()
                self.assertEqual(len(seen), 1)
                manifest['n_train'] = 1
                manifest['n_val'] = 2
                process.atomic_json(root / 'data/manifest.json', manifest)
                process.main()
                self.assertEqual(seen, [[1800., 2200.], [1800.]])
                self.assertFalse(process.inputs_valid(cache, first['fingerprint'], 4))
                with np.load(cache) as data:
                    second = json.loads(str(data['cache_provenance'].item()))
                self.assertNotEqual(first['calibration_fingerprint'], second['calibration_fingerprint'])
                # A wavelet modification must invalidate the calibration ledger.
                process.atomic_npy(root / 'work/source_wavelet.npy', gpu.RICKER * 2)
                self.assertFalse(process.calibration_valid(root / 'work', second['calibration_fingerprint']))

    def test_legacy_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.npz'
            np.savez(path, images=np.zeros((1, 3, 5, 70, 70)), surface_velocity=np.array([2000.]))
            self.assertFalse(process.inputs_valid(path, 'anything', 1))


if __name__ == '__main__':
    unittest.main()

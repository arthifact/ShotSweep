"""Metric reference parity and safeguards against invalid experiment comparisons."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'openfwi' / 'physics'))
import metrics as M

spec = importlib.util.spec_from_file_location('shotsweep_evaluation', ROOT / 'openfwi' / '4_test.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
torch.set_num_threads(2)


def independent_ssim(a, b):
    """NumPy window reduction of the published OpenFWI SSIM definition."""
    position = np.arange(11, dtype=np.float64) - 5
    gaussian = np.exp(-(position ** 2) / (2 * 1.5 ** 2))
    kernel = np.outer(gaussian, gaussian)
    kernel /= kernel.sum()

    def average(value):
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(value, 5), (11, 11))
        return np.einsum('ijxy,xy->ij', windows, kernel)

    ma, mb = average(a), average(b)
    va, vb = average(a * a) - ma * ma, average(b * b) - mb * mb
    covariance = average(a * b) - ma * mb
    return np.mean(((2 * ma * mb + .01 ** 2) * (2 * covariance + .03 ** 2)) /
                   ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2)))


class MetricTests(unittest.TestCase):
    def test_matches_independent_official_definition_at_image_edges(self):
        rng = np.random.default_rng(42)
        a, b = rng.random((17, 19)), rng.random((17, 19))
        self.assertAlmostEqual(M.ssim(a, b), independent_ssim(a, b), places=12)

    def test_velocity_range_regression(self):
        a, b = np.full((70, 70), 2850.), np.full((70, 70), 3150.)
        result = M.metrics(a, b)
        self.assertAlmostEqual(result['mae'], .2)
        self.assertAlmostEqual(result['rmse'], .2)
        self.assertAlmostEqual(result['ssim'], .9758969670229866, places=10)

    def test_identical_images_and_per_image_scores(self):
        a = torch.rand((2, 16, 17), dtype=torch.float64)
        score = M.torch_ssim(a, a, size_average=False)
        torch.testing.assert_close(score, torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(M.torch_ssim(a, a), M.torch_ssim(a[:, None], a[:, None]))

    def test_corrected_loss_gradient_moves_toward_target(self):
        prediction = torch.full((1, 70, 70), -.2, requires_grad=True)
        truth = torch.full_like(prediction, .1)
        loss = (prediction - truth).abs().mean() + .5 * (
            1 - M.torch_ssim((prediction + 1) / 2, (truth + 1) / 2))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess(prediction.grad.sum().item(), 0)

    def test_retains_small_errors_and_does_not_clip_large_errors(self):
        truth = np.full((11, 11), 4500.)
        self.assertGreater(M.metrics(truth + .001, truth)['mae'], 0)
        self.assertAlmostEqual(M.metrics(truth + 300, truth)['mae'], .2)

    def test_bad_shapes_and_nonfinite_values_fail(self):
        with self.assertRaises(ValueError):
            M.torch_ssim(torch.ones(2, 3), torch.ones(2, 3))
        with self.assertRaises(ValueError):
            M.metrics(np.full((7, 7), np.nan), np.ones((7, 7)))
        with self.assertRaises(ValueError):
            M.torch_ssim(torch.ones(1, 7, 7), torch.ones(1, 7, 7), window_size=4)

    def test_rmse_is_computed_over_all_family_pixels(self):
        truth = np.full((2, 11, 11), 3000.)
        prediction = truth.copy()
        prediction[1] += 300
        score = E.score_predictions(prediction, truth, np.array(['a', 'a']))['a']
        self.assertAlmostEqual(score['mae'], .1)
        self.assertAlmostEqual(score['rmse'], np.sqrt(.02))
        self.assertNotAlmostEqual(score['rmse'], .1)

    def test_scoring_rejects_missing_predictions(self):
        with self.assertRaisesRegex(ValueError, 'matching'):
            E.score_predictions(np.zeros((1, 7, 7)), np.zeros((2, 7, 7)), ['a', 'a'])


def checkpoint(model='ours', seed=0, experiment='pilot'):
    config = dict(recipe_version=2, model=model, loss='l1ssim', seed=seed,
                  experiment=experiment, steps=20, batch=2, lr=.0005, drop=.3,
                  families='flatvel-a', train_limit_per_family=8,
                  val_limit_per_family=4, split=dict(n_train=400, n_val=52, n_total=500),
                  manifest_fingerprint='manifest-hash',
                  data_fingerprints=dict(train='train-hash', val='val-hash'),
                  normalization_scale=2., ema_decay=.999, bn_batches=2)
    config['physics_fingerprints'] = None if model in E.RAW_MODELS else dict(train='image-train', val='image-val')
    config['physics_artifact_fingerprint'] = None if model in E.RAW_MODELS else 'physics-artifacts'
    return dict(model=model, loss='l1ssim', seed=seed, experiment=experiment,
                kind='final', status='complete', steps=20, completed_steps=20,
                params=1, scale=2., state={}, config=config)


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, name, ck):
        path = self.root / (name + '.pt')
        torch.save(ck, path)
        return str(path)

    def data_manifest(self):
        gathers, models = self.root / 'gathers.npy', self.root / 'models.npy'
        np.save(gathers, np.ones((1, 5, 2, 2), dtype=np.float32))
        np.save(models, np.full((1, 1, 2, 2), 3000., dtype=np.float32))
        return dict(n_train=400, n_val=52, n_total=500, families={'flatvel-a': dict(
            gathers=gathers.name, models=models.name,
            sha256=dict(gathers=hashlib.sha256(gathers.read_bytes()).hexdigest(),
                        models=hashlib.sha256(models.read_bytes()).hexdigest()))})

    def test_empty_glob_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'No checkpoints matched'):
            E.select_checkpoints([])

    def test_completed_comparison_allows_models_and_unique_seeds(self):
        paths = [self.save('ours0', checkpoint()), self.save('ours1', checkpoint(seed=1)),
                 self.save('baseline0', checkpoint(model='inversionnet')),
                 self.save('raw0', checkpoint(model='ours_raw'))]
        selected, skipped = E.select_checkpoints(paths)
        self.assertEqual(len(selected), 4)
        self.assertEqual(skipped, [])

    def test_image_models_must_share_physics_inputs(self):
        changed = checkpoint(model='ours_k1')
        changed['config']['physics_fingerprints']['train'] = 'changed-images'
        paths = [self.save('ours', checkpoint()), self.save('changed', changed)]
        with self.assertRaisesRegex(ValueError, 'incompatible physics inputs'):
            E.select_checkpoints(paths)

    def test_raw_models_cannot_claim_hidden_physics_inputs(self):
        raw = checkpoint(model='ours_raw')
        raw['config']['physics_artifact_fingerprint'] = 'unexpected-physics'
        with self.assertRaisesRegex(ValueError, 'raw-only model'):
            E.select_checkpoints([self.save('raw', raw)])

    def test_evaluation_detects_changed_physics_artifacts(self):
        with patch.object(E.T, 'physics_artifact_fingerprint', return_value='changed', create=True):
            with self.assertRaisesRegex(ValueError, 'Physics inputs changed'):
                E.validate_physics_provenance([('ours.pt', checkpoint())], self.root)

    def test_raw_evaluation_does_not_require_physics_files(self):
        with patch.object(E.T, 'physics_artifact_fingerprint', create=True) as fingerprint:
            E.validate_physics_provenance([('raw.pt', checkpoint(model='ours_raw'))], self.root)
        fingerprint.assert_not_called()

    def test_smoke_or_resumable_run_does_not_change_report(self):
        incomplete = checkpoint()
        incomplete.update(kind='resume', status='interrupted', completed_steps=2)
        final = self.save('full', checkpoint())
        selected, skipped = E.select_checkpoints([self.save('smoke', incomplete), final])
        self.assertEqual([path for path, _ in selected], [final])
        self.assertEqual(len(skipped), 1)

    def test_incomplete_final_is_excluded(self):
        bad = checkpoint()
        bad['completed_steps'] = 2
        with self.assertRaisesRegex(ValueError, 'No completed final checkpoints'):
            E.select_checkpoints([self.save('bad', bad)])

    def test_duplicate_seed_cannot_be_averaged_as_another_run(self):
        paths = [self.save('full', checkpoint()), self.save('duplicate_tag', checkpoint())]
        with self.assertRaisesRegex(ValueError, 'duplicate model/loss/seed'):
            E.select_checkpoints(paths)

    def test_different_training_budget_cannot_be_pooled(self):
        smoke = checkpoint(seed=1)
        smoke['steps'] = smoke['completed_steps'] = smoke['config']['steps'] = 2
        paths = [self.save('full', checkpoint()), self.save('smoke', smoke)]
        with self.assertRaisesRegex(ValueError, 'incompatible.*steps'):
            E.select_checkpoints(paths)

    def test_different_data_cannot_be_pooled(self):
        changed = checkpoint(seed=1)
        changed['config']['data_fingerprints']['train'] = 'different-data'
        paths = [self.save('full', checkpoint()), self.save('changed', changed)]
        with self.assertRaisesRegex(ValueError, 'incompatible.*data_fingerprints'):
            E.select_checkpoints(paths)

    def test_explicit_experiment_selection(self):
        pilot = self.save('pilot', checkpoint())
        full = self.save('full', checkpoint(experiment='full'))
        with self.assertRaisesRegex(ValueError, 'Multiple experiments'):
            E.select_checkpoints([pilot, full])
        selected, _ = E.select_checkpoints([pilot, full], experiment='pilot')
        self.assertEqual([path for path, _ in selected], [pilot])
        with self.assertRaisesRegex(ValueError, 'No completed final checkpoints'):
            E.select_checkpoints([pilot], experiment='missing')

    def test_scores_keep_precision_until_table_formatting(self):
        row = dict(model='ours', loss='l1ssim', family='a', mae=.123456789,
                   rmse=.234567891, ssim=.987654321)
        text = E.write_table([row], ['a'], self.root / 'table.md')
        self.assertIn('0.1235', text)
        self.assertEqual(row['mae'], .123456789)
        self.assertIn('RMSE', text)

    def test_evaluation_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'batch size'):
            E.predict(checkpoint(), dict(y=np.zeros((1, 70, 70))), batch_size=0)

    def test_changed_manifest_cannot_redefine_the_heldout_split(self):
        manifest = dict(n_train=452, n_val=24, n_total=500)
        with patch.object(E.T, 'read_manifest', return_value=manifest):
            with self.assertRaisesRegex(ValueError, 'split differs'):
                E.validate_dataset_provenance(checkpoint(), self.root)

    def test_changed_manifest_fingerprint_is_rejected(self):
        manifest = dict(n_train=400, n_val=52, n_total=500)
        with patch.object(E.T, 'read_manifest', return_value=manifest), \
                patch.object(E.T, 'manifest_fingerprint', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                E.validate_dataset_provenance(checkpoint(), self.root)

    def test_raw_data_swap_with_unchanged_manifest_is_rejected(self):
        manifest = self.data_manifest()
        ck = checkpoint(model='ours_raw')
        ck['config']['manifest_fingerprint'] = E.T.manifest_fingerprint(manifest)
        with patch.object(E.T, 'read_manifest', return_value=manifest):
            E.validate_dataset_provenance(ck, self.root)
            # Same shape and dtype, finite values, unchanged manifest and paths.
            np.save(self.root / 'gathers.npy', np.full((1, 5, 2, 2), 2., dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'gathers file checksum changed'):
                E.validate_dataset_provenance(ck, self.root)

    def test_raw_data_without_recorded_file_identity_is_rejected(self):
        manifest = self.data_manifest()
        del manifest['families']['flatvel-a']['sha256']
        ck = checkpoint(model='inversionnet')
        ck['config']['manifest_fingerprint'] = E.T.manifest_fingerprint(manifest)
        with patch.object(E.T, 'read_manifest', return_value=manifest):
            with self.assertRaisesRegex(ValueError, 'missing gathers file checksum'):
                E.validate_dataset_provenance(ck, self.root)

    def test_cli_keeps_validation_separate_and_forwards_pilot_controls(self):
        path = self.save('ours', checkpoint())
        data = dict(y=np.zeros((1, 11, 11), dtype=np.float32),
                    family=np.array(['flatvel-a']), index=np.array([400]))
        manifest = self.data_manifest()
        arguments = ['4_test.py', '--checkpoints', path, '--data-root', str(self.root),
                     '--split', 'val', '--device', 'cpu', '--batch', '3', '--experiment', 'pilot',
                     '--families', 'flatvel-a', '--limit-per-family', '1']
        with patch.object(sys, 'argv', arguments), \
                patch.object(E.T, 'read_manifest', return_value=manifest), \
                patch.object(E.T, 'manifest_fingerprint', return_value='manifest-hash'), \
                patch.object(E.T, 'physics_artifact_fingerprint', return_value='physics-artifacts', create=True), \
                patch.object(E.T, 'load_data', return_value=data) as loader, \
                patch.object(E, 'predict', return_value=np.full((1, 11, 11), 3000.001)) as predict, \
                contextlib.redirect_stdout(io.StringIO()):
            E.main()
        loader.assert_called_once_with('val', 'flatvel-a', images=True, data_root=str(self.root),
                                       limit_per_family=1)
        self.assertEqual(predict.call_args.args[2], 3)
        self.assertEqual(str(E.T.DEV), 'cpu')
        results = self.root / 'results' / 'val'
        self.assertTrue((results / 'table.md').exists())
        row = json.loads((results / 'scores.jsonl').read_text())
        self.assertGreater(row['mae'], 0)
        self.assertLess(row['mae'], .000001)
        self.assertEqual(row['split'], 'val')
        self.assertFalse((self.root / 'results' / 'test').exists())


if __name__ == '__main__':
    unittest.main()

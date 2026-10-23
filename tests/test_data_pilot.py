import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load('data_step', 'openfwi/1_data.py')
P = load('prepare_pilot', 'openfwi/tools/prepare_pilot.py')


class DataPilotTests(unittest.TestCase):
    def test_preparation_records_file_identity_and_preserves_original_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'; (root / 'data').mkdir(parents=True)
            np.save(root / 'data/g.npy', np.ones((4, 5, 1000, 70), np.float32))
            np.save(root / 'data/m.npy', np.full((4, 1, 70, 70), 2000, np.float32))
            source = dict(n_train=2, n_val=1, n_total=4, families={'flat': dict(
                gathers='data/g.npy', models='data/m.npy', source_indices=[4, 5, 400, 452])})
            (root / 'data/manifest.json').write_text(json.dumps(source))
            output = Path(tmp) / 'pilot'
            manifest = P.prepare(output, root, n_train=2, n_val=1, n_test=1)
            entry = manifest['families']['flat']
            self.assertEqual(entry['source_indices'], [4, 5, 400, 452])
            for key in ('gathers', 'models'):
                with (output / entry[key]).open('rb') as stream:
                    self.assertEqual(entry['sha256'][key], hashlib.file_digest(stream, 'sha256').hexdigest())
            with self.assertRaisesRegex(ValueError, 'new directory'):
                P.prepare(output, root, n_train=2, n_val=1, n_test=1)

    def test_subset_keeps_original_partition_membership(self):
        ids = P.subset_indices(dict(n_train=400, n_val=52, n_total=500), 3, 2, 2)
        np.testing.assert_array_equal(ids, [0, 1, 2, 400, 401, 452, 453])

    def test_subset_refuses_missing_validation_and_oversized_request(self):
        with self.assertRaises(KeyError):
            P.subset_indices(dict(n_train=452, n_total=500), 2, 2, 2)
        with self.assertRaises(ValueError):
            P.subset_indices(dict(n_train=4, n_val=2, n_total=8), 5, 2, 2)

    def test_check_pair_rejects_nonfinite_and_bad_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, m = Path(tmp) / 'g.npy', Path(tmp) / 'm.npy'
            gathers = np.ones((2, 5, 1000, 70), np.float32)
            models = np.full((2, 1, 70, 70), 2000, np.float32)
            np.save(g, gathers); np.save(m, models)
            self.assertEqual(set(D.check_pair(g, m, 2)), {'gathers', 'models'})
            gathers[-1, -1, -1, -1] = np.nan
            np.save(g, gathers)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                D.check_pair(g, m, 2)
            np.save(g, np.ones((2, 5, 1000, 70), np.float32))
            np.save(m, models[:, 0])
            with self.assertRaisesRegex(ValueError, 'shapes'):
                D.check_pair(g, m, 2)


if __name__ == '__main__':
    unittest.main()

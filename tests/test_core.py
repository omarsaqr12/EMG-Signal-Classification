"""Fast, deterministic checks that use synthetic signals, not participant data."""

import unittest

import numpy as np

from emg.classification import leave_one_out_accuracy
from emg.features import extract_feature_matrix, feature_names
from emg.preprocessing import apply_filter
from emg.segmentation import segment_trials
from scripts.run_fixed_k_sensitivity import evaluate


class PipelineTests(unittest.TestCase):
    def test_segment_trials_drops_short_runs_and_preserves_labels(self):
        signal = np.arange(20, dtype=float).reshape(10, 2)
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        trials = segment_trials(signal, labels, window_samples=3)
        self.assertEqual([trial.label for trial in trials], [0, 1])
        np.testing.assert_array_equal(trials[0].data, signal[:3])
        np.testing.assert_array_equal(trials[1].data, signal[3:6])

    def test_feature_dimensions_and_finite_values(self):
        signal = np.zeros((8, 2), dtype=float)
        trials = segment_trials(signal, np.zeros(8, dtype=int), window_samples=8)
        for domain, width in (("time", 10), ("frequency", 8), ("combined", 18)):
            with self.subTest(domain=domain):
                features, labels = extract_feature_matrix(trials, fs=100, domain=domain)
                self.assertEqual(features.shape, (1, width))
                self.assertEqual(len(feature_names(2, domain=domain)), width)
                self.assertTrue(np.isfinite(features).all())
                np.testing.assert_array_equal(labels, [0])

    def test_none_filter_preserves_signal(self):
        signal = np.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_array_equal(apply_filter(signal, kind="none"), signal)

    def test_fixed_k_knn_on_separable_toy_data(self):
        features = np.array([[0.0], [0.1], [10.0], [10.1]])
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(leave_one_out_accuracy(features, labels, name="knn", k=1), 1.0)

    def test_fixed_k_rejects_nonpositive_k(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate(0, subjects=())


if __name__ == "__main__":
    unittest.main()

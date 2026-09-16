"""Tests for honest within-subject nested model selection."""

import unittest

import numpy as np

from scripts.run_nested_cv import nested_accuracy


class NestedCVTests(unittest.TestCase):
    def test_separable_three_class_problem(self):
        X = np.array([[offset + 0.01 * i] for offset in (0, 10, 20)
                      for i in range(6)], dtype=float)
        y = np.repeat([0, 1, 2], 6)
        score, chosen = nested_accuracy(X, y, k_values=(1, 3),
                                        outer_splits=3, inner_splits=2)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(set(chosen) <= {1, 3})

    def test_rejects_too_few_trials_per_class(self):
        with self.assertRaisesRegex(ValueError, "outer_splits"):
            nested_accuracy(np.arange(7).reshape(-1, 1),
                            np.array([0, 0, 0, 0, 0, 1, 1]))

    def test_rejects_k_larger_than_inner_training_fold(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.repeat([0, 1], 10)
        with self.assertRaisesRegex(ValueError, "minimum inner training size"):
            nested_accuracy(X, y, k_values=(100,), outer_splits=5, inner_splits=3)

    def test_rejects_nan_features(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        X[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            nested_accuracy(X, np.repeat([0, 1], 10))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from multiclass_intrusion import CLASSES, compute_roc_metrics


class RocAucTests(unittest.TestCase):
    def test_perfect_scores_with_reordered_model_classes(self):
        order = list(reversed(CLASSES))
        truth = CLASSES * 2
        scores = np.array([[float(c == y) for c in order] for y in truth])
        summary, rows = compute_roc_metrics(truth, scores, order)
        self.assertEqual(summary['roc_auc_ovr_macro'], 1.0)
        self.assertEqual(summary['roc_auc_defined_classes'], 10)
        self.assertTrue(all(r['roc_auc_ovr'] == 1 for r in rows))

    def test_agrees_with_sklearn_multiclass_macro(self):
        order = sorted(CLASSES)
        truth = np.array(order * 3)
        scores = np.random.default_rng(42).random((30, 10))
        scores /= scores.sum(axis=1, keepdims=True)
        summary, _ = compute_roc_metrics(truth, scores, order)
        expected = roc_auc_score(truth, scores, labels=order, multi_class='ovr', average='macro')
        self.assertAlmostEqual(summary['roc_auc_ovr_macro'], expected)

    def test_missing_classes_are_explicit(self):
        summary, rows = compute_roc_metrics(['Normal', 'DoS'], [[.9, .1], [.2, .8]], ['Normal', 'DoS'])
        self.assertIsNone(summary['roc_auc_ovr_macro'])
        self.assertEqual(summary['roc_auc_ovr_macro_supported'], 1)
        self.assertEqual(summary['roc_auc_defined_classes'], 2)
        self.assertEqual(next(r for r in rows if r['class'] == 'Worms')['status'], 'no_positive_examples')

    def test_single_class_has_no_valid_auc(self):
        summary, _ = compute_roc_metrics(['Normal'], [[1]], ['Normal'])
        self.assertIsNone(summary['roc_auc_ovr_macro_supported'])
        self.assertEqual(summary['roc_auc_defined_classes'], 0)

    def test_unlearned_class_and_invalid_probabilities(self):
        summary, rows = compute_roc_metrics(['Normal', 'DoS'], [[1], [1]], ['Normal'])
        self.assertEqual(next(r for r in rows if r['class'] == 'DoS')['status'], 'class_not_learned')
        with self.assertRaises(ValueError):
            compute_roc_metrics(['Normal'], [[np.nan]], ['Normal'])

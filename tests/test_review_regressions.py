import tempfile
from pathlib import Path
import unittest
import pandas as pd
import multiclass_intrusion as mi
from dashboard import safe_csv
from create_report import generate_pdf


class ReviewRegressions(unittest.TestCase):
    def test_mangled_target_header_rejected(self):
        frame = pd.DataFrame({'dur': [1], 'label.1': [1]})
        with self.assertRaisesRegex(ValueError, 'leak labels'):
            mi.get_predictor_columns(frame, frame)

    def test_infinite_predictor_rejected(self):
        train = pd.DataFrame({'dur': [1, float('inf')], 'attack_cat': ['Normal', 'DoS']})
        test = pd.DataFrame({'dur': [3], 'attack_cat': ['Normal']})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'infinite predictors'):
                mi.run(train, test, folder, True)

    def test_missing_predictors_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing training'):
            mi.get_predictor_columns(pd.DataFrame({'a': [1], 'b': [2]}), pd.DataFrame({'a': [1]}))

    def test_duplicate_index_does_not_drop_training_rows(self):
        frame = pd.DataFrame({'dur': range(10), 'attack_cat': ['Normal'] * 10}, index=[0] * 10)
        train, val, _ = mi.stratified_train_val_split(frame)
        self.assertEqual(len(train) + len(val), 10)
        self.assertFalse(set(train.dur) & set(val.dur))

    def test_sampling_exact_size_and_rare_classes(self):
        frame = pd.DataFrame({'attack_cat': ['Normal'] * 103 + ['Worms'] * 3 + ['DoS'] * 17})
        for limit in (6, 7, 11, 37):
            sample = mi.stratified_subset(frame, limit)
            self.assertEqual(len(sample), limit)
            self.assertGreaterEqual(sample.attack_cat.value_counts().min(), 2)
        with self.assertRaises(ValueError):
            mi.stratified_subset(frame, 2)

    def test_missing_labels_are_not_normal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            pd.DataFrame({'dur': [1, 2], 'attack_cat': ['Normal', None]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing or unsupported'):
                mi.load_original(path, path, 0, 0)

    def test_overlap_ignores_id_and_label(self):
        train = pd.DataFrame({'dur': [1, 2], 'id': [1, 2], 'attack_cat': ['Normal', 'DoS']})
        test = pd.DataFrame({'dur': [2], 'id': [99], 'attack_cat': ['Normal']})
        with self.assertRaisesRegex(ValueError, 'both train and test'):
            mi.check_overlap(train, test, mi.get_predictor_columns(train, test))

    def test_macro_metric_uses_fixed_ten_classes(self):
        metrics = mi.compute_metrics(['Normal'], ['Normal'], 'fixture', 'validation')
        self.assertAlmostEqual(metrics['macro_f1'], 0.1)

    def test_empty_validation_has_actionable_error(self):
        train = pd.DataFrame({'dur': [1, 2], 'attack_cat': ['Normal', 'DoS']})
        test = pd.DataFrame({'dur': [3], 'attack_cat': ['Normal']})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'No validation rows'):
                mi.run(train, test, folder, True)

    def test_csv_formula_neutralized_without_changing_numbers(self):
        data = pd.DataFrame({'=heading': ['  =HYPERLINK("x")'], 'value': [-2.5]})
        result = safe_csv(data)
        self.assertIn("'=heading", result)
        self.assertIn("'  =HYPERLINK", result)
        self.assertIn('-2.5', result)
        self.assertEqual(data.columns[0], '=heading')

    def test_srn_cannot_escape_output_folder(self):
        for value in ('../outside', 'C:/outside', '<img src="x"/>'):
            with self.assertRaises(ValueError):
                generate_pdf(value)

"""Staged training, inference contracts and synchronized GitHub Pages assets."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import multiclass_intrusion as mi
from website_core import ROOT, FEATURE_GROUPS, predict_flows, prediction_inputs, read_prediction_csv


class WebsiteFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        train, test = mi.make_smoke_data()
        cls.train = mi.stratified_subset(train, 400)
        cls.test = mi.stratified_subset(test, 150)
        cls.events = []
        fitted = mi.fit_experiment(cls.train, cls.test, cls.folder.name, smoke=True,
                                   progress=lambda n, text: cls.events.append((n, text)))
        cls.outputs_before_testing = {p.name for p in Path(cls.folder.name).iterdir()}
        cls.meta, cls.model = mi.evaluate_experiment(fitted)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_predictions_use_the_evaluated_pipeline_without_target_leakage(self):
        sample = self.test.head(8)
        results, scores = predict_flows(self.model, self.train, sample)
        np.testing.assert_array_equal(results.predicted_attack, self.model.predict(prediction_inputs(sample, self.train)))
        np.testing.assert_allclose(scores.drop(columns='record').sum(axis=1), 1)
        changed = sample.assign(attack_cat='Normal', label=1, id=999)
        again, _ = predict_flows(self.model, self.train, changed)
        np.testing.assert_array_equal(results.predicted_attack, again.predicted_attack)
        self.assertEqual(self.meta['predictor_columns'], mi.get_predictor_columns(self.train, self.test))
        self.assertNotIn('attack_cat', self.meta['predictor_columns'])

    def test_invalid_uploads_and_measurements_rejected(self):
        for data in (b'dur,dur\n1,2', b'dur, DUR\n1,2'):
            with self.assertRaisesRegex(ValueError, 'unique'):
                read_prediction_csv(data)
        with self.assertRaisesRegex(ValueError, '1,000'):
            read_prediction_csv(('dur\n' + '1\n' * 1001).encode())
        with self.assertRaises(ValueError):
            prediction_inputs(self.test.drop(columns='proto'), self.train)
        with self.assertRaisesRegex(ValueError, 'finite numbers'):
            prediction_inputs(self.test.head(1).assign(feature_00='not a number'), self.train)
        with self.assertRaisesRegex(ValueError, 'finite numbers'):
            prediction_inputs(self.test.head(1).assign(feature_00=float('inf')), self.train)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            prediction_inputs(self.test.head(0), self.train)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            prediction_inputs(self.test.head(1).assign(extra_feature=1), self.train)

    def test_blank_and_unknown_categorical_values_supported(self):
        frame = self.test.head(2).copy()
        frame.loc[frame.index[0], 'proto'] = 'previously-unseen-protocol'
        frame.loc[frame.index[1], 'proto'] = np.nan
        frame.loc[frame.index[0], 'feature_00'] = np.nan
        results, _ = predict_flows(self.model, self.train, frame.drop(columns='attack_cat'))
        self.assertEqual(len(results), 2)
        self.assertNotIn('actual_attack', results)

    def test_training_does_not_evaluate_heldout_test_and_reports_real_progress(self):
        self.assertIn('validation_leaderboard.csv', self.outputs_before_testing)
        self.assertNotIn('heldout_test_metrics.csv', self.outputs_before_testing)
        self.assertNotIn('run_metadata.json', self.outputs_before_testing)
        self.assertEqual(self.events[-1][0], 1.0)
        self.assertEqual([n for n, _ in self.events], sorted(n for n, _ in self.events))
        for name in mi.model_choices():
            self.assertTrue(any('Training ' + name in message for _, message in self.events))
        self.assertTrue((Path(self.folder.name) / 'heldout_test_metrics.csv').is_file())

    def test_browser_core_copies_match_executed_code(self):
        for name in ['multiclass_intrusion.py', 'website_core.py']:
            self.assertEqual((ROOT / name).read_bytes(), (ROOT / 'docs/runtime' / name).read_bytes(),
                             'Run python tools/build_website.py after core edits.')

    def test_model_page_covers_the_actual_42_features(self):
        header = pd.read_csv(ROOT / 'reference/data/UNSW_NB15_training_clean.csv', nrows=0)
        predictors = mi.get_predictor_columns(header, header)
        documented = [name for names, _ in FEATURE_GROUPS.values() for name in names.split()]
        self.assertEqual(len(predictors), 42)
        self.assertEqual(set(predictors), set(documented))
        self.assertEqual(len(documented), len(set(documented)))
        preview = json.loads((ROOT / 'docs/site-data.json').read_text(encoding='utf-8'))
        self.assertEqual({f['feature'] for f in preview['features']}, set(predictors))
        self.assertEqual(preview['train_rows'], 10000)
        self.assertEqual(preview['test_rows'], 5000)


if __name__ == '__main__':
    unittest.main()

"""Inference contracts and the actual Streamlit Predict button flow."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import multiclass_intrusion as mi
import website
from website_core import ROOT, predict_flows, prediction_inputs, read_prediction_csv


class WebsiteFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        train, test = mi.make_smoke_data()
        cls.train = mi.stratified_subset(train, 400)
        cls.test = mi.stratified_subset(test, 150)
        cls.meta, cls.model = mi.run_experiment(cls.train, cls.test, cls.folder.name, smoke=True)

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

    def test_predict_button_and_stale_result_reset(self):
        with patch.object(website, 'reference_data', return_value=(self.train, self.test)), \
             patch.object(website, 'experiment', return_value=(self.meta, self.model)):
            app = AppTest.from_string('from website import render_predict\nrender_predict()').run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn('Prediction complete', app.success[0].value)
            results = app.session_state['prediction_output'][0]
            self.assertEqual(len(results), 10)
            self.assertEqual(set(results.predicted_attack) - set(mi.CLASSES), set())
            app.selectbox[0].select('Worms').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.success), 0)

    def test_results_page_contains_evaluation_and_downloads(self):
        with patch.object(website, 'experiment', return_value=(self.meta, self.model)), \
             patch.object(website, 'OUTPUT', Path(self.folder.name)):
            app = AppTest.from_string('from website import render_results\nrender_results()')
            app.session_state['experiment_ready'] = True
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], [
                'Model comparison', 'Held-out evaluation', 'Feature importance', 'Alert queue', 'Downloads'])
            self.assertTrue(any(metric.label == 'Held-out Macro F1' for metric in app.metric))

    def test_model_page_covers_the_actual_42_features(self):
        header = pd.read_csv(ROOT / 'reference/data/UNSW_NB15_training_clean.csv', nrows=0)
        predictors = mi.get_predictor_columns(header, header)
        documented = [name for names, _ in website.FEATURE_GROUPS.values() for name in names.split()]
        self.assertEqual(len(predictors), 42)
        self.assertEqual(set(predictors), set(documented))
        self.assertEqual(len(documented), len(set(documented)))
        app = AppTest.from_string('from website import render_model\nrender_model()').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 42)


if __name__ == '__main__':
    unittest.main()

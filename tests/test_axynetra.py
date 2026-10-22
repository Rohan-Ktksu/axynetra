"""Automated unit & integration tests for the Axynetra multi-class intrusion classification pipeline."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import multiclass_intrusion as mi
import dashboard as dash


class TestAxynetraPipeline(unittest.TestCase):
    def test_1_class_normalization(self):
        """Verify 10 target labels and normalization of Backdoors -> Backdoor and whitespace/case."""
        self.assertEqual(len(mi.CLASSES), 10)
        self.assertEqual(mi.normalized_label("Backdoors"), "Backdoor")
        self.assertEqual(mi.normalized_label(" backdoors "), "Backdoor")
        self.assertEqual(mi.normalized_label("Backdoor"), "Backdoor")
        self.assertEqual(mi.normalized_label("normal"), "Normal")
        self.assertEqual(mi.normalized_label(" NORMAL "), "Normal")
        self.assertEqual(mi.normalized_label("dos"), "DoS")
        self.assertEqual(mi.normalized_label("reconnaissance"), "Reconnaissance")
        for cls_name in mi.CLASSES:
            self.assertEqual(mi.normalized_label(cls_name.lower()), cls_name)

    def test_2_dataset_schema_and_leakage_exclusions(self):
        """Verify attack_cat schema check and strict exclusion of attack_cat, label, and id."""
        train_df = pd.DataFrame({
            "id": [1, 2, 3],
            "dur": [0.1, 0.2, 0.3],
            "proto": ["tcp", "udp", "tcp"],
            "label": [0, 1, 1],
            "attack_cat": ["Normal", "Backdoors", "DoS"],
        })
        test_df = train_df.copy()
        predictors = mi.get_predictor_columns(train_df, test_df)
        self.assertEqual(predictors, ["dur", "proto"])
        for forbidden in ("attack_cat", "label", "id"):
            self.assertNotIn(forbidden, predictors)

        # Missing attack_cat must raise ValueError in load_original
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = Path(tmpdir) / "bad.csv"
            pd.DataFrame({"dur": [1.0, 2.0]}).to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                mi.load_original(bad_csv, bad_csv, max_train=0, max_test=0)

    def test_3_smoke_data_generation(self):
        """Verify synthetic smoke data shape, columns, and presence of all 10 classes."""
        train_df, test_df = mi.make_smoke_data()
        self.assertEqual(len(train_df), 1700)
        self.assertEqual(len(test_df), 500)
        self.assertIn("attack_cat", train_df.columns)
        self.assertIn("proto", train_df.columns)
        self.assertIn("service", train_df.columns)
        self.assertEqual(set(train_df["attack_cat"].unique()), set(mi.CLASSES))
        self.assertEqual(set(test_df["attack_cat"].unique()), set(mi.CLASSES))

    def test_4_stratified_validation_split_and_rare_class_fallback(self):
        """Verify rare classes with >=2 rows appear in both train_sub and val_sub, and 1-row fallback is logged."""
        rows = []
        # 9 classes with 10 rows each, plus 'Worms' with only 2 rows, plus a 1-row ultra-rare test class
        for c in mi.CLASSES[:-1]:
            for i in range(10):
                rows.append({"f1": float(i), "attack_cat": c})
        rows.append({"f1": 99.0, "attack_cat": "Worms"})
        rows.append({"f1": 100.0, "attack_cat": "Worms"})
        df = pd.DataFrame(rows)

        train_sub, val_sub, meta = mi.stratified_train_val_split(df, val_fraction=0.20)
        self.assertIn("Worms", set(train_sub["attack_cat"]))
        self.assertIn("Worms", set(val_sub["attack_cat"]))
        self.assertEqual(len(meta["rare_class_fallbacks"]), 0)

        # Now test fallback when a class has only 1 sample
        df_single = pd.concat([
            df[df["attack_cat"] != "Worms"],
            pd.DataFrame([{"f1": 55.0, "attack_cat": "Worms"}])
        ], ignore_index=True)
        tr_s, val_s, meta_s = mi.stratified_train_val_split(df_single, val_fraction=0.20)
        self.assertIn("Worms", set(tr_s["attack_cat"]))
        self.assertNotIn("Worms", set(val_s["attack_cat"]))
        self.assertEqual(len(meta_s["rare_class_fallbacks"]), 1)
        self.assertIn("Worms", meta_s["rare_class_fallbacks"][0])

    def test_5_preprocessor_fitted_only_on_train(self):
        """Verify numeric median/scaler and categorical encoder are fitted strictly on training data."""
        X_train = pd.DataFrame({
            "num": [10.0, 20.0, 30.0],
            "cat": ["tcp", "udp", "tcp"],
        })
        X_test_unseen = pd.DataFrame({
            "num": [1000.0, np.nan],
            "cat": ["icmp_unseen", "tcp"],
        })
        pre = mi.build_preprocessor(X_train)
        pre.fit(X_train)
        # Median of X_train['num'] must be 20.0, unaffected by 1000.0 in X_test_unseen
        num_imputer = pre.named_transformers_["numeric"].named_steps["impute"]
        self.assertAlmostEqual(float(num_imputer.statistics_[0]), 20.0)
        # Transforming unseen category must succeed without raising error
        transformed = pre.transform(X_test_unseen)
        self.assertEqual(transformed.shape[0], 2)

    def test_6_pipeline_execution_artifacts_and_val_test_separation(self):
        """Verify end-to-end execution, expected artifacts, validation selection, and dashboard loading."""
        train_df, test_df = mi.make_smoke_data()
        # Use a fast stratified subset for unit testing
        train_small = mi.stratified_subset(train_df, 400)
        test_small = mi.stratified_subset(test_df, 150)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            meta = mi.run(train_small, test_small, out_dir, smoke=True, val_fraction=0.20)

            expected_files = [
                "model_comparison.csv",
                "validation_leaderboard.csv",
                "heldout_test_metrics.csv",
                "classification_report.txt",
                "confusion_matrix.png",
                "feature_importance.csv",
                "feature_importance.png",
                "example_soc_alerts.csv",
                "run_metadata.json",
                "validation_roc_auc.csv",
                "heldout_test_roc_auc.csv",
                "roc_curves.png",
            ]
            for fname in expected_files:
                self.assertTrue((out_dir / fname).exists(), f"Missing expected artifact: {fname}")

            # Verify validation vs held-out test separation
            val_df = pd.read_csv(out_dir / "validation_leaderboard.csv")
            test_df_metrics = pd.read_csv(out_dir / "heldout_test_metrics.csv")
            self.assertEqual(len(val_df), 4)
            self.assertTrue((val_df["split"] == "validation").all())
            self.assertEqual(len(test_df_metrics), 1)
            self.assertEqual(test_df_metrics.iloc[0]["split"], "heldout_test")

            # Winner must be the model with the highest validation macro_f1
            expected_winner = val_df.sort_values("macro_f1", ascending=False).iloc[0]["model"]
            self.assertEqual(meta["best_model"], expected_winner)
            self.assertEqual(test_df_metrics.iloc[0]["model"], expected_winner)
            self.assertEqual(meta["fit_train_rows"] + meta["val_rows"], meta["train_rows"])
            self.assertEqual(len(meta['heldout_test_roc_auc']), 10)
            self.assertEqual(meta['heldout_test_evaluation']['roc_auc_defined_classes'], 10)
            self.assertTrue(0 <= meta['heldout_test_evaluation']['roc_auc_ovr_macro'] <= 1)

            # Verify dashboard artifact loading helper
            bundle = dash.load_dashboard_artifacts(out_dir)
            self.assertTrue(bundle["smoke"])
            self.assertEqual(len(bundle["leaderboard_df"]), 4)
            self.assertEqual(len(bundle["test_metrics_df"]), 1)
            self.assertFalse(bundle["alerts_df"].empty)
            self.assertIn("top_feature_deviations_illustrative", bundle["alerts_df"].columns)


if __name__ == "__main__":
    unittest.main()

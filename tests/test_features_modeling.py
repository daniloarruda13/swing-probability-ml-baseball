import importlib.util
import unittest


HAS_ANALYSIS_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "pandas", "sklearn")
)


@unittest.skipUnless(HAS_ANALYSIS_DEPS, "requires numpy, pandas, and scikit-learn")
class FeatureAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd

        from swing_probability.features import (
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
            PLAYER_ID_FEATURES,
        )

        rng = np.random.default_rng(42)
        rows = 180
        data = {
            column: rng.normal(size=rows)
            for column in NUMERIC_FEATURES
        }
        data.update(
            {
                "balls": rng.integers(0, 4, rows),
                "strikes": rng.integers(0, 3, rows),
                "outs_when_up": rng.integers(0, 3, rows),
                "inning": rng.integers(1, 10, rows),
                "zone": rng.integers(1, 15, rows),
                "bat_score": rng.integers(0, 10, rows),
                "fld_score": rng.integers(0, 10, rows),
                "sz_top": rng.normal(3.4, 0.1, rows),
                "sz_bot": rng.normal(1.6, 0.1, rows),
                "plate_x": rng.normal(0, 0.8, rows),
                "plate_z": rng.normal(2.5, 0.8, rows),
                "pitch_type": rng.choice(["FF", "SL", "CH"], rows),
                "stand": rng.choice(["L", "R"], rows),
                "p_throws": rng.choice(["L", "R"], rows),
                "inning_topbot": rng.choice(["Top", "Bot"], rows),
                "batter": rng.integers(100, 120, rows),
                "pitcher": rng.integers(200, 230, rows),
            }
        )
        latent = data["plate_x"] + 0.6 * data["strikes"] - 0.2 * data["balls"]
        data["description"] = np.where(latent > 0.3, "foul", "called_strike")
        cls.frame = pd.DataFrame(data)
        cls.frame.loc[3, "release_speed"] = np.nan
        cls.frame.loc[5, "pitch_type"] = None
        cls.player_columns = PLAYER_ID_FEATURES
        cls.categorical_columns = CATEGORICAL_FEATURES

    def test_preparation_excludes_ids_by_default_and_labels_target(self):
        from swing_probability.features import feature_columns, prepare_pitch_data

        features, target = prepare_pitch_data(self.frame)
        self.assertEqual(tuple(features.columns), feature_columns())
        self.assertFalse(set(self.player_columns).intersection(features.columns))
        self.assertEqual(set(target.unique()), {0, 1})
        self.assertEqual(target.name, "swing")

    def test_optional_player_ids_are_categorical_strings(self):
        from swing_probability.features import prepare_pitch_data

        features, _ = prepare_pitch_data(self.frame, include_player_ids=True)
        for column in self.player_columns:
            self.assertEqual(str(features[column].dtype), "string")

    def test_missing_schema_is_reported_together(self):
        from swing_probability.features import prepare_pitch_data

        with self.assertRaisesRegex(ValueError, "description.*release_speed"):
            prepare_pitch_data(self.frame.drop(columns=["description", "release_speed"]))

    def test_middle_middle_filter_drops_missing_geometry(self):
        import numpy as np

        from swing_probability.features import middle_middle_pitches

        frame = self.frame.iloc[:4].copy()
        frame.loc[:, "plate_x"] = [0.0, 0.49, 0.5, np.nan]
        frame.loc[:, "plate_z"] = (frame["sz_top"] + frame["sz_bot"]) / 2
        result = middle_middle_pitches(frame)
        self.assertEqual(result.index.tolist(), [0, 1])
        self.assertTrue(result["center_strike_zone"].notna().all())

    def test_pipeline_handles_missing_and_unseen_categories(self):
        import numpy as np

        from swing_probability.features import prepare_pitch_data
        from swing_probability.modeling import evaluate_probabilities, make_baseline_pipeline

        train = self.frame.iloc[:140].copy()
        test = self.frame.iloc[140:].copy()
        test.loc[:, "pitch_type"] = "UNSEEN"
        train_features, train_target = prepare_pitch_data(train)
        test_features, test_target = prepare_pitch_data(test)
        model = make_baseline_pipeline()
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(test_features)[:, 1]
        self.assertEqual(probabilities.shape, (40,))
        self.assertTrue(np.isfinite(probabilities).all())
        metrics = evaluate_probabilities(test_target, probabilities)
        self.assertEqual(set(metrics), {"accuracy", "brier_score", "roc_auc"})

    def test_probability_metrics_have_expected_values(self):
        from swing_probability.modeling import evaluate_probabilities

        metrics = evaluate_probabilities([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.6])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.0925)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_probability_validation_rejects_bad_inputs(self):
        from swing_probability.modeling import evaluate_probabilities

        with self.assertRaises(ValueError):
            evaluate_probabilities([0, 1], [0.2, 1.2])
        with self.assertRaises(ValueError):
            evaluate_probabilities([1, 1], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()

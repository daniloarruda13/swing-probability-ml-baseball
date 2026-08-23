import unittest

from swing_probability.labels import (
    SWING_DESCRIPTIONS,
    TAKE_DESCRIPTIONS,
    classify_description,
    label_descriptions,
)


class LabelTests(unittest.TestCase):
    def test_all_declared_swings_are_positive(self):
        self.assertTrue(SWING_DESCRIPTIONS)
        self.assertEqual(
            label_descriptions(sorted(SWING_DESCRIPTIONS)),
            (1,) * len(SWING_DESCRIPTIONS),
        )

    def test_all_declared_takes_are_negative(self):
        self.assertTrue(TAKE_DESCRIPTIONS)
        self.assertEqual(
            label_descriptions(sorted(TAKE_DESCRIPTIONS)),
            (0,) * len(TAKE_DESCRIPTIONS),
        )

    def test_observed_bunt_outcomes_are_swings(self):
        for description in ("foul_bunt", "missed_bunt", "bunt_foul_tip"):
            with self.subTest(description=description):
                self.assertEqual(classify_description(description), 1)

    def test_every_description_observed_in_notebook_is_classified(self):
        observed = {
            "hit_into_play", "swinging_strike", "ball", "called_strike",
            "foul", "foul_bunt", "blocked_ball", "swinging_strike_blocked",
            "hit_by_pitch", "foul_tip", "missed_bunt", "pitchout",
            "bunt_foul_tip",
        }
        self.assertTrue(observed.issubset(SWING_DESCRIPTIONS | TAKE_DESCRIPTIONS))

    def test_matching_is_trimmed_and_case_insensitive(self):
        self.assertEqual(classify_description("  SWINGING_STRIKE "), 1)
        self.assertEqual(classify_description(" Called_Strike "), 0)

    def test_unknown_and_missing_descriptions_are_rejected(self):
        for description in ("new_provider_value", "", None):
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    classify_description(description)

    def test_bulk_error_reports_position(self):
        with self.assertRaisesRegex(ValueError, "position 1"):
            label_descriptions(["ball", "unknown"])


if __name__ == "__main__":
    unittest.main()

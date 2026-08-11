from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import NUMERIC_FEATURES, TARGET_COL
from src.features import build_pipeline
from src.score_new_transactions import score_dataframe
from src.validation import (
    DataValidationError,
    validate_binary_target,
    validate_scoring_dataframe,
    validate_threshold,
    validate_training_dataframe,
)


class ValidationAndScoringTests(unittest.TestCase):
    def _demo_transactions(self) -> pd.DataFrame:
        """Create a demo transactions dataframe with real dataset columns."""
        return pd.DataFrame(
            {
                "Time": [0, 3600, 7200, 10800, 14400, 18000, 21600, 25200],
                "V1": [-1.3598, 0.4549, -0.5516, 0.3546, -1.4656, 0.2123, -0.8445, 0.1234],
                "V2": [-0.0727, 1.0145, 0.2357, -0.6789, 0.4321, -0.5432, 0.7654, -0.2341],
                "V3": [-0.4202, 0.3218, -0.1234, 0.5432, -0.2345, 0.6543, -0.1234, 0.3456],
                "V4": [0.0846, -0.1235, 0.5432, -0.2341, 0.1234, -0.5432, 0.2341, -0.1234],
                "V5": [-0.0163, 0.7654, -0.0876, 0.3456, -0.2345, 0.1234, -0.3456, 0.5432],
                # For simplicity, create remaining V features as zeros
                "V6": [0.0] * 8,
                "V7": [0.0] * 8,
                "V8": [0.0] * 8,
                "V9": [0.0] * 8,
                "V10": [0.5, -0.3, 0.2, -0.4, 0.1, -0.5, 0.3, -0.2],
                "V11": [0.0] * 8,
                "V12": [0.0] * 8,
                "V13": [0.0] * 8,
                "V14": [0.4, -0.2, 0.3, -0.1, 0.2, -0.4, 0.1, -0.3],
                "V15": [0.0] * 8,
                "V16": [0.0] * 8,
                "V17": [0.0] * 8,
                "V18": [0.0] * 8,
                "V19": [0.0] * 8,
                "V20": [0.0] * 8,
                "V21": [0.0] * 8,
                "V22": [0.0] * 8,
                "V23": [0.0] * 8,
                "V24": [0.0] * 8,
                "V25": [0.0] * 8,
                "V26": [0.0] * 8,
                "V27": [0.0] * 8,
                "V28": [0.0] * 8,
                "Amount": [20.0, 450.0, 35.0, 800.0, 15.0, 900.0, 42.0, 1000.0],
                "Class": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_training_dataframe_validation_accepts_valid_schema(self) -> None:
        df = self._demo_transactions()
        validate_training_dataframe(df)

    def test_scoring_dataframe_does_not_require_target(self) -> None:
        df = self._demo_transactions().drop(columns=[TARGET_COL])
        validate_scoring_dataframe(df)

    def test_missing_required_feature_raises_clear_error(self) -> None:
        df = self._demo_transactions().drop(columns=[NUMERIC_FEATURES[0]])
        with self.assertRaisesRegex(DataValidationError, "missing required columns"):
            validate_scoring_dataframe(df)

    def test_invalid_numeric_feature_raises(self) -> None:
        df = self._demo_transactions()
        df["Amount"] = df["Amount"].astype(object)
        df.loc[0, "Amount"] = "not-a-number"
        with self.assertRaisesRegex(DataValidationError, "invalid numeric"):
            validate_scoring_dataframe(df)

    def test_binary_target_validation_rejects_non_binary_labels(self) -> None:
        df = self._demo_transactions()
        df.loc[0, TARGET_COL] = 2
        with self.assertRaisesRegex(DataValidationError, "binary"):
            validate_binary_target(df)

    def test_threshold_validation_rejects_invalid_values(self) -> None:
        for value in [-0.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(DataValidationError):
                validate_threshold(value)

    def test_score_dataframe_adds_probability_and_flag(self) -> None:
        df = self._demo_transactions()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        model = build_pipeline()
        model.fit(X, y)

        scored = score_dataframe(df, threshold=0.5, model=model)

        self.assertIn("fraud_probability", scored.columns)
        self.assertIn("fraud_flag", scored.columns)
        self.assertIn("reason_codes", scored.columns)
        self.assertEqual(len(scored), len(df))
        self.assertTrue(scored["fraud_probability"].between(0, 1).all())
        self.assertTrue(set(scored["fraud_flag"].unique()).issubset({0, 1}))
        self.assertTrue(
            np.all(
                scored["fraud_probability"].to_numpy()[:-1]
                >= scored["fraud_probability"].to_numpy()[1:]
            )
        )


if __name__ == "__main__":
    unittest.main()

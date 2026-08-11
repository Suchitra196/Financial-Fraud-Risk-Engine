from __future__ import annotations

import unittest

import pandas as pd

from src.reason_codes import (
    add_reason_codes,
    high_amount_cutoff,
    humanize_feature_name,
    reason_codes_for_row,
    shap_reason_codes,
    split_reason_codes,
)


class ReasonCodeTests(unittest.TestCase):
    def _demo_scored(self) -> pd.DataFrame:
        """Create demo scored data with real dataset columns."""
        return pd.DataFrame(
            {
                "Time": [0, 3600, 7200],
                "V1": [-1.36, 0.45, -0.55],
                "V3": [-0.42, 0.32, -0.12],
                "V4": [0.08, -0.12, 0.54],
                "V10": [0.5, -0.3, 0.2],
                "V14": [0.4, -0.2, 0.3],
                "Amount": [25.0, 900.0, 80.0],
                "fraud_probability": [0.05, 0.97, 0.55],
                "fraud_flag": [0, 1, 1],
            }
        )

    def test_high_amount_cutoff_uses_batch_quantile(self) -> None:
        df = self._demo_scored()
        cutoff = high_amount_cutoff(df, quantile=0.50)
        self.assertEqual(cutoff, 80.0)

    def test_reason_codes_for_high_risk_row_are_analyst_friendly(self) -> None:
        row = self._demo_scored().iloc[1]
        reasons = reason_codes_for_row(
            row,
            threshold=0.5,
            amount_cutoff=800.0,
            max_reasons=10,
        )

        joined = " | ".join(reasons).lower()
        self.assertIn("critical model risk", joined)
        self.assertIn("amount", joined)

    def test_add_reason_codes_adds_string_column(self) -> None:
        result = add_reason_codes(self._demo_scored(), threshold=0.5)

        self.assertIn("reason_codes", result.columns)
        self.assertEqual(len(result), 3)
        self.assertTrue(result["reason_codes"].str.len().gt(0).all())

    def test_humanize_feature_name_handles_transformed_names(self) -> None:
        self.assertEqual(
            humanize_feature_name("numeric__V1"),
            "V1 (PCA component 1)",
        )
        self.assertEqual(
            humanize_feature_name("numeric__Amount"),
            "transaction amount",
        )

    def test_shap_reason_codes_use_direction(self) -> None:
        reasons = shap_reason_codes(
            [0.8, -0.4, 0.1],
            [
                "numeric__V1",
                "numeric__V4",
                "numeric__Amount",
            ],
            max_reasons=2,
        )

        self.assertEqual(len(reasons), 2)
        self.assertIn("increased fraud risk", reasons[0])
        self.assertIn("reduced fraud risk", reasons[1])

    def test_split_reason_codes_handles_empty_values(self) -> None:
        self.assertEqual(split_reason_codes(None), [])
        self.assertEqual(
            split_reason_codes("High risk; Large amount"), ["High risk", "Large amount"]
        )


if __name__ == "__main__":
    unittest.main()

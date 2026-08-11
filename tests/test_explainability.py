from __future__ import annotations

import unittest

import numpy as np

from src.reason_codes import positive_class_shap_values

try:
    import shap  # noqa: F401

    HAS_SHAP = True
except Exception:  # pragma: no cover - shap is an optional heavy dependency
    HAS_SHAP = False


class PositiveClassShapValuesTests(unittest.TestCase):
    """Unit tests for the version-robust SHAP normalizer (no shap required)."""

    def test_legacy_list_returns_class_one(self) -> None:
        class0 = np.zeros((4, 3))
        class1 = np.ones((4, 3))
        result = positive_class_shap_values([class0, class1])
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(np.allclose(result, 1.0))

    def test_modern_3d_array_selects_positive_class(self) -> None:
        # Shape (n_samples, n_features, n_classes); class 1 is all ones.
        arr = np.zeros((4, 3, 2))
        arr[..., 1] = 1.0
        result = positive_class_shap_values(arr)
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(np.allclose(result, 1.0))

    def test_two_dimensional_array_passes_through(self) -> None:
        arr = np.arange(12, dtype=float).reshape(4, 3)
        result = positive_class_shap_values(arr)
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(np.allclose(result, arr))


@unittest.skipUnless(HAS_SHAP, "shap is not installed")
class ShapIntegrationTests(unittest.TestCase):
    """End-to-end SHAP smoke test; skipped cleanly when shap is unavailable."""

    def test_global_shap_runs_and_writes_figure(self) -> None:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")

        from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, TARGET_COL
        from src.explain import compute_and_plot_global_shap
        from src.features import build_pipeline

        # Use real test data instead of synthetic
        test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
        df = pd.read_csv(test_path).head(300)  # Use first 300 real samples for speed
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        pipeline = build_pipeline()
        pipeline.fit(X, y)

        out_path = FIGURES_DIR / "shap_summary.png"
        if out_path.exists():
            out_path.unlink()

        # Must not raise; modern shap returns a 3-D array that the old
        # shap_values[1] indexing mishandled.
        compute_and_plot_global_shap(pipeline, X, max_samples=100)

        self.assertTrue(out_path.exists())

    def test_single_row_positive_class_shape(self) -> None:
        import pandas as pd
        import scipy.sparse as sp

        from src.config import PROCESSED_DATA_DIR, TARGET_COL
        from src.features import build_pipeline

        # Use real test data instead of synthetic
        test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
        df = pd.read_csv(test_path).head(300)  # Use first 300 real samples for speed
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        pipeline = build_pipeline()
        pipeline.fit(X, y)

        preprocessor = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["clf"]
        x_row = X.iloc[[0]]
        x_t = preprocessor.transform(x_row)
        x_t = x_t.toarray() if sp.issparse(x_t) else x_t

        explainer = shap.TreeExplainer(clf)
        single = positive_class_shap_values(explainer.shap_values(x_t))[0]

        self.assertEqual(single.shape, (x_t.shape[1],))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SCORE_COLUMNS = {"fraud_probability", "fraud_flag", "risk_band", "reason_codes"}


FRIENDLY_FEATURE_NAMES = {
    # V-features are PCA components; map to their fraud relevance
    "V1": "V1 (PCA component 1)",
    "V3": "V3 (PCA component 3)",
    "V4": "V4 (PCA component 4)",
    "V10": "V10 (PCA component 10)",
    "V14": "V14 (PCA component 14)",
    # Time and Amount
    "Time": "transaction timestamp",
    "Amount": "transaction amount",
}


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return str(value)


def high_amount_cutoff(df: pd.DataFrame, quantile: float = 0.95) -> float | None:
    """Return a robust high-amount cutoff from the current scoring batch."""
    if "Amount" not in df.columns or len(df) == 0:
        return None

    values = pd.to_numeric(df["Amount"], errors="coerce").dropna()
    if len(values) == 0:
        return None

    return float(values.quantile(quantile))


def reason_codes_for_row(
    row: pd.Series | dict,
    *,
    threshold: float = 0.5,
    amount_cutoff: float | None = None,
    max_reasons: int = 5,
) -> list[str]:
    """Create analyst-friendly rule-based reason codes for one transaction.

    These reason codes are intentionally simple and deterministic. They summarize
    obvious risk drivers in the input features and score output. They are not a
    causal explanation and should be reviewed together with SHAP/model evidence.

    For the real ULB Credit Card Fraud dataset:
    - Model score is primary indicator
    - High transaction amount (unusual for this batch)
    - Unusual transaction time (off-hours)
    - High anomaly scores in key PCA components (V1, V3, V4, V10, V14)
    """
    reasons: list[str] = []

    fraud_probability = _safe_float(row.get("fraud_probability"))
    if fraud_probability is not None:
        if fraud_probability >= 0.75:
            reasons.append("Critical model risk score")
        elif fraud_probability >= threshold:
            reasons.append("Model score is above the review threshold")

    # Check for high absolute values in key PCA components correlated with fraud
    for v_component in ["V1", "V3", "V4", "V10", "V14"]:
        value = _safe_float(row.get(v_component))
        if value is not None and abs(value) >= 2.5:
            reasons.append(f"High anomaly score in {v_component}")

    # High transaction amount
    amount = _safe_float(row.get("Amount"))
    if amount is not None and amount_cutoff is not None and amount >= amount_cutoff:
        reasons.append("Transaction amount is high for this batch")

    # Unusual transaction time (off-hours: late night or early morning)
    time_seconds = _safe_float(row.get("Time"))
    if time_seconds is not None:
        # Time is seconds since first transaction in dataset
        # Roughly: 24 hours per 86400 seconds, divide by 3600 for "hour-like" value
        hour_approx = (time_seconds / 3600) % 24
        if hour_approx <= 5 or hour_approx >= 23:
            reasons.append("Transaction occurred during unusual hours")

    if not reasons:
        reasons.append("No strong rule-based risk drivers identified")

    return reasons[:max_reasons]


def add_reason_codes(
    df_scored: pd.DataFrame,
    *,
    threshold: float = 0.5,
    max_reasons: int = 5,
) -> pd.DataFrame:
    """Add a semicolon-separated reason-code column to scored transactions."""
    df = df_scored.copy()
    cutoff = high_amount_cutoff(df)

    df["reason_codes"] = [
        "; ".join(
            reason_codes_for_row(
                record,
                threshold=threshold,
                amount_cutoff=cutoff,
                max_reasons=max_reasons,
            )
        )
        for record in df.to_dict("records")
    ]

    return df


def humanize_feature_name(feature_name: str) -> str:
    """Convert transformed sklearn feature names into analyst-friendly text."""
    raw = str(feature_name)

    for prefix in ("numeric__", "num__", "categorical__", "cat__"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break

    return FRIENDLY_FEATURE_NAMES.get(raw, raw.replace("_", " "))


def positive_class_shap_values(shap_values: object) -> np.ndarray:
    """Return SHAP values for the positive (fraud) class, across shap versions.

    ``TreeExplainer.shap_values`` has returned different shapes over time for
    binary classifiers:

    - legacy shap: a list ``[class_0_array, class_1_array]``;
    - modern shap (>= 0.43): a single array shaped
      ``(n_samples, n_features, n_classes)``.

    This normalizes both (and an already-2-D array) to the class-1 values so the
    rest of the code can stay version-agnostic.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[..., 1]
    return arr


def shap_reason_codes(
    shap_values: Iterable[float],
    feature_names: Iterable[str],
    *,
    max_reasons: int = 5,
) -> list[str]:
    """Convert SHAP values into concise analyst-friendly reason codes."""
    # Reshape returns tuple[int, ...] but mypy expects tuple[int]
    # This is safe at runtime; numpy's reshape is correctly dimensioned
    values = np.asarray(list(shap_values), dtype=float).reshape(-1)
    names = np.asarray(list(feature_names), dtype=object).reshape(-1)

    n = min(len(values), len(names))
    if n == 0:
        return ["No SHAP reason codes available"]

    values = values[:n]  # type: ignore[assignment]
    names = names[:n]  # type: ignore[assignment]

    order = np.argsort(np.abs(values))[::-1]
    reasons: list[str] = []

    for idx in order:
        value = float(values[idx])
        if np.isclose(value, 0.0):
            continue

        feature = humanize_feature_name(str(names[idx]))
        direction = "increased" if value > 0 else "reduced"
        reasons.append(f"{feature} {direction} fraud risk")

        if len(reasons) >= max_reasons:
            break

    return reasons or ["No strong SHAP drivers identified"]


def split_reason_codes(value: str | float | None) -> list[str]:
    """Split a saved reason-code string back into displayable list items."""
    if value is None or pd.isna(value):
        return []

    return [item.strip() for item in str(value).split(";") if item.strip()]

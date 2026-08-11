


# Fraud Risk Detection Engine

A production-minded, end-to-end fraud detection system for financial transactions — built to simulate real-world fintech risk workflows. The system goes beyond a simple classifier: it handles class imbalance, cost-sensitive decision thresholds, batch scoring, explainability, and an interactive analyst dashboard.

---

## Why This Project

Most fraud detection tutorials stop at training a model and printing an accuracy score. In practice, fraud detection is a *decision system* — a wrong threshold costs money, a missed fraud costs more, and an analyst needs to understand *why* a transaction was flagged, not just that it was. This project is built around those real constraints.

---

## What It Does

- Generates a realistic synthetic fraud dataset with class imbalance, label noise, and overlapping fraud/legitimate patterns
- Trains a scikit-learn classification pipeline with preprocessing and model steps
- Evaluates using fraud-appropriate metrics: ROC-AUC, PR-AUC, Brier score, and cost-weighted scoring
- Searches for the optimal decision threshold across four business policies
- Scores new transaction CSV files with fraud probabilities, binary flags, and ranked output
- Generates SHAP explanations and analyst-facing reason codes per transaction
- Provides an interactive Streamlit dashboard for fraud triage and threshold adjustment

---

## Project Structure

```
fraud-risk-detection-engine/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
│
├── data/
│   ├── raw/
│   │   └── synthetic_fraud_dataset.csv
│   └── processed/
│       ├── transactions_train.csv
│       └── transactions_test.csv
│
├── models/
│   ├── fraud_pipeline.joblib       # Trained model pipeline
│   └── threshold.json              # Saved decision threshold
│
├── reports/
│   ├── figures/
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── calibration_curve.png
│   │   ├── threshold_cost_curve.png
│   │   └── threshold_tradeoffs.png
│   └── metrics/
│       ├── metrics.json
│       ├── evaluation_summary.json
│       ├── threshold_policy.json
│       ├── threshold_policy.csv
│       └── threshold_policy.md
│
├── src/
│   ├── config.py                   # Central config and constants
│   ├── generate_synthetic_data.py  # Synthetic dataset generator
│   ├── data_prep.py                # Train/test split and preparation
│   ├── features.py                 # Preprocessing pipeline
│   ├── train_model.py              # Model training and saving
│   ├── evaluate.py                 # Metrics, plots, threshold search
│   ├── threshold_policy.py         # Business threshold policy artifacts
│   ├── score_new_transactions.py   # Batch scoring CLI
│   ├── validation.py               # Input schema validation
│   ├── reason_codes.py             # Analyst-readable risk explanations
│   ├── explain.py                  # SHAP explanation utilities
│   └── dashboard_utils.py          # Streamlit helper logic
│
├── tests/                          # 8 unit test modules
│
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

---

## System Workflow

```
Raw transactions
      ↓
Data preparation & validation
      ↓
Train/test split
      ↓
Preprocessing + model training
      ↓
Metric evaluation & threshold search
      ↓
Threshold policy artifacts
      ↓
Batch scoring (fraud probability + reason codes)
      ↓
Streamlit dashboard (SHAP + analyst triage)
```

---

## Getting Started

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/fraud-risk-detection-engine.git
cd fraud-risk-detection-engine

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Generate synthetic data
python -m src.generate_synthetic_data --rows 3500 --fraud-rate 0.08 --label-noise 0.04 --seed 42 --output data/raw/synthetic_fraud_dataset.csv

# Prepare train/test split
python -m src.data_prep

# Train the model
python -m src.train_model

# Evaluate and generate reports
python -m src.evaluate

# Score new transactions
python -m src.score_new_transactions data/processed/transactions_test.csv --output_csv reports/metrics/scored_transactions.csv
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

---

## Key Components

### Synthetic Data Generator

The dataset is generated with realistic fraud characteristics — not a clean, separable toy dataset. It includes:

- Configurable fraud rate (default 8%) with class imbalance
- Label noise to simulate mislabeled real-world data
- Overlapping fraud and legitimate transaction patterns
- Lower-risk-looking fraud and higher-risk-looking legitimate transactions

This makes threshold selection and precision/recall tradeoffs meaningful rather than trivial.

### Cost-Sensitive Threshold Optimization

A single accuracy score doesn't reflect the real cost of fraud decisions. Missing a fraud case and wrongly flagging a legitimate transaction have different financial consequences. The threshold search evaluates every candidate threshold across four business policies:

| Policy | Purpose |
|---|---|
| `cost_optimized` | Minimizes the configured FP/FN cost |
| `balanced_f1` | Balances precision and recall |
| `high_recall` | Prioritizes catching fraud, accepts more false positives |
| `high_precision` | Reduces false positives, may miss some fraud |
| `review_capacity` | Keeps flagged rate within analyst review limits |

Policy artifacts are exported as JSON, CSV, and Markdown for easy review.

### Explainability Layer

Two levels of explanation are generated per transaction:

**SHAP** — computes how each model feature contributed to the individual prediction, rendered as force plots in the dashboard.

**Reason codes** — human-readable risk signals surfaced to analysts, for example:
```
High device risk score
Transaction amount is high for this batch
Transaction occurred during unusual hours
Merchant category is elevated risk
```

### Batch Scoring CLI

```bash
python -m src.score_new_transactions <input.csv> --output_csv <output.csv>
```

Output columns added to the scored file:

| Column | Description |
|---|---|
| `fraud_probability` | Model-estimated fraud probability (0–1) |
| `fraud_flag` | Binary flag at the saved threshold |
| `reason_codes` | Human-readable risk drivers |

Output is sorted by descending fraud probability so the riskiest transactions appear first.

### Streamlit Dashboard

The dashboard supports:
- Uploading transaction CSV files for scoring
- Adjusting the fraud decision threshold interactively
- Viewing risk score distribution and flagged transaction count
- Reviewing top-risk transactions in a ranked table
- Inspecting individual transactions with SHAP force plots
- Downloading the scored CSV output

---

## Model Performance

Results on the synthetic dataset (seed 42):

| Metric | Value |
|---|---|
| ROC-AUC | 0.976 |
| PR-AUC | 0.838 |
| Brier Score | 0.061 |
| Selected Threshold | 0.35 |
| Precision @ threshold | 0.481 |
| Recall @ threshold | 0.974 |
| Flagged Rate | 22.3% |

> These results are on synthetic data and do not represent real-world fraud detection performance.

---

## Evaluation Charts

The evaluation step generates six charts saved to `reports/figures/`:

- **ROC Curve** — ranking quality across thresholds
- **Precision-Recall Curve** — more informative than ROC for imbalanced fraud data
- **Calibration Curve** — whether predicted probabilities are reliable
- **Threshold Cost Curve** — how FP/FN cost assumptions affect threshold selection
- **Threshold Tradeoffs** — precision, recall, FPR, and flagged rate across all thresholds
- **Confusion Matrix** — at the selected operating threshold

---

## Testing & CI

Run tests locally:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions CI pipeline (`ci.yml`) runs on every push and validates:
- Dependency installation
- Source compilation
- All unit tests
- Full pipeline run (data generation → training → evaluation → scoring)
- Scored output schema correctness
- Expected model and report artifacts

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, pandas, NumPy, scikit-learn, SciPy |
| Explainability | SHAP |
| Dashboard | Streamlit, matplotlib |
| Testing & CI | unittest, GitHub Actions |
| Serialization | joblib |

---

## Limitations

This is a portfolio and learning project, not a production system. Specific limitations:

- Data is synthetic — results do not prove real-world performance
- No real-time streaming inference
- No drift monitoring or retraining automation
- No fairness or subgroup analysis
- No compliance or regulatory controls
- Reason codes are heuristic, not causal

A production fraud system would require live monitoring, adversarial testing, compliance review, audit logging, and human escalation workflows.

---

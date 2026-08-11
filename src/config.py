from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Real-world dataset: ULB Credit Card Fraud Detection
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Column configuration
# Real dataset: Class = 1 is fraud, 0 is non-fraud
TARGET_COL = "Class"

# PCA-transformed features V1-V28 are already scaled; Time and Amount need scaling
NUMERIC_FEATURES = [
    "Time",  # Transaction timestamp (will be scaled)
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",  # Transaction amount (will be scaled)
]

# No categorical features in this dataset
CATEGORICAL_FEATURES: list[str] = []

# All features for the model
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Threshold search configuration
THRESHOLD_GRID = [i / 100 for i in range(5, 100, 5)]  # 0.05, 0.10, ..., 0.95

# Very simple "business" cost assumptions
COST_FALSE_NEGATIVE = 10.0  # missing a fraud is very expensive
COST_FALSE_POSITIVE = 1.0  # incorrectly flagging a normal transaction

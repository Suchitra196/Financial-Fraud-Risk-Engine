import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from .validation import validate_training_dataframe


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the real-world ULB Credit Card Fraud Detection dataset.

    Dataset info:
    - 284,807 rows, 31 columns
    - V1-V28: PCA-transformed features (already scaled)
    - Time: seconds since first transaction (needs scaling)
    - Amount: transaction amount in EUR (needs scaling)
    - Class: 0=non-fraud, 1=fraud (492 cases, 0.172% of data)
    - No missing values
    """
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    print(f"Loaded raw data from: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    validate_training_dataframe(df, context="ULB Credit Card Fraud Detection dataset")
    return df


def train_test_split_stratified(df: pd.DataFrame):
    """Create a stratified train/test split on the fraud label.

    Preserves the 0.172% fraud ratio in both train and test sets.
    """
    validate_training_dataframe(df, context="raw fraud dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    return train_df, test_df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train and test data to the processed folder."""
    train_path = PROCESSED_DATA_DIR / "transactions_train.csv"
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to:  {test_path}")


def main() -> None:
    df = load_raw_data()
    print(f"\nLoaded raw data with shape: {df.shape}")

    train_df, test_df = train_test_split_stratified(df)
    print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("\n=== Train set class distribution ===")
    print(train_df[TARGET_COL].value_counts())
    print("\nTrain fraud ratio:")
    print(train_df[TARGET_COL].value_counts(normalize=True))

    print("\n=== Test set class distribution ===")
    print(test_df[TARGET_COL].value_counts())
    print("\nTest fraud ratio:")
    print(test_df[TARGET_COL].value_counts(normalize=True))

    save_processed(train_df, test_df)
    print("\n[COMPLETE] Data preparation complete!")


if __name__ == "__main__":
    main()

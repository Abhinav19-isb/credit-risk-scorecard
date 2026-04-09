"""
generate_synthetic_data.py
Generates synthetic credit card default data for testing the pipeline
when the UCI dataset is unavailable or for stress-testing edge cases.

Usage:
    python scripts/generate_synthetic_data.py --rows 1000 --output data/synthetic_data.csv
    python scripts/generate_synthetic_data.py --rows 1000 --scenario bad_data --output data/synthetic_bad.csv
    python scripts/generate_synthetic_data.py --rows 1000 --scenario missing_cols --output data/synthetic_missing.csv
    python scripts/generate_synthetic_data.py --rows 1000 --scenario imbalanced --output data/synthetic_imbalanced.csv
"""

import argparse
import numpy as np
import pandas as pd

def generate_clean(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate clean synthetic data matching UCI schema."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()
    df["ID"] = range(1, n_rows + 1)
    df["LIMIT_BAL"] = rng.choice(
        [10000, 20000, 30000, 50000, 80000, 100000, 200000, 500000],
        size=n_rows
    )
    df["SEX"]       = rng.integers(1, 3, size=n_rows)
    df["EDUCATION"] = rng.integers(1, 5, size=n_rows)
    df["MARRIAGE"]  = rng.integers(1, 4, size=n_rows)
    df["AGE"]       = rng.integers(21, 75, size=n_rows)

    # Payment status columns — higher = more delayed
    for col in ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]:
        df[col] = rng.choice([-2,-1,0,1,2,3], size=n_rows, p=[0.1,0.2,0.4,0.15,0.1,0.05])

    # Bill amounts
    for i, col in enumerate(["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]):
        df[col] = rng.integers(0, 200000, size=n_rows)

    # Payment amounts
    for col in ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]:
        df[col] = rng.integers(0, 50000, size=n_rows)

    # Target — ~33% default rate
    pay_risk = df[["PAY_0","PAY_2","PAY_3"]].max(axis=1)
    prob = np.where(pay_risk >= 2, 0.75, np.where(pay_risk >= 1, 0.45, 0.15))
    df["default.payment.next.month"] = rng.binomial(1, prob)

    return df


def generate_bad_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Clean data with injected nulls and duplicates for failure testing."""
    df = generate_clean(n_rows, seed)

    # Inject nulls into BILL_AMT1
    null_idx = np.random.choice(df.index, size=int(n_rows * 0.02), replace=False)
    df.loc[null_idx, "BILL_AMT1"] = np.nan

    # Inject duplicates
    dup_rows = df.sample(n=int(n_rows * 0.01), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def generate_missing_cols(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Clean data with required columns removed — tests pipeline halt."""
    df = generate_clean(n_rows, seed)
    df.drop(columns=["PAY_0", "PAY_2"], inplace=True)
    return df


def generate_imbalanced(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Clean data with only 5% default rate — tests SMOTE fallback."""
    df = generate_clean(n_rows, seed)
    default_idx = df[df["default.payment.next.month"] == 1].index
    # Keep only 15% of defaulters to get ~5% overall default rate
    drop_idx = np.random.choice(default_idx, size=int(len(default_idx) * 0.85), replace=False)
    df.drop(index=drop_idx, inplace=True)
    return df.reset_index(drop=True)


SCENARIOS = {
    "clean":         generate_clean,
    "bad_data":      generate_bad_data,
    "missing_cols":  generate_missing_cols,
    "imbalanced":    generate_imbalanced,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit data for testing")
    parser.add_argument("--rows",     type=int, default=1000,       help="Number of rows to generate")
    parser.add_argument("--scenario", type=str, default="clean",    choices=list(SCENARIOS.keys()),
                        help="Data scenario: clean | bad_data | missing_cols | imbalanced")
    parser.add_argument("--output",   type=str, default="data/synthetic_data.csv", help="Output CSV path")
    parser.add_argument("--seed",     type=int, default=42,         help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating {args.rows} rows — scenario: {args.scenario}")
    df = SCENARIOS[args.scenario](args.rows, args.seed)

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows x {len(df.columns)} cols)")
    print(f"Default rate: {df['default.payment.next.month'].mean():.1%}" if "default.payment.next.month" in df.columns else "Target column not present (missing_cols scenario)")

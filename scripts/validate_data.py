"""
validate_data.py
Stage 1: Data Validation & Profiling for Credit Risk Scorecard
Usage: python scripts/validate_data.py --output outputs/validation_report.json
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo


# ── CONFIG ──────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]
TARGET_COLUMN = "default.payment.next.month"
NULL_THRESHOLD = 0.30       # Reject if > 30% nulls in any critical column
DUPLICATE_THRESHOLD = 0.05  # Warn if > 5% duplicate rows


# ── LOAD DATA ────────────────────────────────────────────────────────────────
def load_data():
    """Load UCI Credit Card Default dataset and rename columns to friendly names."""
    print("\n📦 Loading UCI Credit Card Default Dataset...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    # UCI returns X1, X2... — rename to meaningful names
    column_mapping = {
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
        "X6": "PAY_0",  "X7": "PAY_2",  "X8": "PAY_3",  "X9": "PAY_4",
        "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
        "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6"
    }
    X = X.rename(columns=column_mapping)

    df = pd.concat([X, y], axis=1)
    df.rename(columns={"Y": "DEFAULT"}, inplace=True)
    print(f"   ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   ✅ Columns renamed: X1–X23 → LIMIT_BAL, SEX, ... PAY_AMT6")
    return df



# ── STAGE 1A: COLUMN VALIDATION ──────────────────────────────────────────────
def validate_columns(df):
    """Check all required columns exist with correct types."""
    print("\n🔍 Stage 1A: Column Validation")
    results = {"status": "PASS", "issues": []}

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        results["status"] = "FAIL"
        results["issues"].append(f"Missing required columns: {missing}")
        print(f"   ❌ FAIL — Missing columns: {missing}")
    else:
        print(f"   ✅ All {len(REQUIRED_COLUMNS)} required columns present")

    # Check target exists
    if "DEFAULT" not in df.columns:
        results["status"] = "FAIL"
        results["issues"].append("Target column 'DEFAULT' missing")
        print("   ❌ FAIL — Target column missing")
    else:
        print(f"   ✅ Target column 'DEFAULT' present")

    return results


# ── STAGE 1B: NULL ANALYSIS ───────────────────────────────────────────────────
def validate_nulls(df):
    """Check null percentages across all columns."""
    print("\n🔍 Stage 1B: Null Value Analysis")
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    critical_nulls = null_pct[null_pct > NULL_THRESHOLD * 100]

    results = {
        "status": "PASS",
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentages": null_pct.to_dict(),
        "critical_violations": critical_nulls.to_dict()
    }

    if len(critical_nulls) > 0:
        results["status"] = "FAIL"
        print(f"   ❌ FAIL — Columns exceeding {NULL_THRESHOLD*100}% nulls: {critical_nulls.to_dict()}")
    else:
        print(f"   ✅ Total nulls: {results['total_nulls']} — All columns within threshold")

    return results


# ── STAGE 1C: DUPLICATE CHECK ─────────────────────────────────────────────────
def validate_duplicates(df):
    """Detect and report duplicate rows."""
    print("\n🔍 Stage 1C: Duplicate Row Check")
    n_dupes = df.duplicated().sum()
    dupe_pct = n_dupes / len(df)

    results = {
        "status": "PASS",
        "duplicate_count": int(n_dupes),
        "duplicate_percentage": round(dupe_pct * 100, 2)
    }

    if dupe_pct > DUPLICATE_THRESHOLD:
        results["status"] = "WARN"
        print(f"   ⚠️  WARN — {n_dupes:,} duplicate rows ({dupe_pct*100:.1f}%) — exceeds 5% threshold")
    else:
        print(f"   ✅ Duplicate rows: {n_dupes:,} ({dupe_pct*100:.2f}%) — within threshold")

    return results


# ── STAGE 1D: VALUE RANGE VALIDATION ─────────────────────────────────────────
def validate_value_ranges(df):
    """Check categorical and numeric columns for out-of-range values."""
    print("\n🔍 Stage 1D: Value Range Validation")
    issues = []

    checks = {
        "SEX":       ([1, 2], "Expected 1=Male, 2=Female"),
        "EDUCATION": ([1, 2, 3, 4], "Expected 1-4 (0,5,6 treated as Others)"),
        "MARRIAGE":  ([1, 2, 3], "Expected 1-3 (0 treated as Others)"),
        "DEFAULT":   ([0, 1], "Expected binary 0/1"),
    }

    for col, (valid_vals, note) in checks.items():
        if col not in df.columns:
            continue
        unexpected = df[~df[col].isin(valid_vals)][col].unique()
        if len(unexpected) > 0:
            issues.append(f"{col}: unexpected values {unexpected} — {note}")
            print(f"   ⚠️  {col}: undocumented values {unexpected} found → will recode as 'Others'")
        else:
            print(f"   ✅ {col}: all values within expected range")

    # Numeric range checks
    if (df["AGE"] < 18).any() or (df["AGE"] > 100).any():
        issues.append("AGE: values outside 18–100 range detected")
        print("   ⚠️  AGE: outliers outside 18–100 detected")
    else:
        print("   ✅ AGE: within valid range (18–100)")

    if (df["LIMIT_BAL"] <= 0).any():
        issues.append("LIMIT_BAL: zero or negative credit limits detected")
        print("   ⚠️  LIMIT_BAL: zero/negative values detected")
    else:
        print("   ✅ LIMIT_BAL: all values positive")

    return {"status": "WARN" if issues else "PASS", "issues": issues}


# ── STAGE 1E: CLASS BALANCE CHECK ────────────────────────────────────────────
def validate_class_balance(df):
    """Check target variable distribution for class imbalance."""
    print("\n🔍 Stage 1E: Target Class Balance")
    counts = df["DEFAULT"].value_counts()
    default_rate = counts.get(1, 0) / len(df) * 100

    results = {
        "non_default_count": int(counts.get(0, 0)),
        "default_count": int(counts.get(1, 0)),
        "default_rate_pct": round(default_rate, 2),
        "imbalance_flag": default_rate < 20 or default_rate > 50
    }

    print(f"   Non-Default (0): {results['non_default_count']:,} ({100-default_rate:.1f}%)")
    print(f"   Default     (1): {results['default_count']:,} ({default_rate:.1f}%)")
    if results["imbalance_flag"]:
        print(f"   ⚠️  Class imbalance detected — recommend SMOTE or class_weight='balanced'")
    else:
        print(f"   ✅ Class balance acceptable")

    return results


# ── STAGE 1F: STATISTICAL PROFILING ──────────────────────────────────────────
def profile_statistics(df):
    """Generate descriptive statistics for all numeric columns."""
    print("\n🔍 Stage 1F: Statistical Profiling")
    stats = df.describe().round(2).to_dict()
    print(f"   ✅ Statistical profile generated for {len(df.columns)} columns")
    return stats


# ── STAGE 1G: VISUALISATIONS ──────────────────────────────────────────────────
def generate_charts(df, output_dir="outputs/charts"):
    """Generate and save data quality visualisation charts."""
    print("\n📊 Generating Data Quality Charts...")
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Default Rate Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["DEFAULT"].value_counts()
    axes[0].bar(["Non-Default (0)", "Default (1)"], counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[0].set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Customer Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

    # Chart 2: Credit Limit Distribution by Default Status
    df.groupby("DEFAULT")["LIMIT_BAL"].plot(kind="kde", ax=axes[1],
                                             label=["Non-Default", "Default"])
    axes[1].set_title("Credit Limit Distribution by Default Status", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Credit Limit (NT$)")
    axes[1].legend(["Non-Default", "Default"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart1_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: chart1_class_distribution.png")

    # Chart 3: Null Heatmap
    fig, ax = plt.subplots(figsize=(14, 4))
    null_data = df.isnull().sum().to_frame(name="Null Count").T
    sns.heatmap(null_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=False)
    ax.set_title("Null Value Count per Column", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart2_null_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: chart2_null_heatmap.png")

    # Chart 4: Payment Delay Distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    df["PAY_0"].value_counts().sort_index().plot(kind="bar", ax=ax,
                                                  color="#3498db", edgecolor="black")
    ax.set_title("Payment Status Distribution (Most Recent Month — PAY_0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Payment Status (-1=On Time, 1–9=Months Delayed)")
    ax.set_ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart3_payment_status.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: chart3_payment_status.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(output_path="outputs/validation_report.json"):
    print("=" * 60)
    print("  CREDIT RISK SCORECARD — STAGE 1: DATA VALIDATION")
    print("=" * 60)

    df = load_data()

    report = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_validation":    validate_columns(df),
        "null_validation":      validate_nulls(df),
        "duplicate_validation": validate_duplicates(df),
        "range_validation":     validate_value_ranges(df),
        "class_balance":        validate_class_balance(df),
        "statistical_profile":  profile_statistics(df),
    }

    generate_charts(df)

    # Overall status
    statuses = [v.get("status", "PASS") for v in report.values() if isinstance(v, dict)]
    report["overall_status"] = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  OVERALL STATUS: {report['overall_status']}")
    print(f"  Report saved → {output_path}")
    print(f"  Charts saved → outputs/charts/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Data Validation")
    parser.add_argument("--output", default="outputs/validation_report.json",
                        help="Path to save validation report JSON")
    args = parser.parse_args()
    main(args.output)

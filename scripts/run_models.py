import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report, confusion_matrix  )

# stage 2 : load and clean data
def load_and_clean():
    """Load UCI dataset, rename columns, recode undocumented values."""
    print("\n📦 Loading & Cleaning Data...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    col_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6"
    }
    X = X.rename(columns=col_map)
    df = pd.concat([X, y.rename(columns={"Y": "DEFAULT"})], axis=1)

    # Recode undocumented values
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"   ✅ Data Loaded & Cleaned: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df



#Stage 3: feature engineering
def feature_engineering(df):
    """Create new features and scale data."""
    print("\n🔧 Performing Feature Engineering...")
    bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    df["AVG_BILL_AMT"]    = df[bill_cols].mean(axis=1)
    df["UTILISATION_RATE"] = (df["AVG_BILL_AMT"] / df["LIMIT_BAL"]).clip(0, 1)
    print("   ✅ UTILISATION_RATE: avg bill / credit limit (clipped 0–1)")


    # Feature 2: Payment Ratio — how much of bill is being paid back
    pay_cols  = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    df["AVG_PAY_AMT"]  = df[pay_cols].mean(axis=1)
    df["PAYMENT_RATIO"] = (df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"] + 1)).clip(0, 5)
    print("   ✅ PAYMENT_RATIO: avg payment / avg bill")

    # Feature 3: Max Delay — worst payment delay in last 6 months

    delay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    df["MAX_DELAY"]  = df[delay_cols].max(axis=1)
    df["MEAN_DELAY"] = df[delay_cols].mean(axis=1)
    print("   ✅ MAX_DELAY / MEAN_DELAY: worst and average payment delay")

    # Feature 4: Delay Count — number of months with any payment delay (PAY > 0)
    df["DELAY_COUNT"] = (df[delay_cols] > 0).sum(axis=1)
    print("   ✅ DELAY_COUNT: number of months with payment delay ")

    # Feature 5: Bill trend — is debt growing or shrinking?
    # Formula: most recent bill - oldest bill (positive = growing debt)
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    print("   ✅ BILL_TREND: recent bill minus oldest bill (growth indicator)")

     # Feature 6: Pay trend — is payment amount increasing?
    df["PAY_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]
    print("   ✅ PAY_TREND: recent payment minus oldest payment")

    print(f"   ➕ Created 6 new features. Total columns now: {df.shape[1]}")


    print("✅ Feature Engineering Completed.")
    return df

    # ── STAGE 3: PREPARE FEATURE MATRIX
def prepare_features(df):
    """select final feature set and split into X and y."""

    feature_cols = [
        # Original features
        "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE", "SEX",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        # Engineered features
        "UTILISATION_RATE", "PAYMENT_RATIO",
        "MAX_DELAY", "MEAN_DELAY", "DELAY_COUNT",
        "BILL_TREND", "PAY_TREND"]
    

    X = df[feature_cols].fillna(0)  # Fill any missing values with 0 (if any)
    y = df["DEFAULT"]

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"\n   ✅ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    return X, X_scaled, y, feature_cols

# ── STAGE 3: MODELLING ────────────────────────────────────────────────────────
def run_logistic_regression(X_scaled, y):
    """Model 1: Logistic Regression — interpretable scorecard base model."""
    print("\n🤖 Stage 3A: Logistic Regression (Scorecard Model)...")

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000, C=0.1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    model.fit(X_scaled, y)  # Fit on full data for feature importance
    y_prob =  model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    results = {
        "model": "Logistic Regression",
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std": round(auc_scores.std(), 4),
        "train_auc": round(roc_auc_score(y, y_prob), 4),
        "gini": round(2 * roc_auc_score(y, y_prob) - 1, 4),
        "ks_statistic": round(compute_ks(y, y_prob), 4),
         }

    print(f"✅ Logistic Regression completed. CV AUC: {results['cv_auc_mean']} ± {results['cv_auc_std']}")
    print(f" Gini: {results['gini']}, KS: {results['ks_statistic']}")
    return model, y_prob, y_pred, results

def run_gradient_boosting(X, y):
    """Model 2: Gradient Boosting — high-performance comparison model."""
    print("\n🤖 Stage 3B: Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    model.fit(X, y)  # Fit on full data for feature importance
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    results = {
        "model": "Gradient Boosting",
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std": round(auc_scores.std(), 4),
        "train_auc": round(roc_auc_score(y, y_prob), 4),
        "gini": round(2 * roc_auc_score(y, y_prob) - 1, 4),
        "ks_statistic": round(compute_ks(y, y_prob), 4),
    }

    print(f"   ✅ CV AUC: {results['cv_auc_mean']} ± {results['cv_auc_std']}")
    print(f"   ✅ Gini:   {results['gini']}")
    print(f"   ✅ KS:     {results['ks_statistic']}")
    return model, y_prob, y_pred, results

# ── STAGE 4: VALIDATION METRICS ───────────────────────────────────────────────
def compute_ks(y_true, y_prob):
    """KS Statistic: max separation between default and non-default score distributions."""
    df = pd.DataFrame({"y": y_true, "prob": y_prob}).sort_values("prob", ascending=False)
    df["cum_default"] = (df["y"]==1).cumsum() / (df["y"]==1).sum()
    df["cum_non_default"] = (df["y"]==0).cumsum() / (df["y"]==0).sum()
    return (df["cum_default"] - df["cum_non_default"]).abs().max()


def compare_models(lr_results, gb_results):
    """Compare both models and pick winner based on CV AUC + KS."""
    print("\n📊 Stage 4: Model Comparison & Validation")
    print(f"\n   {'Metric':<20} {'Logistic Regression':>22} {'Gradient Boosting':>20}")
    print(f"   {'-'*62}")
    for metric in ["cv_auc_mean", "gini", "ks_statistic"]:
        print(f"   {metric:<20} {lr_results[metric]:>22} {gb_results[metric]:>20}")


    winner = "Gradient Boosting" if gb_results["cv_auc_mean"] > lr_results["cv_auc_mean"] else "Logistic Regression"
    print(f"\n   🏆 Best model by CV AUC: {winner}")

    # Industry thresholds
    best = gb_results if winner == "Gradient Boosting" else lr_results
    print(f"\n   📋 Industry Benchmark Validation:")
    print(f"   AUC  > 0.70 (good): {'✅' if best['cv_auc_mean'] > 0.70 else '❌'} ({best['cv_auc_mean']})")
    print(f"   Gini > 0.40 (good): {'✅' if best['gini'] > 0.40 else '❌'} ({best['gini']})")
    print(f"   KS   > 0.30 (good): {'✅' if best['ks_statistic'] > 0.30 else '❌'} ({best['ks_statistic']})")

    return winner


def build_scorecard(y_prob, output_dir):
    """Map predicted default probability to a 300–850 credit score."""
    print("\n💳 Building Credit Score Mapping (300–850 scale)...")

    # Standard credit score formula: higher score = lower risk
    scores = 850 - (y_prob * 550)  # Map 0% default to 850, 100% default to 300
    scores = scores.round().astype(int)
    scores = scores.clip(300, 850)

    bins = [300, 500, 580, 670, 740, 800, 850]
    labels = ["Very Poor", "Poor","Fair", "Good", "Very Good", "Exceptional"]
    
    tiers = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    tier_counts = tiers.value_counts().sort_index()

    print("\n   Credit Score Tier Distribution:")
    for tier, count in tier_counts.items():
        print(f"   {tier:<15} {count:>6,} customers ({count/len(scores)*100:.2f}%) ")

    # Save score output
    score_df = pd.DataFrame({"Predicted_Prob": y_prob, "Credit_Score": scores, "Score_Tier": tiers})
    score_df.to_csv(os.path.join(output_dir, "credit_scores.csv"), index=False)
    print(f"\n   ✅ Credit scores saved to {os.path.join(output_dir, 'credit_scores.csv')}")
    return scores, tiers

# ── CHARTS ────────────────────────────────────────────────────────────────────
def generate_model_charts(y, lr_prob, gb_prob, scores, feature_cols, gb_model, output_dir):
    """Generate model validation and scorecard charts."""
    print("\n📊 Generating Model Charts...")
    os.makedirs(f"{output_dir}/charts", exist_ok=True)

    # Chart 1: ROC Curves — both models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for prob, label, color in [(lr_prob, "Logistic Regression", "#3498db"),
                                (gb_prob, "Gradient Boosting", "#e74c3c")]:
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, lw=2)
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    # Chart 2: Feature Importance (Gradient Boosting)
    importances = pd.Series(gb_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
    importances.plot(kind="barh", ax=axes[1], color="#2ecc71", edgecolor="black")
    axes[1].set_title("Top Feature Importances\n(Gradient Boosting)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/charts/chart4_roc_and_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: chart4_roc_and_importance.png")

    # Chart 3: Credit Score Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(scores, bins=50, color="#9b59b6", edgecolor="black", alpha=0.8)
    for x, label in [(500,"Poor"),(580,"Fair"),(670,"Good"),(740,"Very Good"),(800,"Exceptional")]:
        ax.axvline(x, color="red", linestyle="--", alpha=0.6)
        ax.text(x+2, ax.get_ylim()[1]*0.85, label, fontsize=8, color="red", rotation=90)
    ax.set_title("Credit Score Distribution of 30,000 Customers", fontsize=13, fontweight="bold")
    ax.set_xlabel("Credit Score (300–850)")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/charts/chart5_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: chart5_score_distribution.png")

#main

def main(output_dir = "outputs/models"):
    print("=" * 60)
    print("🚀 Running Credit Risk Scorecard Stages 2-4: Modeling")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean()
    df = feature_engineering(df)
    X, X_scaled, y, feature_cols = prepare_features(df)

    lr_model, lr_prob, lr_pred, lr_results = run_logistic_regression(X_scaled, y)
    gb_model, gb_prob, gb_pred, gb_results = run_gradient_boosting(X, y)

    winner = compare_models(lr_results, gb_results)

    # use winning model prob for scorecard
    best_prob = gb_prob if winner == "Gradient Boosting" else lr_prob
    scores, tiers = build_scorecard(pd.Series(best_prob), output_dir)

    generate_model_charts(y, lr_prob, gb_prob, scores, feature_cols, gb_model, output_dir)

    summary = {
        "logistic_regression": lr_results,
        "gradient_boosting": gb_results,
        "best_model": winner,
        "scorecard_built": True
    }   

    with open(f"{output_dir}/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  STAGE 2–4 COMPLETE")
    print(f"  Winner: {winner}")
    print(f"  Results saved → {output_dir}/model_results.json")
    print(f"  Scores  saved → {output_dir}/credit_scores.csv")
    print(f"  Charts  saved → {output_dir}/charts/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Modelling")
    parser.add_argument("--output", default="outputs/models",
                        help="Directory to save model outputs")
    args = parser.parse_args()
    main(args.output)
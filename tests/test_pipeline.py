"""
test_pipeline.py
Unit tests for Credit Risk Scorecard pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.validate_data import (
    validate_columns,
    validate_nulls,
    validate_duplicates,
    validate_value_ranges,
    validate_class_balance,
)
from scripts.run_models import feature_engineering, prepare_features, compute_ks


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal 500-row synthetic dataframe matching UCI schema."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(10000, 500000, n),
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n),
        "MARRIAGE": np.random.choice([1, 2, 3], n),
        "AGE": np.random.randint(21, 65, n),
        "PAY_0": np.random.choice([-1, 0, 1, 2], n),
        "PAY_2": np.random.choice([-1, 0, 1, 2], n),
        "PAY_3": np.random.choice([-1, 0, 1], n),
        "PAY_4": np.random.choice([-1, 0, 1], n),
        "PAY_5": np.random.choice([-1, 0, 1], n),
        "PAY_6": np.random.choice([-1, 0, 1], n),
        "BILL_AMT1": np.random.randint(0, 100000, n).astype(float),
        "BILL_AMT2": np.random.randint(0, 100000, n).astype(float),
        "BILL_AMT3": np.random.randint(0, 100000, n).astype(float),
        "BILL_AMT4": np.random.randint(0, 100000, n).astype(float),
        "BILL_AMT5": np.random.randint(0, 100000, n).astype(float),
        "BILL_AMT6": np.random.randint(0, 100000, n).astype(float),
        "PAY_AMT1": np.random.randint(0, 50000, n).astype(float),
        "PAY_AMT2": np.random.randint(0, 50000, n).astype(float),
        "PAY_AMT3": np.random.randint(0, 50000, n).astype(float),
        "PAY_AMT4": np.random.randint(0, 50000, n).astype(float),
        "PAY_AMT5": np.random.randint(0, 50000, n).astype(float),
        "PAY_AMT6": np.random.randint(0, 50000, n).astype(float),
        "DEFAULT": np.random.choice([0, 1], n, p=[0.78, 0.22]),
    })
    return df


# ── STAGE 1: VALIDATION TESTS ─────────────────────────────────────────────────

class TestColumnValidation:
    def test_all_required_columns_present(self, sample_df):
        result = validate_columns(sample_df)
        assert result["status"] == "PASS", f"Expected PASS, got: {result}"

    def test_missing_column_triggers_fail(self, sample_df):
        df_broken = sample_df.drop(columns=["LIMIT_BAL"])
        result = validate_columns(df_broken)
        assert result["status"] == "FAIL"

    def test_missing_target_triggers_fail(self, sample_df):
        df_no_target = sample_df.drop(columns=["DEFAULT"])
        result = validate_columns(df_no_target)
        assert result["status"] == "FAIL"


class TestNullValidation:
    def test_no_nulls_passes(self, sample_df):
        result = validate_nulls(sample_df)
        assert result["status"] == "PASS"

    def test_high_null_column_fails(self, sample_df):
        df_nulls = sample_df.copy()
        df_nulls.loc[:200, "LIMIT_BAL"] = np.nan   # >30% nulls
        result = validate_nulls(df_nulls)
        assert result["status"] == "FAIL"


class TestDuplicateValidation:
    def test_no_duplicates_passes(self, sample_df):
        result = validate_duplicates(sample_df)
        assert result["status"] == "PASS"

    def test_many_duplicates_warns(self, sample_df):
        # Force >5% duplicates
        df_dupes = pd.concat([sample_df, sample_df.iloc[:50]], ignore_index=True)
        result = validate_duplicates(df_dupes)
        assert result["status"] == "WARN"


class TestValueRangeValidation:
    def test_valid_ranges_pass(self, sample_df):
        result = validate_value_ranges(sample_df)
        assert result["status"] in ("PASS", "WARN")   # undocumented values may warn

    def test_invalid_sex_value_warns(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, "SEX"] = 99
        result = validate_value_ranges(df_bad)
        assert len(result["issues"]) > 0


class TestClassBalance:
    def test_default_rate_computed(self, sample_df):
        result = validate_class_balance(sample_df)
        assert "default_rate_pct" in result
        assert 0 < result["default_rate_pct"] < 100

    def test_severely_imbalanced_flagged(self):
        n = 500
        df_imbal = pd.DataFrame({
            "DEFAULT": [0] * 490 + [1] * 10   # 2% default rate
        })
        result = validate_class_balance(df_imbal)
        assert result["imbalance_flag"], f"Got: {result['imbalance_flag']!r} ({type(...).__name__})"

# ── STAGE 2-4: MODELLING TESTS ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_new_features_created(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        new_cols = ["UTILISATION_RATE", "PAYMENT_RATIO", "MAX_DELAY",
                    "MEAN_DELAY", "DELAY_COUNT", "BILL_TREND", "PAY_TREND"]
        for col in new_cols:
            assert col in df_eng.columns, f"Missing engineered feature: {col}"

    def test_utilisation_rate_bounded(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        assert df_eng["UTILISATION_RATE"].between(0, 1).all(), \
            "UTILISATION_RATE should be clipped between 0 and 1"

    def test_no_nulls_after_engineering(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        assert df_eng[["UTILISATION_RATE", "PAYMENT_RATIO", "MAX_DELAY"]].isnull().sum().sum() == 0


class TestPrepareFeatures:
    def test_output_shapes_consistent(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        X, X_scaled, y, feature_cols = prepare_features(df_eng)
        assert X.shape[0] == X_scaled.shape[0] == len(y)
        assert X.shape[1] == X_scaled.shape[1] == len(feature_cols)

    def test_no_nulls_in_feature_matrix(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        X, X_scaled, y, _ = prepare_features(df_eng)
        assert X.isnull().sum().sum() == 0
        assert np.isnan(X_scaled.values).sum() == 0


class TestKSStatistic:
    def test_ks_between_zero_and_one(self):
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000, p=[0.78, 0.22])
        y_prob = np.random.uniform(0, 1, 1000)
        ks = compute_ks(y_true, y_prob)
        assert 0.0 <= ks <= 1.0, f"KS should be 0-1, got {ks}"

    def test_perfect_model_ks_is_one(self):
        y_true = np.array([1] * 100 + [0] * 100)
        y_prob = np.array([0.99] * 100 + [0.01] * 100)
        ks = compute_ks(y_true, y_prob)
        assert ks > 0.95, f"Perfect model KS should be ~1.0, got {ks}"

    def test_random_model_ks_near_zero(self):
        np.random.seed(99)
        y_true = np.random.choice([0, 1], 5000)
        y_prob = np.random.uniform(0, 1, 5000)
        ks = compute_ks(y_true, y_prob)
        assert ks < 0.10, f"Random model KS should be ~0, got {ks}"


# ── SCORECARD OUTPUT TESTS ────────────────────────────────────────────────────

class TestScorecardOutput:
    def test_scores_within_fico_range(self):
        """Credit scores must always fall within 300-850 FICO range."""
        from scripts.run_models import build_scorecard
        import tempfile
        y_prob = pd.Series(np.random.uniform(0, 1, 1000))
        with tempfile.TemporaryDirectory() as tmpdir:
            scores, _ = build_scorecard(y_prob, tmpdir)
        assert scores.min() >= 300, f"Score below 300 found: {scores.min()}"
        assert scores.max() <= 850, f"Score above 850 found: {scores.max()}"

    def test_high_risk_gets_low_score(self):
        """Probability of 1.0 (certain default) must map to score 300."""
        from scripts.run_models import build_scorecard
        import tempfile
        y_prob = pd.Series([1.0, 0.5, 0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            scores, _ = build_scorecard(y_prob, tmpdir)
        assert scores.iloc[0] == 300, "P(default)=1.0 must map to score 300"
        assert scores.iloc[2] == 850, "P(default)=0.0 must map to score 850"

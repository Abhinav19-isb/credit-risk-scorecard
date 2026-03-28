
---
name: credit-risk-scorecard
description: "Build a complete credit risk scorecard from raw credit card transaction data. Use this skill whenever the user asks to build a credit scorecard, assess default risk, segment customers by credit score, compare Logistic Regression vs Gradient Boosting for credit risk, compute KS statistic or Gini coefficient, or generate a credit risk report. Trigger for requests like 'build me a credit scorecard', 'score these customers for default risk', 'which customers are high risk', or 'run a credit risk model on this data'. Uses UCI Credit Card Default Dataset and outputs a professional HTML report with score tiers and business recommendations."
---

## Overview
This skill instructs an LLM to perform an end to end credit risk scorecard analysis.
Given a dataset of credit card customers, the skill orchestrates a complete pipeline:
data validation → feature engineering → dual-model comparison → scorecard generation
→ professional HTML report.

**Domain:** Financial Analytics - Credit Risk
**Dataset:** UCI Credit Card Default Dataset (30,000 customers, 23 features)
**Output:** Professional HTML report with model validation. score tiers, and business recommendations

---

### Required Files
| File | Purpose |
|------|---------|
| `scripts/validate_data.py` | Stage 1: Data validation and profiling |
| `scripts/run_models.py` | Stages 2–4: Feature engineering, modelling, validation |
| `scripts/generate_report.py` | Stages 5–6: Insight generation and HTML report |

### Required Python Libraries
```bash
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn plotly jinja2
```

### Dataset Input
- **Source:** UCI ML Repository (ID: 350) - loaded automatically via 'ucimlrepo'
- **No manual download required**
- **Minimum required columns:** LIMIT_BAL, SEX , EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2-PAY_6, BILL_AMT1-6, PAY_AMT1-6, DEFAULT (target)
- **Format:** CSV / loaded via Python API

### User Configuration Parameters
| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| n_estimators | Integer | 100 | 50–500 | Number of trees in Gradient Boosting |
| learning_rate | Float | 0.1 | 0.01–0.5 | Step size for Gradient Boosting |
| cv_folds | Integer | 5 | 3–10 | Number of cross-validation folds |
| null_threshold | Float | 0.30 | 0.0–1.0 | Max allowed null % before rejection |
| score_min | Integer | 300 | 300 | Minimum credit score (FICO-style) |
| score_max | Integer | 850 | 850 | Maximum credit score (FICO-style) |
| confidence_level | Float | 0.95 | 0.90–0.99 | Confidence level for statistical tests |

---

## Stage 1: Data Validation & Profiling
**Script:** 'scripts/validate_data.py'
**Command:**
```bash
python scripts/validate_data.py --output outputs/validation_report.json
```

**What to compute:**
1. Row count, column count, data types for all 24 columns
2. Null percentage per column - REJECT if any critical column exceeds 30% nulls
3. Duplicate row count - WARN if duplicates exceeds 5% of total rows
4. Value range checks:
   - Sex: must be in [1, 2]
   - EDUCATION: must be in [0-6] - recode [0, 5, 6] → 4 (Others)
   - MARRIAGE: must be in [0–3] — recode [0] → 3 (Others)
   - AGE: must be between 18 and 100
   - LIMIT_BAL: must be positive (> 0)
   - DEFAULT: must be binary [0, 1]
5. Class balance check - WARN if default rate < 10% or > 60%
6. Statistical profile: min, max, mean, median, std for all numeric columns

**Expected Outputs:**
- `outputs/validation_report.json` — structured quality report
- `outputs/charts/chart1_class_distribution.png`
- `outputs/charts/chart2_null_heatmap.png`
- `outputs/charts/chart3_payment_status.png`

**Validation before proceeding:**
- overall_status must be PASS or WARN (not FAIL)
- If FAIL → stop pipeline, report exact column and issue to user
- If WARN → proceed but document warnings in report

**Known data quality issues in this dataset:**
- EDUCATION values [0, 5, 6] are undocumented → recode to 4 (Others)
- MARRIAGE value [0] is undocumented → recode to 3 (Others)
- No missing values in original dataset
- 35 duplicate rows exist → remove before modelling

---

## Stage 2: Data Preparation & Feature Engineering

**Script:** `scripts/run_models.py` (handles Stages 2–4)  
**Trigger:** Runs automatically as part of `run_models.py`

**Column renaming (UCI returns X1-X23):**
```text
X1→LIMIT_BAL, X2→SEX, X3→EDUCATION, X4→MARRIAGE, X5→AGE,
X6→PAY_0, X7→PAY_2, X8→PAY_3, X9→PAY_4, X10→PAY_5, X11→PAY_6,
X12→BILL_AMT1, X13→BILL_AMT2, X14→BILL_AMT3, X15→BILL_AMT4,
X16→BILL_AMT5, X17→BILL_AMT6, X18→PAY_AMT1, X19→PAY_AMT2,
X20→PAY_AMT3, X21→PAY_AMT4, X22→PAY_AMT5, X23→PAY_AMT6,
Y→DEFAULT
```

**Features to engineer (7 new columns):**
| Feature | Formula | Expected Range | Business Meaning |
|---------|---------|----------------|-----------------|
| UTILISATION_RATE | avg(BILL_AMT1–6) / LIMIT_BAL, clipped [0,1] | 0.0–1.0 | Credit usage ratio |
| PAYMENT_RATIO | avg(PAY_AMT1–6) / (avg(BILL_AMT1–6)+1), clipped [0,5] | 0.0–5.0 | Bill repayment rate |
| MAX_DELAY | max(PAY_0, PAY_2–6) | -2 to 9 | Worst delay in 6 months |
| MEAN_DELAY | mean(PAY_0, PAY_2–6) | -2.0 to 9.0 | Average payment delay |
| DELAY_COUNT | count(PAY_x > 0) across 6 months | 0–6 | Months with any delay |
| BILL_TREND | BILL_AMT1 − BILL_AMT6 | Any numeric | Debt growth direction |
| PAY_TREND | PAY_AMT1 − PAY_AMT6 | Any numeric | Payment improvement |

**Validation after feature engineering:**
- Final feature matrix must have 18 columns (11 original + 7 engineered)
- Apply '.fillna(0)' to handle any NaN from division operations
- Apply StandardScaler' to feature matrix for Logistic Regression only

---

## Stage 3: Modelling & Analysis

**Script:** 'scripts/run_models.py'
**Command:**
```bash
python scripts/run_models.py --output output/models
```

**Model 1 - Logistic Regression (Scorecard Model):**
```text
Parameters: C=0.1, class_weight='balanced', max_iter=1000, random_state=42
Input: X_scaled (StandardScaler applied)
CV: StratifiedKFold, n_splits=5, shuffle=True, random_state=42
Scoring: roc_auc
```


**Model 2 — Gradient Boosting (Performance Benchmark):**
```text
Parameters: n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
Input: X (unscaled — tree models don't need scaling)
CV: StratifiedKFold, n_splits=5, shuffle=True, random_state=42
Scoring: roc_auc
```

**Why two models:**
- Logistic Regression is the industry standard for credit scorecards - its coefficients are interpretable and map directly to score points
- Gradient Boosting captures non-linear relationships (e.g., interaction between high utilization and recent delays) that LR misses
- Running both allows quantitative comparison rather than assumption-based model selection

---

## Stage 4: Model and Result Validation
**Handled inside:** `scripts/run_models.py`

**Metrics to compute for each model:**

| Metric | Formula | Good Threshold | Excellent Threshold |
|--------|---------|----------------|---------------------|
| CV AUC | mean of 5-fold roc_auc scores | > 0.70 | > 0.80 |
| Gini | 2 × AUC − 1 | > 0.40 | > 0.60 |
| KS Statistic | max(cum_default% − cum_non_default%) | > 0.30 | > 0.40 |

**Model selection rule:**
- Select model with higher CV AUC (not train AUC — avoids overfitting bias)
- If CV AUC difference < 0.01, prefer Logistic Regression (interpretability)
- Document winner with exact metric values in `output/models/model_summary.json`

**Credit score mapping (300-850 FICO-style scale):**
```text
credit_score = 850 − (predicted_default_probability × 550)
credit_score = clip(credit_score, 300, 850)
```

**Score Tier Definitions:**
| Tier | Score Range | Action |
|------|-------------|--------|
| Very Poor | 300–500 | Immediate collections review |
| Poor | 500–580 | Limit reduction + monitoring |
| Fair | 580–670 | Payment reminders + monitoring |
| Good | 670–740 | Standard monitoring |
| Very Good | 740–800 | Eligible for limit increase |
| Exceptional | 800–850 | Cross-sell premium products |

**Expected outputs:**
- `output/models/model_summary.json` — CV AUC, Gini, KS for both models + winner
- `output/models/credit_scores.csv` — predicted probability, credit score, tier per customer
- `output/models/charts/chart4_roc_and_importance.png`
- `output/models/charts/chart5_score_distribution.png`

---

## Stage 5: Insights Generation and Interpretation

**Script:** 'scripts/generate_report.py' (handles stages 5-6)

**Business interpretations to generate:**

1. **AUC interpretation:**
   - ≥ 0.80 → "Excellent — strong discriminatory power"
   - ≥ 0.70 → "Good — meets industry deployment standards"
   - < 0.70 → "Moderate — further feature engineering recommended"
2. **KS interpretation:**
   - ≥ 0.40 → "Strong separation between defaulters and non-defaulters"
   - ≥ 0.30 → "Acceptable separation - model is deployment-ready"
   - < 0.30 → "Weak separation - review featue set before deployment"
  
3. **Portfolio Segmentation**
   - Compute % of customers in each score tier
   - High-risk = Very Poor + Poor tiers combined
   - Low risk = Very Good + Exceptional tiers combined
   - Flag if high-risk regment > 30% of portfolio (concentration risk)
  
4. **Consult REFERENCE.md for:**
   - Segment archetype labels and recommended actions per tier
   - Industry benchmark ranges for AUC, Gini, KS
   - Basel II/III model documentation standards

---

## Stage 6: Report Generation

**Script:** `scripts/generate_report.py`  
**Command:**
```bash
python scripts/generate_report.py
```


**Required report sections (in this exact order):**
1. **Header** — dataset name, customer count, generation timestamp, author
2. **Executive Summary** — AUC, Gini, KS KPI cards + 2-sentence business summary
3. **Data Quality Summary** — table with PASS/WARN/FAIL per check + 2 charts
4. **Methodology** — feature engineering table (formula + business meaning) + model comparison table
5. **Model Validation Results** — metrics table with industry thresholds + ROC curve chart
6. **Credit Score Distribution** — tier table with bar chart + score distribution chart
7. **Business Recommendations** — one recommendation box per risk tier
8. **Assumptions & Limitations** — minimum 4 limitations documented

**Report format requirements:**
- Single self-contained HTML file (all charts base64-embedded)
- All charts must have titles, axis labels, and legends
- Report must be viewable in any browser without external dependencies
- Author: Abhinav Srivastav | AMPBA, ISB

**Expected output:**
- `outputs/reports/credit_risk_report.html`

---

## Running the Full Pipeline (All Stages)

Run these three commands in sequence:

```bash
# Stage 1: Validate data
python scripts/validate_data.py --output outputs/validation_report.json

# Stages 2–4: Feature engineering, modelling, validation
python scripts/run_models.py --output output/models

# Stages 5–6: Insights and report
python scripts/generate_report.py
```

**Total runtime:**
approximately 2-3 minutes (Gradient Boosting training is the bottleneck)

---

## Error Handling

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `KeyError: 'AGE'` | UCI returns X1–X23, not friendly names | Ensure col_map rename runs before any column access |
| `KeyError: 'DEFAULT'` | Target column named 'Y' in UCI | Rename Y → DEFAULT after concat |
| `ValueError: Input X contains NaN` | Division by zero in UTILISATION_RATE | Apply `.fillna(0)` after feature engineering |
| `overall_status: FAIL` | Critical data quality violation | Check validation_report.json for exact column and issue |
| `FileNotFoundError` | Wrong output path | Confirm `output/models/` (no s) vs `outputs/` (with s) |


---

## Output Files Summary

| File | Stage | Description |
|------|-------|-------------|
| `outputs/validation_report.json` | 1 | Data quality report |
| `outputs/charts/chart1–3_*.png` | 1 | Data quality charts |
| `output/models/model_summary.json` | 4 | Model metrics and winner |
| `output/models/credit_scores.csv` | 4 | Per-customer scores and tiers |
| `output/models/charts/chart4–5_*.png` | 4 | Model validation charts |
| `outputs/reports/credit_risk_report.html` | 6 | Final professional report |



# 💳 Credit Risk Scorecard

An end-to-end credit risk scoring pipeline built on the **UCI Default of Credit Card Clients dataset** (30,000 records). The project trains Logistic Regression and Gradient Boosting models, engineers six domain-driven features, validates performance against industry benchmarks, and maps predicted default probabilities to a FICO-aligned 300–850 credit score scale.

***

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Credit Scorecard Design](#-credit-scorecard-design)
- [How to Run](#-how-to-run)
- [CI/CD](#-cicd)
- [Requirements](#-requirements)
- [Results](#-results)

***

## 🎯 Project Overview

| Item | Detail |
|---|---|
| **Goal** | Predict probability of credit card default; convert to an interpretable score |
| **Dataset** | UCI ML Repo — Default of Credit Card Clients (ID 350) |
| **Records** | 30,000 customers |
| **Target** | `default.payment.next.month` (binary: 1 = default) |
| **Default rate** | ~22% |
| **Models** | Logistic Regression + Gradient Boosting Classifier |
| **Score range** | 300–850 (FICO-aligned) |

***

## 📦 Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Fetched programmatically via `ucimlrepo`:
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=350)
```

### Key Columns (after rename)

| Column | Description |
|---|---|
| `LIMIT_BAL` | Credit limit (NTD) |
| `SEX` | Gender (1=Male, 2=Female) |
| `EDUCATION` | Education level (1=Graduate, 2=University, 3=High school, 4=Other) |
| `MARRIAGE` | Marital status (1=Married, 2=Single, 3=Other) |
| `AGE` | Age in years |
| `PAY_0` to `PAY_6` | Repayment status for months Sep–Apr (-2=No consumption, -1=Paid in full, 0=Min paid, 1–8=Months delayed) |
| `BILL_AMT1` to `BILL_AMT6` | Statement balance for months Sep–Apr (NTD) |
| `PAY_AMT1` to `PAY_AMT6` | Payment amounts for months Sep–Apr (NTD) |
| `DEFAULT` | Target: 1 = defaulted next month, 0 = did not default |

### Data Cleaning

- Columns renamed from `X1`–`X23` to readable names
- `EDUCATION` values `{0, 5, 6}` recoded to `4` (Other — undocumented in original paper)
- `MARRIAGE` value `0` recoded to `3` (Other)
- Duplicate rows dropped

***

## 📁 Project Structure

```
credit-risk-scorecard/
│
├── scripts/
│   ├── validate_data.py          # Stage 1 — data validation (schema, types, ranges, default rate)
│   ├── generate_synthetic_data.py # Synthetic data generator (4 scenarios for testing)
│   └── run_models.py             # Stages 2–4 — modelling, scorecard, charts
│
├── notebooks/
│   └── 01_EDA_and_Feature_Engineering.ipynb  # EDA, WoE, IV, bivariate analysis
│
├── tests/
│   └── test_validate_data.py     # 12 unit tests for validate_data.py
│
├── outputs/
│   └── models/
│       ├── credit_scores.csv     # Predicted probability + score + tier per customer
│       ├── model_summary.json    # CV AUC, Gini, KS for both models
│       └── charts/               # ROC curves, feature importance, score distribution
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: validate → test → model → upload report
│
├── requirements.txt
├── data_dictionary.md
└── README.md
```

***

## 🔄 Pipeline Stages

```
Stage 1 │ validate_data.py  ─── Schema check, missingness, bounds, default rate
   ↓
Stage 2 │ run_models.py     ─── Load UCI dataset via ucimlrepo, clean & recode
   ↓
Stage 3 │ run_models.py     ─── Feature engineering (6 new features), model training
   ↓
Stage 4 │ run_models.py     ─── Model comparison, scorecard scaling (300–850), charts
```

***

## 🔧 Feature Engineering

Six new features are engineered from the raw payment history:

| Feature | Formula | Business Rationale |
|---|---|---|
| `AVG_BILL_AMT` | Mean of `BILL_AMT1`–`BILL_AMT6` | Average outstanding balance |
| `UTILISATION_RATE` | `AVG_BILL_AMT / LIMIT_BAL` (clipped 0–1) | How much of the credit limit is being used |
| `PAYMENT_RATIO` | `AVG_PAY_AMT / (AVG_BILL_AMT + 1)` (clipped 0–5) | How much of the bill is being repaid |
| `MAX_DELAY` | `max(PAY_0, PAY_2, ..., PAY_6)` | Worst single-month payment delay |
| `MEAN_DELAY` | `mean(PAY_0, PAY_2, ..., PAY_6)` | Average payment delay over 6 months |
| `DELAY_COUNT` | Count of months where `PAY > 0` | Number of months with any delay |
| `BILL_TREND` | `BILL_AMT1 - BILL_AMT6` | Whether total debt is growing (+) or shrinking (–) |
| `PAY_TREND` | `PAY_AMT1 - PAY_AMT6` | Whether payment amounts are increasing (+) or decreasing (–) |

***

## 📊 Model Performance

Both models are evaluated using 5-fold Stratified Cross-Validation. The winner is selected by CV AUC. A ≤0.01 AUC difference favours Logistic Regression for interpretability.

| Metric | Logistic Regression | Gradient Boosting | Industry Threshold |
|---|---|---|---|
| CV AUC | ~0.776 | ~0.783 | > 0.70 ✅ |
| Gini Coefficient | ~0.552 | ~0.566 | > 0.40 ✅ |
| KS Statistic | ~0.412 | ~0.424 | > 0.30 ✅ |

> **Model choice:** Gradient Boosting selected as best model. Logistic Regression retained as interpretable fallback for regulatory contexts.

***

## 💳 Credit Scorecard Design

### Probability → Score Mapping

The predicted default probability `p` is linearly mapped to the 300–850 scale:

```
Credit Score = 850 − (p × 550)
```

- `p = 0.0` (no risk) → Score = **850** (Exceptional)
- `p = 1.0` (certain default) → Score = **300** (Very Poor)
- Scores are clipped to `[300, 850]`

### Score Tiers

| Score Band | Tier | Typical Action |
|---|---|---|
| 800–850 | Exceptional | Approve, best rate |
| 740–799 | Very Good | Approve, preferred rate |
| 670–739 | Good | Approve, standard rate |
| 580–669 | Fair | Conditional approval |
| 500–579 | Poor | Decline or secured product |
| 300–499 | Very Poor | Decline |

> **Note:** This mapping uses a simplified linear transform. A production scorecard would use a PDO (Points to Double the Odds) log-odds calibration to ensure score intervals correspond to consistent risk multiples.

***

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate data (optional — for synthetic scenarios)

```bash
# Generate synthetic test data first
python scripts/generate_synthetic_data.py --rows 1000 --output data/synthetic_data.csv

# Run validation
python scripts/validate_data.py --input data/synthetic_data.csv
python scripts/validate_data.py --input data/synthetic_data.csv --strict   # warnings → failures
```

### 3. Run the full modelling pipeline

```bash
python scripts/run_models.py
python scripts/run_models.py --output my_output_dir   # custom output path
```

### 4. Run unit tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=scripts --cov-report=term-missing
```

### 5. Open the EDA notebook

```bash
jupyter notebook notebooks/01_EDA_and_Feature_Engineering.ipynb
```

***

## ⚙️ CI/CD

Every push to `main` triggers the GitHub Actions workflow (`.github/workflows/ci.yml`):

1. ✅ Install dependencies
2. ✅ Generate synthetic data
3. ✅ Run `validate_data.py`
4. ✅ Run `pytest` with coverage
5. ✅ Run full model pipeline (`run_models.py`)
6. ✅ Upload HTML scorecard report as a build artifact

***

## 📦 Requirements

```
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
xgboost==2.0.3
lightgbm==4.3.0
matplotlib==3.8.2
seaborn==0.13.2
jinja2==3.1.3
scipy==1.12.0
imbalanced-learn==0.12.0
groq==0.5.0
ucimlrepo
pytest
pytest-cov
```

***

## 📈 Results

After running the pipeline, `outputs/models/` contains:

| File | Content |
|---|---|
| `credit_scores.csv` | Predicted probability, credit score, and tier for all 30,000 customers |
| `model_summary.json` | CV AUC, Gini, KS for both models + winner |
| `charts/chart4_roc_and_importance.png` | ROC curve comparison + GBM feature importance |
| `charts/chart5_score_distribution.png` | Credit score distribution across all customers |

***

## 👤 Author

**Abhinav Srivastav**  
Analytics & Credit Risk | ISB AMPBA  
[GitHub](https://github.com/Abhinav19-isb) · [LinkedIn](https://www.linkedin.com/in/abhinav-srivastav-isb)

***

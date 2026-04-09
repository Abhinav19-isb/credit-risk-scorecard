---
name: credit-risk-scorecard
description: "Build a complete credit risk scorecard from any credit card transaction
data. Use this skill whenever the user asks to build a credit scorecard, assess default
risk, segment customers by credit score, compare Logistic Regression vs Gradient
Boosting for credit risk, compute KS statistic or Gini coefficient, or generate a
credit risk report. Trigger for requests like 'build me a credit scorecard', 'score
these customers for default risk', 'which customers are high risk', or 'run a credit
risk model on this data'. Accepts any CSV, Excel, or Parquet file with credit data, or
auto-downloads the UCI Credit Card Default Dataset if no file is provided."
---

# Credit Risk Scorecard Skill

Build a production-ready credit risk scorecard from any credit card dataset.
Accepts user-provided files (CSV, Excel, Parquet) or auto-downloads the UCI
benchmark dataset. Outputs a professional HTML report with score tiers and
business recommendations.

## When To Use

- User provides a credit data file and wants a risk scorecard
- User asks for default probability scores on their customer base
- User wants to compare Logistic Regression vs Gradient Boosting for credit risk
- User asks for KS Statistic, Gini, or AUC on a credit dataset
- User wants customers segmented into risk tiers (Very Poor to Exceptional)
- No file provided — use UCI Credit Card Default Dataset as benchmark

## Step 1 — Gather Inputs

### Required from User
1. **Data file** (Optional) — CSV, Excel (.xlsx), or Parquet file path
   - If not provided, auto-download UCI Credit Card Default Dataset via ucimlrepo
   - Example: "use my file at data/customers.csv"

2. **Target column** (Optional, default: `default.payment.next.month`)
   - Binary column: 1 = default, 0 = no default
   - If different name, user must specify: "my target column is called DEFAULTED"

3. **Score range** (Optional, default: 300–850 FICO scale)
   - User can specify: "use a 0–1000 score range"

### Minimum Required Columns
The input file must contain AT LEAST these column types:

| Column Type | UCI Name | Acceptable Alternatives |
|-------------|----------|------------------------|
| Credit limit | LIMIT_BAL | credit_limit, limit, CREDIT_LIMIT |
| Payment status (recent) | PAY_0 | pay_status, payment_status, PAY_1 |
| Bill amounts (1-6 months) | BILL_AMT1..6 | bill_amount, statement_balance |
| Payment amounts (1-6 months) | PAY_AMT1..6 | payment_amount, amount_paid |
| Target variable | default.payment.next.month | default, defaulted, TARGET |

### Optional Columns
| Column | UCI Name | Behaviour if Missing |
|--------|----------|----------------------|
| Age | AGE | Excluded from features, no impact on core model |
| Sex | SEX | Excluded from features |
| Education | EDUCATION | Excluded from features |
| Marriage | MARRIAGE | Excluded from features |
| Customer ID | ID | Auto-generated if missing |

### Validation Rules
- Minimum 500 rows required (warn if < 2,000)
- Target column must be binary (0/1 or True/False)
- At least one PAY_STATUS column required
- At least one BILL_AMT column required
- At least one PAY_AMT column required
- Default rate must be between 1% and 70% (warn outside this range)
- Duplicate rows: auto-remove, log count in report
- Null values > 20% in any column: exclude that column with warning
- Null values < 20%: median impute numeric, mode impute categorical

## Step 2 — Load Data


### Path A — User Provided File

Run validate_data.py with the --input flag pointing to your file.

    python scripts/validate_data.py --input data/customers.csv --target default.payment.next.month --output outputs/validation_report.json

Supported input formats:

    CSV (.csv)
    Excel (.xlsx)
    Parquet (.parquet)

The --target flag specifies your binary default column name.
If not provided, defaults to: default.payment.next.month

### Path B — Auto-Download UCI Dataset (default fallback)

If no file is provided, the skill auto-downloads the UCI Credit Card Default Dataset.

    python scripts/validate_data.py --input uci --output outputs/validation_report.json

## Step 3 — Run Models

    python scripts/run_models.py --input outputs/validated_data.parquet --target default.payment.next.month --score-min 300 --score-max 850 --output outputs/models

## Step 4 — Generate Report

    python scripts/generate_report.py --results outputs/models/model_summary.json --output outputs/reports/credit_risk_report.html


## Step 5 — Output

Present the HTML report to the user. Provide a brief summary of:
- Model winner and why (interpretability vs performance tradeoff)
- Portfolio default rate and high-risk tier concentration
- Top 3 business recommendations based on score tier distribution
- Any data quality warnings raised during validation

## Failure Handling

| Failure | Action |
|---------|--------|
| File not found | Ask user to re-provide path, offer UCI fallback |
| Missing required columns | List missing columns, halt pipeline, show required schema |
| Target column not binary | Ask user to confirm target column name and encoding |
| < 500 rows | Warn that results unreliable, ask user to confirm proceed |
| Default rate < 1% or > 70% | Warn of potential data issue, ask user to confirm |
| Null > 20% in column | Exclude column, warn user, proceed |

## Important Notes

1. This is NOT financial advice — include disclaimer in all reports
2. Model trained on provided data only — does not generalise beyond the input dataset
3. FICO scale (300–850) used by default — adjustable via --score-min and --score-max
4. Logistic Regression preferred when AUC difference vs Gradient Boosting is < 0.01
5. All intermediate outputs saved to outputs/ folder for audit trail

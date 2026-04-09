# Credit Risk Scorecard Builder

An LLM Skill Package that instructs an AI to perform end-to-end
credit risk analysis — from raw data to feature engineering to
dual-model scorecard to professional HTML report.

## Domain
Financial Analytics — Credit Risk

## Dataset

UCI Credit Card Default Dataset (30,000 customers, 23 features)

Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

License: CC BY 4.0

Synthetic data generator: scripts/generate_synthetic_data.py

Used for pipeline testing — generates clean, bad_data, missing_cols, and imbalanced scenarios


## Pipeline Stages
1. Data Validation and Profiling
2. Feature Engineering (utilisation rate, payment ratio, delay features)
3. Modelling — Logistic Regression Scorecard vs Gradient Boosting
4. Model Validation — KS Statistic, Gini, AUC-ROC
5. Credit Score Generation (FICO-aligned 300-850 scale)
6. Professional HTML Report Generation

## Model Results

| Metric | Logistic Regression (Winner) | Gradient Boosting |
|--------|------------------------------|-------------------|
| CV AUC | 0.783 | 0.787 |
| Gini | 0.566 | 0.574 |
| KS Statistic | 0.424 | 0.465 |

Logistic Regression selected for interpretability (AUC diff < 0.01)

## Repository Structure

    credit-risk-scorecard/
    SKILL.md                         Skill specification
    REFERENCE.md                     Domain knowledge reference
    README.md                        This file
    requirements.txt                 Python dependencies
    data/
        data_dictionary.md           Column definitions, source, license
    scripts/
        validate_data.py             Stage 1 - Data validation
        run_models.py                Stages 2-5 - Modelling and scoring
        generate_report.py           Stage 6 - HTML report generation
    templates/
        report_template.html         Jinja2 HTML report template
    execution_evidence/
        credit_risk_report.html      Sample generated report

## Installation

    git clone https://github.com/Abhinav19-isb/credit-risk-scorecard.git
    cd credit-risk-scorecard
    pip install -r requirements.txt

## Usage

Run the full pipeline in sequence:

    python scripts/validate_data.py --output outputs/validation_report.json
    python scripts/run_models.py --output outputs/models
    python scripts/generate_report.py

The report is saved to outputs/reports/credit_risk_report.html

## As an LLM Skill

This repo is packaged as a Claude.ai Skill. Once uploaded, trigger it with:

"Build me a credit risk scorecard and analyse default risk for credit card customers"

Claude will orchestrate the full pipeline automatically using the SKILL.md specification.

## Author
Abhinav Srivastav | AMPBA, ISB | Credit Analytics Portfolio Project
GitHub: https://github.com/Abhinav19-isb/credit-risk-scorecard


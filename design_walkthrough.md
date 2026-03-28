# Design Walkthrough — Credit Risk Scorecard Builder
**Author:** Abhinav Srivastav | AMPBA, ISB  
**Date:** March 2026  
**Repository:** https://github.com/Abhinav19-isb/credit-risk-scorecard

---

## 1. Why This Domain?

I chose Credit Risk Scorecard as my analytics skill for two reasons.

First, it directly aligns with my 4+ years of professional experience in credit card
analytics at Proclink Consulting, where I manage a portfolio of 1.1 million customers
and monitor default risk regularly. I already understand the business language —
DPD, collections triggers, limit management — so I could encode genuine domain
knowledge into REFERENCE.md rather than generic textbook definitions.

Second, credit risk is one of the most regulated and structured analytics domains.
Every model decision must be documented, justified, and auditable. This made it a
perfect candidate for LLM skill authoring — the pipeline is deterministic, the
validation thresholds are industry-defined, and the outputs must be precise.

---

## 2. Why These Pipeline Steps?

### Stage 1 — Data Validation First
In production credit risk environments, bad data is the #1 cause of model failure.
I designed Stage 1 to be a hard gate — if the overall_status is FAIL, the pipeline
stops completely. This mirrors real bank model governance where a model cannot proceed
to development without a signed-off data quality report.

The specific checks (null thresholds, value ranges, class balance) were chosen based
on real issues I've seen in credit data:
- EDUCATION/MARRIAGE recoding — undocumented values are common in survey-based data
- Class imbalance warning — 22% default rate requires handling or models will predict
  "no default" for everyone and appear 78% accurate while being useless

### Stage 2 — Feature Engineering Before Modelling
Raw transactional columns (BILL_AMT1–6, PAY_AMT1–6) are not directly informative —
their meaning comes from relationships. UTILISATION_RATE captures the credit usage
pattern; PAYMENT_RATIO captures repayment behaviour; MAX_DELAY captures the worst
risk signal. These 7 engineered features encode domain expertise that a raw ML model
would struggle to learn from 18 separate columns alone.

### Stage 3 — Two Models, Not One
Running a single model and calling it done is the most common anti-pattern in
credit risk modelling. I deliberately chose:
- **Logistic Regression** — industry standard, fully interpretable, required for
  Basel II documentation. Its coefficients can be converted to scorecard points.
- **Gradient Boosting** — captures non-linear interactions (e.g., high utilisation
  combined with recent delays is far more dangerous than either alone).

The comparison forces quantitative justification of the winner — not assumption.

### Stage 4 — Three Validation Metrics
AUC alone is insufficient for credit risk. I chose KS, Gini, and AUC because:
- **AUC** measures overall ranking ability
- **KS** measures the maximum separation at the optimal threshold — critical for
  setting collections cutoffs
- **Gini** is the standard metric used in Basel model documentation (= 2×AUC−1)

All three must pass their respective thresholds before the model is accepted.

### Stages 5–6 — Separation of Computation and Reporting
The report script reads JSON/CSV outputs — it never re-runs models. This is the
single-responsibility principle. In production, reports are regenerated daily with
new data; models are retrained quarterly. Keeping them separate means a report
format change never requires retraining.

---

## 3. Design Decisions in SKILL.md

### Exact Command Specifications
Every stage in SKILL.md specifies the exact command, expected output, and validation
check before proceeding. This was deliberately modelled on how production runbooks
are written — an LLM (or a new analyst) should be able to execute the pipeline
without asking a single clarifying question.

### Error Handling Table
The error handling section in SKILL.md was written from real bugs encountered during
development. The `KeyError: 'AGE'` error (UCI returns X1–X23, not friendly names)
and `ValueError: Input X contains NaN` (division by zero in UTILISATION_RATE) both
occurred during development and are now documented with exact fix instructions.

### Path Inconsistency Documentation
The project has an inconsistency: `output/models/` (no s) vs `outputs/` (with s).
This happened because the models script was written before the folder naming convention
was finalised. It is documented in SKILL.md's error handling table so the LLM knows
which path to use for each file — a real-world imperfection that was handled
transparently rather than hidden.

---

## 4. What Failed During Development

### Issue 1 — UCI Column Names
The UCI `ucimlrepo` library returns columns named X1–X23 and Y, not the friendly
names documented in the dataset description. This caused all column validation checks
to fail with KeyError on the first run. Fix: added a `col_map` rename dictionary
as the very first operation after loading data.

### Issue 2 — NaN in Feature Matrix
UTILISATION_RATE = avg_bill / LIMIT_BAL caused NaN when LIMIT_BAL was zero for
some customers. This crashed Logistic Regression with "Input X contains NaN".
Fix: applied `.fillna(0)` after feature engineering, documented in SKILL.md.

### Issue 3 — Target Column Name
The target column is named `Y` in the UCI API response, not
`default.payment.next.month` as documented on the UCI website. Fix: renamed Y →
DEFAULT after concat, updated all downstream references.

### Issue 4 — Path Inconsistency
`run_models.py` saved outputs to `output/models/` while `validate_data.py` saved
to `outputs/`. This caused FileNotFoundError in `generate_report.py` on first run.
Fix: hardcoded correct paths in each script and documented the inconsistency.

---

## 5. What I Would Improve With More Time

1. **Proper train/test split** — current model uses full dataset for both training
   and scoring. Production requires time-based splitting (train on months 1–4,
   test on months 5–6) to prevent data leakage.

2. **Weight of Evidence (WoE) encoding** — replace raw categorical values with WoE
   scores for EDUCATION, MARRIAGE, SEX. This is standard practice for logistic
   regression scorecards and improves both performance and interpretability.

3. **SMOTE for class imbalance** — evaluate oversampling against class_weight
   using precision-recall AUC, not just ROC AUC, to find the optimal threshold
   for the collections use case.

4. **PSI monitoring dashboard** — add a Stage 7 that computes Population Stability
   Index between the development dataset and any new scoring dataset, with automated
   alerts if PSI > 0.25.

5. **Interactive HTML report** — add Plotly charts with hover tooltips and a
   parameter sensitivity slider showing how score tier distribution changes with
   different model thresholds.

---

## 6. Portfolio & Career Relevance

This project demonstrates four skills that are directly relevant to Data Scientist
and Analytics Manager roles:

- **End-to-end ML pipeline ownership** — from raw data to deployed scorecard
- **Domain expertise encoding** — REFERENCE.md shows deep credit risk knowledge,
  not just generic ML application
- **Production engineering mindset** — separation of concerns, error handling,
  reproducibility (random_state=42 throughout), structured outputs
- **LLM orchestration** — authoring a skill that instructs an AI to perform
  complex domain-specific analysis is an emerging and highly valued capability

The GitHub repository serves as a live portfolio piece demonstrating all of the above.


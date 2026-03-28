# REFERENCE.md - Credit Risk Scorecard Domian Knowledge

This file contains all domain specific knowledge for the Credit Risk Scorecard skillm
The LLM must consult this file during Stage 5 (Insights Generation) to interpret resuts, assign labels, and generate recommendations. Do not hallucinate domain facts —
use only the benchmarks and definitions documented here.

---

## 1. Credit Risk Fundamentals

### What is a Credit Scorecard?
A credit scorecard is a statistical model that assigns a numeric score to each customer represting thier probability of defaulting on a loan or credit card payment. 
Higher score = Lower risk.
Scores are used by banks to make lending decisions, set credit limits, and prioritse collection activity.

### Key Industry Terms
| Term | Definition |
|------|-----------|
| Default | Failure to make required payment for 90+ days (varies by institution) |
| PD (Probability of Default) | Model output: likelihood customer will default in next 12 months |
| LGD (Loss Given Default) | % of outstanding balance lost if customer defaults |
| EAD (Exposure at Default) | Total outstanding balance at time of default |
| Expected Loss | PD × LGD × EAD — core Basel II/III risk metric |
| KS Statistic | Max separation between default/non-default score distributions |
| Gini Coefficient | 2 × AUC − 1; measures model discrimination power |
| PSI | Population Stability Index — measures score distribution shift over time |
| DPD | Days Past Due — number of days a payment is overdue |
| Vintage | The month/year a credit account was opened |

---

## 2. Feature Definition & Formulas

All engineered features used in this scorecard:

### UTILIZATION_RATE
- **Formula:** avg(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6) / LIMIT_BAL
- **Clipping:** [0, 1]
- **Business meaning:** Measures how much of the credit limit is being used on average
- **Risk signal:** Higher utilisation → higher default risk
- **Industry benchmark:** Utilisation > 0.80 is considered high risk; < 0.30 is low risk

### PAYMENT_RATIO
- **Formula:** avg(PAY_AMT1–6) / (avg(BILL_AMT1–6) + 1)
- **Clipping:** [0, 5]
- **Business meaning:** What fraction of the outstanding bill is being repaid each month
- **Risk signal:** Low payment ratio → customer is revolving balance → higher risk
- **Industry benchmark:** Ratio < 0.05 (minimum payment only) signals high stress

### MAX_DELAY
- **Formula:** max(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
- **Business meaning:** Worst payment delay experienced in the last 6 months
- **Risk signal:** Single severe delay (MAX_DELAY ≥ 2) is a strong default predictor
- **PAY encoding:** -2=no consumption, -1=paid in full, 0=minimum paid, 1–9=months delayed

### MEAN_DELAY
- **Formula:** mean(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
- **Business Meaning:** Average payment delay across 6 months
- **Risk signal:** Distinguishes chronic late payers (high mean) from one-time events

### DELAY_COUNT
- **Formula:** count of months where PAY_x > 0
- **Range:** 0-6
- **Business meaning:** How many months in the last 6 had any payment delay
- **Risk signal:** DELAY_COUNT ≥ 3 indicates persistent payment difficulty

### BILL_TREND
- **Formula:** BILL_AMT1 − BILL_AMT6 (most recent minus oldest)
- **Business meaning:** Direction of debt accumulation
- **Risk signal:** Positive value = debt growing = higher risk; Negative = paying down debt

### PAY_TREND
- **Formula:** PAY_AMT1 − PAY_AMT6 (most recent minus oldest)
- **Business meaning:** Whether payment behaviour is improving or deteriorating
- **Risk signal:** Negative value = paying less recently = deteriorating behaviour

---

## 3. Model Performance Benchmarks

### AUC-ROC Benchmarks (Credit Risk Industry Standard)
| AUC Range | Rating | Action |
|-----------|--------|--------|
| ≥ 0.80 | Excellent | Deploy with standard monitoring |
| 0.70–0.79 | Good | Deploy with enhanced monitoring |
| 0.60–0.69 | Moderate | Requires feature engineering improvements |
| < 0.60 | Poor | Do not deploy — redesign required |

### Gini Coefficient Benchmarks 
| Gini Range | Rating |
|-----------|--------|
| ≥ 0.60 | Excellent |
| 0.40–0.59 | Good |
| 0.20–0.39 | Moderate |
| < 0.20 | Poor |

### KS Statistic Benchmarks
| KS Range | Rating | Meaning |
|----------|--------|---------|
| ≥ 0.40 | Strong | Model clearly separates defaulters from non-defaulters |
| 0.30–0.39 | Acceptable | Model meets minimum deployment standards |
| 0.20–0.29 | Weak | Marginal — review features before deployment |
| < 0.20 | Poor | Do not deploy |

### PSI (Population Stability Index) Thresholds
| PSI Value | Action |
|-----------|--------|
| < 0.10 | No action — population is stable |
| 0.10–0.25 | Monitor closely — minor shift detected |
| > 0.25 | Trigger model review — significant shift |

---

## 4. Credit Scorecard Tier Definitions & Recommended Actions

### Score Tiers (300–850 FICO-Style Scale)

| Tier | Score Range | Default Probability | Customer Profile |
|------|-------------|--------------------|--------------------|
| Exceptional | 800–850 | < 2% | Excellent payment history, low utilisation, zero delays |
| Very Good | 740–800 | 2–5% | Consistent on-time payments, moderate utilisation |
| Good | 670–740 | 5–15% | Mostly on-time, occasional minor delays |
| Fair | 580–670 | 15–30% | Some delays, moderate-high utilisation |
| Poor | 500–580 | 30–50% | Frequent delays, high utilisation |
| Very Poor | 300–500 | > 50% | Severe delays, maximum utilisation, high default probability |

### Recommended Actions Per Tier
**Exceptional (800–850)**
- Proactively offer credit limit increases (25–50% increase)
- Cross-sell premium card products (travel rewards, cashback)
- Target for pre-approved personal loan offers
- Use as benchmark cohort for acquisition model development

**Very Good (740–800)**
- Eligible for moderate limit increases (10–25%)
- Send loyalty reward offers to maintain engagement
- Low priority for collections or risk intervention

**Good (670–740)**
- Standard monitoring cadence (monthly statement review)
- Send timely payment reminders 5 days before due date
- Eligible for limit increase after 12 months of clean history

**Fair (580–670)**
- Increase monitoring frequency (bi-weekly review)
- Send payment reminders 7 days before due date
- Offer balance transfer at lower interest rate to reduce utilisation
- Block new credit limit increases until improvement shown

**Poor (500–580)**
- Place on watchlist for early collections intervention
- Reduce credit limit to current outstanding balance
- Outbound call at 15 DPD (Days Past Due)
- Offer structured repayment plan

**Very Poor (300–500)**
- Immediate collections review
- Suspend card for new transactions if MAX_DELAY ≥ 3
- Assign to specialist collections team
- Offer debt restructuring or settlement programme
- Flag for potential write-off provisioning if unresponsive

---

## 5. Algorithm Selection Criteria

### When to use Logistic Regression
- When model interpretability is required (regulatory audit, Basel II documentation)
- When coefficients need to map to scorecard points (Weight of Evidence approach)
- When dataset is small (< 10,000 records)
- When linear relationships dominate

### When to Use Gradient Boosting
- When predictive performance is the primary goal
- When non-linear feature interactions are suspected (e.g., high utilisation AND recent delays)
- When dataset is large (> 10,000 records)
- When slight loss of interpretability is acceptable

### Model Selection Rule for This Skill
1. Compute CV AUC for both models using 5-fold StratifiedKFold
2. If GB AUC > LR AUC by more than 0.01 → select Gradient Boosting
3. If difference ≤ 0.01 → select Logistic Regression (interpretability preference)
4. Always document the comparison — never select a model without metric justification

---

## 6. Regulatory Context (Basel II/III)

Credit risk models used by banks are governed by Basel II/III accords. Key requirements:
- **Model Documentation:** Every feature, formula, and assumption must be documented
- **Validation:** Models must be validated on out-of-sample data before deployment
- **Monitoring:** KS, Gini, and PSI must be tracked monthly post-deployment
- **Audit Trail:** All model versions, changes, and decisions must be logged
- **Challenger Framework:** Production model must always be compared against a challenger

This SKILL.md and REFERENCE.md together satisfy the documentation requirements
for a development-stage credit risk model under internal validation standards.

---

## 7. Report Interpretation Templates

Use these templates when generating business language in the report:

### AUC Interpretation
- AUC = 0.78 → *"The model correctly ranks a randomly selected defaulter above a
  randomly selected non-defaulter 78% of the time — well above the 70% industry
  deployment threshold."*

### KS Interpretation
- KS = 0.42 → *"At the optimal score threshold, the model captures 42% more defaulters
  than non-defaulters — indicating strong discriminatory power suitable for
  collections prioritisation."*

### High Risk Segment
- *"X% of the portfolio ({N} customers) fall in the Very Poor and Poor tiers.
  These customers represent disproportionate default risk and should be the primary
  focus of collections and risk mitigation strategies."*

### Low Risk Segment
- *"X% of the portfolio ({N} customers) score above 740, indicating reliable payment
  behaviour. These customers are strong candidates for credit limit increases and
  premium product cross-selling."*

---

## 8. Data Source & Licensing

| Field | Detail |
|-------|--------|
| Dataset Name | Default of Credit Card Clients |
| Provider | UCI Machine Learning Repository |
| Dataset ID | 350 |
| URL | https://archive.uci.edu/dataset/350 |
| Geography | Taiwan |
| Time Period | April–September 2005 |
| Records | 30,000 customers |
| License | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Citation | Yeh, I-C. (2009). UCI ML Repository. https://doi.org/10.24432/C55S3H |
| Preprocessing | Column renaming (X1–X23 → friendly names), duplicate removal (35 rows), categorical recoding |

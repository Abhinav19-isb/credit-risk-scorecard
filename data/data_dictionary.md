# Data Dictionary — UCI Credit Card Default Dataset

## Source
- **Name:** Default of Credit Card Clients
- **Provider:** UCI Machine Learning Repository
- **URL:** https://archive.uci.edu/dataset/350
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Records:** 30,000 customers | **Features:** 23 input + 1 target
- **Citation:** Yeh, I-C. (2009). UCI ML Repository. https://doi.org/10.24432/C55S3H

## How to Load
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=350)
X = dataset.data.features  # returns X1–X23
y = dataset.data.targets   # returns Y
```

## Column Definitions

| Original | Renamed To | Type | Description | Valid Values |
|----------|-----------|------|-------------|--------------|
| X1 | LIMIT_BAL | Float | Credit limit (NT dollar) | 10,000–1,000,000 |
| X2 | SEX | Integer | Gender | 1=Male, 2=Female |
| X3 | EDUCATION | Integer | Education level | 1=Graduate, 2=University, 3=High School, 4=Others |
| X4 | MARRIAGE | Integer | Marital status | 1=Married, 2=Single, 3=Others |
| X5 | AGE | Integer | Age in years | 21–79 |
| X6 | PAY_0 | Integer | Repayment status Sep 2005 | -2=no use,-1=paid full,0=min paid,1–9=months delayed |
| X7 | PAY_2 | Integer | Repayment status Aug 2005 | same as PAY_0 |
| X8 | PAY_3 | Integer | Repayment status Jul 2005 | same as PAY_0 |
| X9 | PAY_4 | Integer | Repayment status Jun 2005 | same as PAY_0 |
| X10 | PAY_5 | Integer | Repayment status May 2005 | same as PAY_0 |
| X11 | PAY_6 | Integer | Repayment status Apr 2005 | same as PAY_0 |
| X12 | BILL_AMT1 | Float | Bill statement Sep 2005 (NT$) | Any numeric |
| X13 | BILL_AMT2 | Float | Bill statement Aug 2005 (NT$) | Any numeric |
| X14 | BILL_AMT3 | Float | Bill statement Jul 2005 (NT$) | Any numeric |
| X15 | BILL_AMT4 | Float | Bill statement Jun 2005 (NT$) | Any numeric |
| X16 | BILL_AMT5 | Float | Bill statement May 2005 (NT$) | Any numeric |
| X17 | BILL_AMT6 | Float | Bill statement Apr 2005 (NT$) | Any numeric |
| X18 | PAY_AMT1 | Float | Payment amount Sep 2005 (NT$) | >= 0 |
| X19 | PAY_AMT2 | Float | Payment amount Aug 2005 (NT$) | >= 0 |
| X20 | PAY_AMT3 | Float | Payment amount Jul 2005 (NT$) | >= 0 |
| X21 | PAY_AMT4 | Float | Payment amount Jun 2005 (NT$) | >= 0 |
| X22 | PAY_AMT5 | Float | Payment amount May 2005 (NT$) | >= 0 |
| X23 | PAY_AMT6 | Float | Payment amount Apr 2005 (NT$) | >= 0 |
| Y | DEFAULT | Integer | **TARGET: Default next month** | 0=No, 1=Yes |

## Known Data Quality Notes
- EDUCATION has undocumented values (0, 5, 6) — recoded to 4 (Others)
- MARRIAGE has undocumented value (0) — recoded to 3 (Others)
- 35 duplicate rows removed before modelling
- No missing values in original dataset
- Class imbalance: 22.1% default rate — handled via class_weight='balanced'

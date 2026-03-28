# Scenario 3 — Parameter Sensitivity Test

## Purpose
Demonstrate how changing key hyperparameters affects model performance,
and verify the skill can evaluate and justify parameter choices
using benchmarks from REFERENCE.md.

## Parameter Changes
| Parameter | Default | Changed To | Reason |
|-----------|---------|-----------|--------|
| n_estimators | 100 | 200 | More trees = potentially better fit |
| learning_rate | 0.1 | 0.05 | Slower learning + more trees |
| cv_folds | 5 | 10 | More rigorous cross-validation |

## Results Comparison
| Metric | Original Run | New Parameters | Change |
|--------|-------------|----------------|--------|
| CV AUC | 0.7842 | 0.7901 | +0.0059 |
| Gini | 0.6029 | 0.6118 | +0.0089 |
| KS Statistic | 0.4536 | 0.4612 | +0.0076 |

## LLM Recommendation
- Marginal improvement (+0.0059 AUC) does not justify 2x computational cost
  for batch/offline scoring
- Recommend keeping default parameters for production
- New parameters suitable for quarterly retraining with larger dataset

## Overall Test Status: ✅ SKILL CORRECTLY EVALUATED PARAMETER TRADE-OFFS

## Full LLM Conversation
https://claude.ai/share/ba5cbafd-8e7f-4722-9749-528c9677c92f

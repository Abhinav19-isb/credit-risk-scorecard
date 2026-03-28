# Scenario 2 — Bad Data / Error Handling Test

## Purpose
Verify that the skill correctly identifies data quality violations,
stops the pipeline, and provides clear actionable error messages
rather than crashing silently.

## Injected Data Problems
- 9 required columns missing: PAY_0, PAY_2–6, BILL_AMT4–6
- 3 columns with critical null violations (> 30% threshold):
  - LIMIT_BAL: 34.2% nulls
  - EDUCATION: 41.7% nulls
  - AGE: 38.5% nulls
- Duplicate rows: 4,820 (16.07%) — exceeds 5% warning threshold
- Overall validation status: FAIL

## Expected Skill Behaviour
- Pipeline must STOP at Stage 1 — must not proceed to modelling
- All violations must be listed with exact column names and percentages
- Clear fix instructions must be provided to the user
- No cryptic Python errors — human-readable explanation required

## Overall Test Status: ✅ SKILL CORRECTLY IDENTIFIED ALL VIOLATIONS

## Full LLM Conversation
https://claude.ai/share/220575e8-dfa8-41a1-a875-bde62a09d232

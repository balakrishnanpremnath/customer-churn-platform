# Final Held-Out Test Results

## Evaluation Protocol

The final model was frozen before the held-out
evaluation and committed to GitHub.

- Model: RandomForestClassifier
- Training snapshots: 14,987
- Test snapshots: 2,772
- Test snapshot date: 2011-09-01
- Target: 90-day purchase inactivity
- Features: 15 historical customer features
- Calibration: Disabled
- Classification threshold: 0.5
- No test-driven hyperparameter tuning

## Final Performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7361 |
| Average precision | 0.5956 |
| Precision | 0.5365 |
| Recall | 0.8163 |
| F1 | 0.6475 |
| Balanced accuracy | 0.6800 |
| Brier score | 0.2173 |

Test inactivity prevalence: 39.29%.

## Confusion Matrix

| | Predicted Active | Predicted Inactive |
|---|---:|---:|
| Actually Active | 915 | 768 |
| Actually Inactive | 200 | 889 |

## Business-Oriented Risk Ranking

Contact budget: 500 customers.

| Metric | Random Forest | Recency |
|---|---:|---:|
| Captured inactive customers | 312 | 311 |
| Precision@500 | 0.6240 | 0.6220 |
| Recall@500 | 0.2865 | 0.2856 |
| Lift@500 | 1.5884 | 1.5833 |

Random Forest captured only one additional
inactive customer at the fixed budget.

## Interpretation

The model achieved moderate discrimination
on the held-out test period.

However, its Top-500 ranking performance was
almost identical to a simple recency baseline.

The validation-period ranking advantage did
not meaningfully carry over to the test period.

This does not demonstrate that a retention
campaign improves customer outcomes.

## Research Limitations

- Results cover one historical test period.
- Some customers occur in multiple snapshots.
- Inactivity is a proxy for customer churn.
- Changes in customer behavior and class
  prevalence may affect performance.
- Model ranking quality is not causal uplift.
- No new model selection will use test results.

## Reproducibility

Frozen configuration:
configs/final_model.json

Frozen evaluation script:
scripts/final_heldout_evaluation.py

Final test dataset SHA-256:
8847fe947742cd79a00bb26fa4114a8dd1f7dbac5241a7863596bc6649ea690b

Evaluation date: 2026-10-18

## Conclusion

A tuned Random Forest can identify customers
at higher risk of future purchase inactivity.

However, the held-out test shows that a
simple recency ranking remains a strong and
competitive operational baseline.

Further research should investigate temporal
robustness and the effectiveness of actual
retention interventions.
# Advanced Machine Learning Results

## Objective

Compare classification models for predicting
90-day customer purchase inactivity.

## Dataset

- Training snapshots: 12,328
- Validation snapshots: 2,659
- ML features: 15
- Held-out test set: Not evaluated

## Validation Results

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---:|---:|---:|---:|
| Dummy | 0.5000 | 0.4994 | 0.6662 | 0.2508 |
| Logistic Regression | 0.7493 | 0.6948 | 0.7180 | 0.2010 |
| Random Forest | 0.7511 | 0.6952 | 0.7294 | 0.1999 |
| XGBoost | 0.7477 | 0.6924 | 0.7282 | 0.2011 |

## Findings

Random Forest achieved the highest validation
ROC-AUC and F1-score.

XGBoost achieved the highest recall among
the three predictive models.

The performance differences are small and
do not establish statistical superiority.

## Research Limitations

- Churn is an inactivity-based proxy.
- Validation covers one prediction date.
- Hyperparameter tuning remains incomplete.
- Probability calibration requires evaluation.
- Final test data remains untouched.

## Next Steps

1. Evaluate validation-based threshold selection.
2. Assess probability calibration.
3. Examine temporal robustness.
4. Freeze the selected model and configuration.
5. Evaluate the held-out test set once.
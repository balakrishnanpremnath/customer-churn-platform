# Baseline Machine Learning Results

## Experiment

**Task:** Predict the absence of customer purchases during the next 90 days using historical transaction behavior.

**Dataset:** UCI Online Retail II, transformed into customer-level temporal snapshots.

**Training:** 12,328 snapshots across four historical cutoff dates.

**Validation:** 2,659 snapshots at 2011-06-01.

**Held-out test:** 2,772 snapshots at 2011-09-01; not evaluated in this experiment.

**Features:** 15 historical customer behavioral features.

## Validation Results

| Metric | Dummy Classifier | Logistic Regression |
| --- | ---: | ---: |
| ROC-AUC | 0.5000 | 0.7493 |
| PR-AUC | 0.4994 | 0.6948 |
| Precision | 0.4994 | 0.6394 |
| Recall | 1.0000 | 0.8185 |
| F1-score | 0.6662 | 0.7180 |
| Balanced accuracy | 0.5000 | 0.6790 |
| Brier score | 0.2508 | 0.2010 |

## Interpretation

Logistic Regression outperformed the prior-based Dummy Classifier on discrimination, precision, F1, balanced accuracy, and Brier score. The Dummy Classifier predicted the positive class for all validation cases, producing recall of 1.0 without meaningful discrimination.

## Limitations

- Churn is defined as a 90-day inactivity proxy, not confirmed permanent customer attrition.
- Validation evaluates a single future snapshot, so wider temporal robustness is not yet established.
- The 0.5 classification threshold has not been optimized for retention costs or business capacity.
- Return transactions are not yet incorporated into the purchase-only feature pipeline.
- Model performance does not demonstrate causal effects of retention interventions.
- The held-out test set remains untouched until model selection is finalized.

## Next Steps

1. Add model-evaluation unit tests.
2. Compare Random Forest and XGBoost on validation data.
3. Assess calibration and cost-sensitive thresholds.
4. Freeze the selected model before final held-out testing.

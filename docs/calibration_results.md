# Probability Calibration Experiment

## Objective

Assess whether sigmoid calibration improves the
reliability of Random Forest churn probabilities.

## Experimental Design

- Model: Tuned Random Forest
- Calibration method: Sigmoid logistic calibration
- Calibration data: 9,659 chronological OOF snapshots
- Validation data: 2,659 customer snapshots
- Held-out test: Not evaluated

## June 2011 Validation Results

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| ROC-AUC | 0.7553 | 0.7553 |
| PR-AUC | 0.6997 | 0.6997 |
| Brier Score | 0.1990 | 0.2075 |
| Log Loss | 0.5797 | 0.6008 |

## Findings

Sigmoid calibration did not improve probability
quality on the validation period.

Brier Score and Log Loss both increased.

ROC-AUC and PR-AUC remained unchanged because
the fitted calibration transformation preserves
the ordering of prediction scores.

## Decision

Retain the uncalibrated tuned Random Forest
as the current preferred model.

This choice is based on validation performance
and is not a claim of statistical superiority.

## Limitations

- Calibration was evaluated on one future period.
- Data distribution may change over time.
- Churn is defined using a purchase-inactivity proxy.
- Probability reliability may differ across risk groups.
- The final held-out test remains untouched.

## Next Steps

1. Evaluate retention prioritization at fixed budgets.
2. Compare precision and recall at different thresholds.
3. Freeze final model-selection decisions.
4. Conduct final held-out evaluation.
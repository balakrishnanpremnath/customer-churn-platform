# Bootstrap Uncertainty Analysis

## Objective

Estimate uncertainty in validation ROC-AUC
and customer risk-ranking performance.

## Experimental Setup

- Evaluation period: June 2011
- Validation customers: 2,659
- Model: Uncalibrated tuned Random Forest
- Bootstrap repetitions: 1,000
- Confidence level: 95%
- Random state: 42

## ROC-AUC Results

| Metric | Result |
|---|---|
| Observed ROC-AUC | 0.7553 |
| Bootstrap 95% CI | [0.736, 0.773] |

## Retention Ranking Comparison

| Budget | RF Captured | Recency Captured | Difference | 95% CI |
|---|---:|---:|---:|---|
| 100 | 74 | 76 | -2 | [-15.025, 9] |
| 500 | 370 | 346 | +24 | [4, 44] |

## Interpretation

At a contact budget of 100, the confidence
interval includes zero. Superiority of Random
Forest is not established.

At a contact budget of 500, the estimated
difference favors Random Forest under the
specified bootstrap procedure.

## Limitations

- Estimates are conditional on fitted predictions.
- Model retraining uncertainty is not included.
- Bootstrap samples may contain duplicate customers.
- Only one temporal validation period is studied.
- Model selection also used validation evidence.
- Results do not establish causal retention impact.
- Held-out test data remains untouched.

## Next Steps

1. Complete automated bootstrap tests.
2. Freeze the selected model configuration.
3. Document final evaluation protocol.
4. Conduct the final held-out test evaluation.
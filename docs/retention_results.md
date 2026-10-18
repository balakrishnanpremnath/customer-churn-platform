# Customer Retention Prioritization Results

## Objective

Evaluate whether the tuned Random Forest
improves customer risk prioritization compared
with a simple recency-based baseline.

## Evaluation Dataset

- Period: June 2011
- Customers: 2,659
- Inactive customers: 1,328
- Inactivity rate: 49.94%
- Held-out test: Not evaluated

## Comparison

| Budget | RF Captures | Recency Captures | RF Precision |
|---|---:|---:|---:|
| 50 | 38 | 37 | 0.760 |
| 100 | 74 | 76 | 0.740 |
| 250 | 181 | 177 | 0.724 |
| 500 | 370 | 346 | 0.740 |

## Key Findings

1. Random Forest outperformed recency ranking
   at contact budgets of 50, 250 and 500.

2. Recency ranking performed better at
   a contact budget of 100.

3. At a budget of 500, Random Forest
   identified 24 additional inactive customers
   compared with recency ranking.

4. Random Forest achieved Lift@500 of 1.4817.

## Limitations

- Results are from one validation period.
- Inactivity is a proxy for customer churn.
- Small ranking differences may be due to chance.
- Ranking ties require further investigation.
- No retention treatment outcomes are available.
- Intervention effectiveness cannot be established.
- Final held-out test data remains untouched.

## Conclusion

Random Forest provides promising customer-risk
prioritization, especially at larger contact
budgets, but is not consistently superior
across all budgets.

Further temporal and statistical evaluation
is necessary before deployment claims.

## Next Steps

1. Assess ranking uncertainty and stability.
2. Finalize model and threshold selection rules.
3. Freeze the model configuration.
4. Evaluate the held-out test once.
5. Build Explainable AI and the dashboard.
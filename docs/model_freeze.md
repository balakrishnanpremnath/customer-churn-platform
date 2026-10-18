# Final Model Freeze Protocol

## Objective

Predict whether an observed customer will have no
recorded purchases during the following 90 days.

This is an inactivity proxy, not evidence of
permanent customer churn.

## Selected Model

RandomForestClassifier

- n_estimators: 200
- max_depth: 6
- min_samples_leaf: 5
- max_features: sqrt
- random_state: 42
- n_jobs: 4
- Probability calibration: Disabled
- Classification threshold: 0.5

Other estimator defaults follow the pinned
scikit-learn version in requirements.txt.

## Feature Engineering

15 numeric features computed using historical
transactions within a 180-day lookback window.

Customer IDs, dates and dataset split labels
are not used as model predictors.

## Data Splitting

Training:
- June 2010
- September 2010
- December 2010
- March 2011

Validation:
- June 2011

Final Held-Out Test:
- September 2011

Final model will be fitted on 14,987
train + validation customer snapshots.

Held-out test contains 2,772 snapshots.

## Model Selection

Hyperparameters were selected using
chronological training-period cross-validation.

June 2011 was used for model assessment and
model-selection decisions.

The final model configuration is frozen
before evaluating September 2011.

## Final Evaluation

Primary metric:
- ROC-AUC

Secondary metrics:
- Average precision
- Brier score
- Balanced accuracy
- Precision
- Recall
- F1

Business-oriented evaluation:
- Captured inactive customers at Top 500
- Precision at Top 500
- Comparison against recency-based ranking

Ranking must use a documented deterministic
tie-breaking rule.

## Test Set Policy

1. Never tune hyperparameters using test results.
2. Do not select another model using test results.
3. Refit the frozen model on train + validation.
4. Evaluate once on September 2011.
5. Report all results, including weak performance.
6. Do not describe risk ranking as causal uplift.

## Research Limitations

- Customer snapshots repeat across time.
- Results may vary during different periods.
- Inactivity is not necessarily permanent churn.
- Historical purchases are observational data.
- Retention campaign effectiveness is untested.
- Validation uncertainty does not include all
  model-selection or temporal uncertainty.
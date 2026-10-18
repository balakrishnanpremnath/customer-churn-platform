# Random Forest Hyperparameter Optimization

## Research Objective

Optimize Random Forest for predicting 90-day customer purchase inactivity using chronological cross-validation.

## Experimental Setup

- Dataset: UCI Online Retail II
- Training snapshots: 12,328
- Validation snapshots: 2,659
- Features: 15 historical behavioral features
- Hyperparameter configurations: 4
- Temporal cross-validation folds: 3
- Primary selection metric: Mean ROC-AUC
- Random state: 42
- Held-out test set: Not evaluated

## Hyperparameter Search

| Max Depth | Minimum Leaf Samples | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Brier |
| --- | ---: | ---: | ---: | ---: |
| 6 | 5 | 0.7401 | 0.7104 | 0.2166 |
| 6 | 15 | 0.7400 | 0.7093 | 0.2168 |
| 10 | 5 | 0.7332 | 0.7008 | 0.2190 |
| 10 | 15 | 0.7359 | 0.7034 | 0.2180 |

## Selected Configuration

- n_estimators: 200
- max_depth: 6
- min_samples_leaf: 5
- max_features: sqrt
- random_state: 42

Selected using the highest mean ROC-AUC across training-period temporal folds.

## Temporal Cross-Validation Results

| Validation Date | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: |
| 2010-09-01 | 0.7187 | 0.5653 | 0.2207 |
| 2010-12-01 | 0.7514 | 0.7986 | 0.2302 |
| 2011-03-01 | 0.7502 | 0.7672 | 0.1989 |

## June 2011 Validation Results

| Metric | Original Random Forest | Tuned Random Forest |
| --- | ---: | ---: |
| ROC-AUC | 0.7511 | 0.7553 |
| PR-AUC | 0.6952 | 0.6997 |
| Precision | 0.6517 | 0.6513 |
| Recall | 0.8283 | 0.8298 |
| F1-score | 0.7294 | 0.7298 |
| Balanced accuracy | 0.6933 | 0.6933 |
| Brier score | 0.1999 | 0.1990 |

## Interpretation

The tuned Random Forest slightly improved ROC-AUC, PR-AUC, F1-score and Brier score on the June 2011 validation snapshot.

These improvements are small and do not establish statistically significant superiority.

## Research Limitations

- Customer churn is represented by a 90-day inactivity proxy.
- Temporal validation performance varies across historical periods.
- Four hyperparameter configurations were evaluated.
- The selected configuration is not guaranteed to be globally optimal.
- Customers may appear across multiple chronological snapshots.
- Probability calibration and business-based threshold selection are pending.
- Final held-out test evaluation has not been performed.

## Reproducibility

Run the tuning script:

```powershell
.\.venv\Scripts\python.exe -m scripts.tune_random_forest
```

Run automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Next Steps

1. Evaluate probability calibration.
2. Perform business-oriented threshold selection.
3. Assess temporal stability and uncertainty.
4. Freeze final model-selection decisions.
5. Evaluate the held-out test set once.

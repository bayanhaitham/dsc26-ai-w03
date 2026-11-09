# Results: Tawseel failed-delivery baseline

## Baseline table

Dataset: `make_parcels(4000, seed=0)`

Split: chronological 80% training / 20% testing.

| Model                          | ROC AUC | PR-AUC |  Brier | Base rate |
| ------------------------------ | ------: | -----: | -----: | --------: |
| Dummy (prior)                  |  0.5000 | 0.2200 | 0.1717 |    0.2200 |
| Rule: Not tried                |       — |      — |      — |         — |
| My model (Logistic Regression) |  0.6480 | 0.3840 | 0.1628 |    0.2200 |

- Training rows: 3,200
- Test rows: 800
- PR-AUC lift over dummy: 0.1640

## Three sentences for Sherif

My Logistic Regression model uses parcel information, customer history, and address text to identify deliveries that are more likely to fail. On the chronological test set, it achieved a PR-AUC of 0.3840 compared with 0.2200 for the dummy baseline, showing that it learned useful predictive patterns. However, its ROC-AUC of 0.6480 indicates moderate discrimination, and these results come from synthetic data, so the model needs further validation on real delivery data before practical use.

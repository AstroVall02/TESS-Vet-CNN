# Results

| Model | Accuracy (%) | ROC-AUC | Avg. precision | Planet P / R (%) | No-planet P / R (%) |
|---|---|---|---|---|---|
| Baseline: light curve + period, depth | 69.35 ± 1.13 | 0.756 ± 0.009 | 0.755 ± 0.010 | 68.2 / 76.8 | 71.2 / 61.3 |
| + catalog features | 72.21 ± 1.30 | 0.803 ± 0.010 | 0.799 ± 0.010 | 72.5 / 74.7 | 71.9 / 69.5 |
| + catalog features + Tmag | 71.99 ± 1.51 | 0.798 ± 0.011 | 0.784 ± 0.012 | 72.8 / 73.4 | 71.1 / 70.4 |

Mean ± sd over 5 training seeds on one fixed test set (n = 447, 52% planets), threshold 0.5. Catalog features vs baseline: **+2.86 points** (positive on 5/5 seeds; bootstrap 95% CI over test rows [-0.8, +6.5]). Adding Tmag: -0.22 points [-2.1, +1.7] (no clear change). Averaging the 5 seeds' probabilities gives 69.8% / 71.1% / 71.6% for the three models. With catalog features, 20.1% of test rows are misclassified by every seed and 35.8% by at least one.

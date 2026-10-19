"""
Test 1: how much do the catalog columns carry without the CNN?

Gradient-boosted trees on scalar columns only, 5-fold x 3-repeat cross-validation grouped by TIC
(so the same star never appears in both train and test). Four nested feature sets:

  A  log_period, log_depth                      (what the CNN aux branch uses today)
  B  A + log_duration
  C  B + Teff, log stellar radius, Tmag
  D  C + implied planet radius (Rjup) + duration ratio (observed / expected for a central transit)

Run:  uv run python3 catalog_baseline.py processed_labels_v2.csv
(needs scikit-learn:  uv add scikit-learn   or   uv pip install scikit-learn)
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--label-col", default="final_label")
ap.add_argument("--tic-col", default="TIC_ID")
ap.add_argument("--period-col", default="Period (days)")
ap.add_argument("--depth-col", default="Depth_ppm")
ap.add_argument("--duration-col", default="Duration_hrs")
ap.add_argument("--teff-col", default="Teff_K")
ap.add_argument("--radius-col", default="Stellar_Radius")
ap.add_argument("--tmag-col", default="TESS_Mag")
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

df = pd.read_csv(a.labels)
need = [a.label_col, a.tic_col, a.period_col, a.depth_col, a.duration_col, a.teff_col, a.radius_col, a.tmag_col]
miss = [c for c in need if c not in df.columns]
if miss:
    raise SystemExit(f"columns not found: {miss}\navailable: {list(df.columns)}")

P, depth, dur = df[a.period_col].astype(float), df[a.depth_col].astype(float), df[a.duration_col].astype(float)
Rs, Teff, Tmag = df[a.radius_col].astype(float), df[a.teff_col].astype(float), df[a.tmag_col].astype(float)

# same cleaning as the CNN: drop rows whose log features would be non-finite
keep = (P > 0) & (depth > 0) & (dur > 0) & df[a.label_col].isin([0, 1])
df, P, depth, dur, Rs, Teff, Tmag = [x[keep] for x in (df, P, depth, dur, Rs, Teff, Tmag)]
y = df[a.label_col].astype(int).values
groups = df[a.tic_col].values
print(f"{len(df)} rows, {len(set(groups))} stars, class balance {y.mean():.3f} planets")

F = pd.DataFrame({
    "log_period": np.log10(P), "log_depth": np.log10(depth), "log_duration": np.log10(dur),
    "teff": Teff, "log_rs": np.log10(Rs.where(Rs > 0)), "tmag": Tmag,
})
# Rp in Jupiter radii: sqrt(depth) * Rs * (Rsun/Rjup = 9.73)
F["rp_rjup"] = np.sqrt(depth / 1e6) * Rs * 9.73
# duration ratio: expected central-transit duration ~ 13 h (P/365.25)^(1/3) Rs^(2/3)  (crude, assumes M ~ R)
F["dur_ratio"] = dur / (13.0 * (P / 365.25) ** (1 / 3) * Rs.where(Rs > 0) ** (2 / 3))
F["log_rp"] = np.log10(F["rp_rjup"].where(F["rp_rjup"] > 0))
F["log_dur_ratio"] = np.log10(F["dur_ratio"].where(F["dur_ratio"] > 0))
print("missing values:", F.isna().sum()[F.isna().sum() > 0].to_dict())

sets = {
    "A  period+depth":            ["log_period", "log_depth"],
    "B  A + duration":            ["log_period", "log_depth", "log_duration"],
    "C  B + Teff,Rs,Tmag":        ["log_period", "log_depth", "log_duration", "teff", "log_rs", "tmag"],
    "D  C + Rp + dur_ratio":      ["log_period", "log_depth", "log_duration", "teff", "log_rs", "tmag", "log_rp", "log_dur_ratio"],
}

print(f"\n{'feature set':26s} {'accuracy':>16s} {'AUC':>8s}")
for name, cols in sets.items():
    accs, aucs = [], []
    for rep in range(a.repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        pred = np.zeros(len(y)); prob = np.zeros(len(y))
        for tr, te in cv.split(F, y, groups):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               l2_regularization=1.0, random_state=rep)
            m.fit(F.iloc[tr][cols], y[tr])
            prob[te] = m.predict_proba(F.iloc[te][cols])[:, 1]
        accs.append(((prob > 0.5) == y).mean()); aucs.append(roc_auc_score(y, prob))
    print(f"{name:26s} {100*np.mean(accs):7.2f}% ± {100*np.std(accs):4.2f}  {np.mean(aucs):8.3f}")

print("\nReading it: compare A with D. Reference: CNN with aux = 70.1% ± 1.0%.")
print("D >= A + 2 points  -> the extra columns carry real signal; go to Test 2 (feed them to the CNN aux branch).")
print("D within 1 point of A -> they add little; stop here and wrap up.")

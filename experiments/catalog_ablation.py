"""
Test 2: is the catalog signal real?  (after catalog_baseline.py)

  1. Label rate vs missingness (is "missing stellar radius" itself informative?)
  2. Drop-one-feature ablation on set D
  3. Single-feature accuracy
  4. D with median imputation, no missingness information at all
  5. D on complete rows only
  6. Label rate by Tmag quartile (does brightness track the label?)

Run:  uv run python3 catalog_ablation.py processed_labels_v2.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
P, depth, dur = (df[c].astype(float) for c in (a.period_col, a.depth_col, a.duration_col))
Rs, Teff, Tmag = (df[c].astype(float) for c in (a.radius_col, a.teff_col, a.tmag_col))
keep = (P > 0) & (depth > 0) & (dur > 0) & df[a.label_col].isin([0, 1])
df, P, depth, dur, Rs, Teff, Tmag = [x[keep] for x in (df, P, depth, dur, Rs, Teff, Tmag)]
y = df[a.label_col].astype(int).values
groups = df[a.tic_col].values

Rs_ok = Rs.where(Rs > 0)
F = pd.DataFrame({"log_period": np.log10(P), "log_depth": np.log10(depth), "log_duration": np.log10(dur),
                  "teff": Teff, "log_rs": np.log10(Rs_ok), "tmag": Tmag})
rp = np.sqrt(depth / 1e6) * Rs_ok * 9.73
ratio = dur / (13.0 * (P / 365.25) ** (1 / 3) * Rs_ok ** (2 / 3))
F["log_rp"] = np.log10(rp.where(rp > 0))
F["log_dur_ratio"] = np.log10(ratio.where(ratio > 0))
D = list(F.columns)


def cv_acc(X, yy, gg):
    accs = []
    for rep in range(a.repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        prob = np.zeros(len(yy))
        for tr, te in cv.split(X, yy, gg):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               l2_regularization=1.0, random_state=rep)
            m.fit(X.iloc[tr], yy[tr])
            prob[te] = m.predict_proba(X.iloc[te])[:, 1]
        accs.append(((prob > 0.5) == yy).mean())
    return 100 * np.mean(accs), 100 * np.std(accs)


def show(name, X, yy=y, gg=groups):
    m, s = cv_acc(X, yy, gg)
    print(f"  {name:34s} {m:6.2f}% ± {s:4.2f}")
    return m


print("1. LABEL RATE vs MISSINGNESS (planet fraction)")
for c in ("log_rs", "teff", "tmag"):
    miss = F[c].isna().values
    if miss.sum():
        print(f"  {c:8s} missing n={miss.sum():4d}  planet rate {y[miss].mean():.3f}   present n={(~miss).sum():4d}  planet rate {y[~miss].mean():.3f}")

print("\n2. DROP-ONE-FEATURE (full set D)")
full = show("all 8 features", F[D])
for c in D:
    m = show(f"without {c}", F[[x for x in D if x != c]])
    print(f"      change: {m - full:+.2f} points")

print("\n3. SINGLE FEATURE ALONE")
for c in D:
    show(c, F[[c]])

print("\n4. SET D, MEDIAN-IMPUTED (no missingness information)")
Fi = F.fillna(F.median())
show("D, imputed", Fi[D])

print("\n5. SET D, COMPLETE ROWS ONLY")
ok = F.notna().all(axis=1).values
print(f"  {ok.sum()} complete rows")
show("D, complete rows", F.loc[ok, D], y[ok], groups[ok])

print("\n6. PLANET RATE BY Tmag QUARTILE")
q = pd.qcut(F["tmag"], 4)
print(pd.Series(y, index=F.index).groupby(q, observed=True).agg(["mean", "size"]).round(3).to_string())

print("\nReading it:")
print(" - Section 4 / 5 close to full-set accuracy -> missingness is not what drives the gain.")
print(" - One feature whose removal costs most of the gain (section 2) -> check what it really measures.")
print(" - Strong Tmag trend in section 6 plus large Tmag cost in section 2 -> possible labeller-visibility effect, not physics.")

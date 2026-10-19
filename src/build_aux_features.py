
#Test 3a: build extra scalar inputs for the CNN aux branch and write processed_labels_v3.csv
#Run:  uv run python3 build_aux_features.py processed_labels_v2.csv --out processed_labels_v3.csv

import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--out", default="processed_labels_v3.csv")
ap.add_argument("--period-col", default="Period (days)")
ap.add_argument("--depth-col", default="Depth_ppm")
ap.add_argument("--duration-col", default="Duration_hrs")
ap.add_argument("--teff-col", default="Teff_K")
ap.add_argument("--radius-col", default="Stellar_Radius")
ap.add_argument("--tmag-col", default="TESS_Mag")
a = ap.parse_args()

df = pd.read_csv(a.labels)
P, depth, dur = (df[c].astype(float) for c in (a.period_col, a.depth_col, a.duration_col))
Rs, Teff, Tmag = (df[c].astype(float) for c in (a.radius_col, a.teff_col, a.tmag_col))
good = (P > 0) & (depth > 0) & (dur > 0)

Rs_ok = Rs.where(Rs > 0)
rp = np.sqrt(depth.where(depth > 0) / 1e6) * Rs_ok * 9.73
ratio = dur.where(dur > 0) / (13.0 * (P.where(P > 0) / 365.25) ** (1 / 3) * Rs_ok ** (2 / 3))
raw = {
    "aux_log_period": np.log10(P.where(P > 0)),
    "aux_log_depth": np.log10(depth.where(depth > 0)),
    "aux_log_duration": np.log10(dur.where(dur > 0)),
    "aux_teff": Teff,
    "aux_log_rs": np.log10(Rs_ok),
    "aux_log_rp": np.log10(rp.where(rp > 0)),
    "aux_log_dur_ratio": np.log10(ratio.where(ratio > 0)),
    "aux_tmag": Tmag,
}
for name, s in raw.items():
    s = s.where(good)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    s = s.fillna(s.median())
    z = (s - s.mean()) / s.std()
    df[name] = z.where(good)
df["aux_rs_missing"] = Rs_ok.isna().astype(float).where(good)

df.to_csv(a.out, index=False)
cols = [c for c in df.columns if c.startswith("aux_")]
print(f"wrote {a.out}: {len(df)} rows, {int(good.sum())} usable")
print(df.loc[good, cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

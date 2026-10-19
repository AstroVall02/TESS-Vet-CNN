"""
Builds everything in results/: metrics tables, figures and a predictions CSV, and (optionally) refreshes the
results block in README.md.

Inputs: the test-set probabilities saved by run_protocol.py (cnn_probs_<tag>_seed<s>.npy), one file per seed.
Place this file in experiments/ and run it from the repo root:

  uv run python experiments/make_results.py --csv data/processed_labels_v3.csv --data-dir <npy light-curve folder> \
      --probs-dir <folder holding the cnn_probs_*.npy files> --readme README.md

Default tags are base_val notag_val tmag_val (baseline, headline model, headline + Tmag), in that order.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_protocol import make_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--probs-dir", default=".")
ap.add_argument("--tags", nargs=3, default=["base_val", "notag_val", "tmag_val"])
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--out", default="results")
ap.add_argument("--readme", default=None)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
NAMES = ["Baseline: light curve + period, depth", "+ catalog features", "+ catalog features + Tmag"]
COL = ["#8b949e", "#2f6fed", "#e8912d"]
C_PLANET, C_FP = "#0f9d8a", "#e0643c"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})

AUX_ALL = ["aux_log_period", "aux_log_depth", "aux_log_duration", "aux_teff", "aux_log_rs", "aux_log_rp",
           "aux_log_dur_ratio", "aux_rs_missing", "aux_tmag"]
_, _, te = make_splits(a.csv, a.data_dir, AUX_ALL)
y = te["final_label"].astype(int).values
tic_file = Path(a.probs_dir) / "test_tic_ids.npy"
if tic_file.exists():
    assert (np.load(tic_file) == te["TIC_ID"].astype(int).values).all(), "test_tic_ids.npy does not match the rebuilt test split"
P = [np.stack([np.load(Path(a.probs_dir) / f"cnn_probs_{t}_seed{s}.npy").ravel() for s in a.seeds]) for t in a.tags]
for p in P:
    assert p.shape[1] == len(y), f"{p.shape[1]} probabilities but {len(y)} test rows"
n, S = len(y), len(a.seeds)
print(f"test rows: {n}, seeds: {S}, planets: {y.mean():.3f}")


def seed_metrics(p):
    pred = p > 0.5
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    return dict(acc=100 * (pred == y).mean(), auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                p_pl=100 * tp / max(tp + fp, 1), r_pl=100 * tp / max(tp + fn, 1),
                p_fp=100 * tn / max(tn + fn, 1), r_fp=100 * tn / max(tn + fp, 1))


M = [pd.DataFrame([seed_metrics(p) for p in pp]) for pp in P]
ms = lambda df, c, d=2: f"{df[c].mean():.{d}f} ± {df[c].std(ddof=1):.{d}f}"
ens_acc = [100 * (((pp.mean(0)) > 0.5) == y).mean() for pp in P]
wrong = (P[1] > 0.5) != y
always_wrong, ever_wrong = 100 * wrong.all(0).mean(), 100 * wrong.any(0).mean()

rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(a.boot, n))
c0, c1, c2 = [((pp > 0.5) == y).mean(0) for pp in P]
def boot(u, v):
    d = 100 * np.array([u[i].mean() - v[i].mean() for i in idx])
    return 100 * (u.mean() - v.mean()), np.percentile(d, 2.5), np.percentile(d, 97.5)
d_cat, d_tm = boot(c1, c0), boot(c2, c1)
paired = (M[1].acc - M[0].acc)

# ---------------------------------------------------------------- tables
rows = ["| Model | Accuracy (%) | ROC-AUC | Avg. precision | Planet P / R (%) | No-planet P / R (%) |", "|---|---|---|---|---|---|"]
for nm, df in zip(NAMES, M):
    rows.append(f"| {nm} | {ms(df, 'acc')} | {ms(df, 'auc', 3)} | {ms(df, 'ap', 3)} | "
                f"{df.p_pl.mean():.1f} / {df.r_pl.mean():.1f} | {df.p_fp.mean():.1f} / {df.r_fp.mean():.1f} |")
table = "\n".join(rows)
tm_verdict = "no clear change" if d_tm[1] <= 0 <= d_tm[2] else ("improvement" if d_tm[0] > 0 else "worse")
notes = (f"Mean ± sd over {S} training seeds on one fixed test set (n = {n}, {100 * y.mean():.0f}% planets), threshold 0.5. "
         f"Catalog features vs baseline: **{paired.mean():+.2f} points** (positive on {int((paired > 0).sum())}/{S} seeds; "
         f"bootstrap 95% CI over test rows [{d_cat[1]:+.1f}, {d_cat[2]:+.1f}]). Adding Tmag: {d_tm[0]:+.2f} points "
         f"[{d_tm[1]:+.1f}, {d_tm[2]:+.1f}] ({tm_verdict}). Averaging the {S} seeds' probabilities gives "
         f"{ens_acc[0]:.1f}% / {ens_acc[1]:.1f}% / {ens_acc[2]:.1f}% for the three models. "
         f"With catalog features, {always_wrong:.1f}% of test rows are misclassified by every seed and {ever_wrong:.1f}% by at least one.")
block = table + "\n\n" + notes
(out / "metrics.md").write_text("# Results\n\n" + block + "\n")
json.dump({"n_test": n, "seeds": S, "models": {nm: {k: [float(v) for v in df[k]] for k in df} for nm, df in zip(NAMES, M)},
           "ensemble_acc": ens_acc, "always_wrong_pct": float(always_wrong), "ever_wrong_pct": float(ever_wrong),
           "catalog_minus_baseline": d_cat, "tmag_minus_catalog": d_tm}, open(out / "metrics.json", "w"), indent=1)
pred_df = te[["TIC_ID", "final_label", "Period (days)", "Depth_ppm", "TESS_Mag"]].copy()
for t, pp in zip(a.tags, P):
    for i, s in enumerate(a.seeds):
        pred_df[f"p_{t}_s{s}"] = pp[i]
    pred_df[f"p_{t}_mean"] = pp.mean(0)
pred_df.round(5).to_csv(out / "test_predictions.csv", index=False)

# ---------------------------------------------------------------- figures
# 1 accuracy by configuration
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, df in enumerate(M):
    ax.bar(i, df.acc.mean(), yerr=df.acc.std(ddof=1), color=COL[i], capsize=5, width=0.6, alpha=0.9)
    ax.scatter(np.full(S, i) + np.linspace(-0.12, 0.12, S), df.acc, color="k", s=14, zorder=3)
    ax.text(i, df.acc.mean() + df.acc.std(ddof=1) + 0.9, f"{df.acc.mean():.1f}%", ha="center", fontweight="bold")
ax.axhline(50, color="k", ls=":", lw=1); ax.text(2.45, 50.4, "chance (balanced)", ha="right", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(["Baseline\n(period, depth)", "+ catalog\nfeatures", "+ catalog\n+ Tmag"])
ax.set_ylim(45, max(df.acc.max() for df in M) + 7); ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="x", visible=False)
ax.set_title(f"Test accuracy, {S} seeds (dots = seeds, bars = mean ± sd)", fontsize=11, pad=12)
fig.savefig(out / "fig_accuracy.png"); plt.close(fig)

# 2 confusion matrix (headline model, pooled over seeds)
cm = np.zeros((2, 2))
for p in P[1]:
    pr = p > 0.5
    cm += [[((~pr) & (y == 0)).sum(), (pr & (y == 0)).sum()], [((~pr) & (y == 1)).sum(), (pr & (y == 1)).sum()]]
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{100 * cmn[i, j]:.1f}%\n(≈{cm[i, j] / S:.0f} rows)", ha="center", va="center",
                color="white" if cmn[i, j] > 0.55 else "black", fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No planet", "Planet"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["No planet", "Planet"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (+ catalog features)\nrow-normalised, averaged over seeds", fontsize=11)
fig.savefig(out / "fig_confusion_matrix.png"); plt.close(fig)

# 3 ROC and PR curves
g = np.linspace(0, 1, 201)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for i in (0, 1):
    tprs, precs = [], []
    for p in P[i]:
        f, t, _ = roc_curve(y, p); tprs.append(np.interp(g, f, t))
        pr, rc, _ = precision_recall_curve(y, p); precs.append(np.interp(g, rc[::-1], pr[::-1]))
    for ax, arr, lab in ((axs[0], tprs, f"{NAMES[i]} (AUC {M[i].auc.mean():.3f})"), (axs[1], precs, f"{NAMES[i]} (AP {M[i].ap.mean():.3f})")):
        m, s = np.mean(arr, 0), np.std(arr, 0, ddof=1)
        ax.plot(g, m, color=COL[i], lw=2, label=lab); ax.fill_between(g, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=COL[i], alpha=0.18)
axs[0].plot([0, 1], [0, 1], "k:", lw=1); axs[1].axhline(y.mean(), color="k", ls=":", lw=1)
axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
axs[1].set(xlabel="Recall (planet)", ylabel="Precision (planet)", title="Precision-recall curve", ylim=(0.4, 1.02))
for ax in axs: ax.legend(loc="lower right" if ax is axs[0] else "lower left", fontsize=9)
fig.savefig(out / "fig_curves.png"); plt.close(fig)

# 4 probability distribution
fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(0, 1, 26)
for lab, c, nm in ((0, C_FP, "True: no planet"), (1, C_PLANET, "True: planet")):
    ax.hist(P[1][:, y == lab].ravel(), bins=bins, density=True, alpha=0.6, color=c, label=nm)
ax.axvline(0.5, color="k", ls="--", lw=1); ax.set_xlabel("Predicted planet probability"); ax.set_ylabel("Density")
ax.set_title("Predicted probabilities (+ catalog features, all seeds pooled)", fontsize=11); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
fig.savefig(out / "fig_prob_distribution.png"); plt.close(fig)

# 5 calibration
fig, ax = plt.subplots(figsize=(4.8, 4.4))
eb = np.linspace(0, 1, 11)
for i in (0, 1):
    pp, yy = P[i].ravel(), np.tile(y, S); b = np.clip(np.digitize(pp, eb) - 1, 0, 9)
    xs = [pp[b == k].mean() for k in range(10) if (b == k).sum() > 20]; ys = [yy[b == k].mean() for k in range(10) if (b == k).sum() > 20]
    ax.plot(xs, ys, "o-", color=COL[i], label=NAMES[i])
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set(xlabel="Mean predicted probability", ylabel="Observed planet fraction", title="Calibration")
ax.legend(fontsize=8, loc="upper left"); fig.savefig(out / "fig_calibration.png"); plt.close(fig)

# 6 accuracy by depth / period tercile
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, ttl in ((axs[0], "Depth_ppm", "transit depth"), (axs[1], "Period (days)", "orbital period")):
    q = pd.qcut(te[col], 3, labels=["lowest third", "middle third", "highest third"]).values
    for i in (0, 1):
        accs = [100 * (((P[i][:, q == lv]) > 0.5) == y[q == lv]).mean() for lv in ["lowest third", "middle third", "highest third"]]
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, accs, 0.38, color=COL[i], label=NAMES[i])
    ax.set_xticks(range(3)); ax.set_xticklabels(["lowest\nthird", "middle\nthird", "highest\nthird"])
    ax.set_ylim(40, 95); ax.set_title(f"Accuracy by {ttl}", fontsize=11); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="x", visible=False)
axs[0].legend(fontsize=8, loc="upper center", ncol=1, framealpha=0.9)
se = 100 * np.sqrt(0.25 / (n / 3))
fig.text(0.5, -0.07, f"≈{n // 3} test rows per bin; one standard error of each bar is about ±{se:.0f} points", ha="center", fontsize=9)
fig.savefig(out / "fig_accuracy_by_bin.png"); plt.close(fig)

print(table); print(); print(notes)
if a.readme:
    rp = Path(a.readme); txt = rp.read_text()
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
    if pat.search(txt):
        rp.write_text(pat.sub(lambda m: m.group(1) + "\n" + block + "\n" + m.group(2), txt)); print(f"\nupdated results block in {rp}")
    else:
        print("\nREADME has no <!-- RESULTS:START --> / <!-- RESULTS:END --> markers; paste results/metrics.md by hand")

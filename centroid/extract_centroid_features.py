"""
Per-star centroid-shift features from SPOC 2-minute light curves (MOM_CENTR1/2).

For every transit event the in-transit centroid is compared with a straight line fitted to the
local out-of-transit centroid; the per-event shifts are combined with inverse-variance weights.
The same statistic is also computed at "fake" epochs (shifted by fractions of the period) to
calibrate for correlated (red) noise.

Output, one row per (TIC, period):
  status            ok | no_data | no_events | period_too_short | bad_ephemeris | error:...
  n_sectors_used, n_events, n_in
  shift_col, shift_row, err_col, err_row      centroid shift in pixels (in-transit minus local baseline)
  cen_z             sqrt((shift_col/err_col)^2 + (shift_row/err_row)^2)       raw significance
  null_rms, n_null  rms z of the fake-epoch shifts (per axis) and how many fake epochs were usable
  cen_z_cal         cen_z / max(null_rms, 1)                                  red-noise-calibrated
  cen_shift_mpx     |shift| in milli-pixels
  cen_offset_px     |shift| / depth  = implied distance of the dimming source from the flux centroid
  cen_offset_err_px approximate 1-sigma error on that

Usage:
  python extract_centroid_features.py processed_labels_v2.csv --out centroid_features.csv \
         --coverage spoc_coverage.csv --label-col final_label
  python extract_centroid_features.py processed_labels_v2.csv --limit 20          # quick trial
  python extract_centroid_features.py processed_labels_v2.csv --tics 415741431,45655786

Resumable: finished stars are skipped on re-run, failed ones are retried.
Pilot first:  --sample 400   (random rows; a later full run continues from them)
Each star's downloads go to a temporary folder that is deleted afterwards.
"""
import argparse
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

NULL_FRACTIONS = (0.25, 0.4, 0.6, 0.75)


# core maths
def event_shifts(t, c1, c2, tcs, dur, in_frac=0.4, oot_lo=0.75, oot_hi=3.0,
                 min_in=5, min_oot=15, min_side=5):
    """Per-event centroid shift (both axes) relative to a local linear baseline."""
    out = []
    for tc in tcs:
        dt = t - tc
        a = np.abs(dt)
        win = a <= oot_hi * dur
        if win.sum() < min_in + min_oot:
            continue
        inn = a <= in_frac * dur
        oot = win & (a >= oot_lo * dur)
        n_in, n_oot = int(inn.sum()), int(oot.sum())
        if n_in < min_in or n_oot < min_oot:
            continue
        if (dt[oot] < 0).sum() < min_side or (dt[oot] > 0).sum() < min_side:
            continue                                    # need baseline on both sides
        res = []
        for c in (c1, c2):
            p = np.polyfit(dt[oot], c[oot], 1)
            resid = c[oot] - np.polyval(p, dt[oot])
            sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            if not np.isfinite(sig) or sig <= 0:
                break
            res += [float(np.mean(c[inn] - np.polyval(p, dt[inn]))),
                    float(sig ** 2 * (1.0 / n_in + 1.0 / n_oot))]
        else:
            out.append(dict(n_in=n_in, d1=res[0], v1=res[1], d2=res[2], v2=res[3]))
    return out


def combine(events):
    """Inverse-variance combination of per-event shifts."""
    if not events:
        return None
    w1 = 1.0 / np.array([e["v1"] for e in events])
    w2 = 1.0 / np.array([e["v2"] for e in events])
    d1 = np.array([e["d1"] for e in events])
    d2 = np.array([e["d2"] for e in events])
    return dict(shift_col=float((w1 * d1).sum() / w1.sum()), err_col=float(w1.sum() ** -0.5),
                shift_row=float((w2 * d2).sum() / w2.sum()), err_row=float(w2.sum() ** -0.5),
                n_events=len(events), n_in=int(sum(e["n_in"] for e in events)))


def _z(c):
    return float(np.hypot(c["shift_col"] / c["err_col"], c["shift_row"] / c["err_row"]))


# light-curve access
def _col(lc, names):
    for n in names:
        try:
            return np.asarray(lc[n].value, dtype=float)
        except Exception:
            continue
    raise KeyError(names)


def process_star(lk, tic, period, epoch, dur_hr, depth_ppm, max_sectors=4, workdir=None):
    out = dict(tic=int(tic), period=float(period), status="")
    dur = float(dur_hr) / 24.0
    t0 = float(epoch) - 2457000.0 if float(epoch) > 2.4e6 else float(epoch)
    if not (period > 0 and dur > 0 and np.isfinite(t0)):
        out["status"] = "bad_ephemeris"
        return out
    if period < 7 * dur:
        out["status"] = "period_too_short"          # windows would overlap neighbouring transits
        return out

    res = lk.search_lightcurve(f"TIC {int(tic)}", author="SPOC", exptime=120)
    if len(res) == 0:
        out["status"] = "no_data"
        return out
    res = res[:max_sectors]

    fracs = [f for f in NULL_FRACTIONS if min(f, 1 - f) * period > 7 * dur]
    real, null = [], {f: [] for f in fracs}
    tmp = tempfile.mkdtemp(dir=workdir)
    n_used = 0
    try:
        coll = res.download_all(download_dir=tmp, quality_bitmask="default")
        for lc in (list(coll) if coll is not None else []):
            t = _col(lc, ["time"])
            c1 = _col(lc, ["centroid_col", "mom_centr1"])
            c2 = _col(lc, ["centroid_row", "mom_centr2"])
            ok = np.isfinite(t) & np.isfinite(c1) & np.isfinite(c2)
            t, c1, c2 = t[ok], c1[ok], c2[ok]
            if len(t) < 200:
                continue
            n_used += 1
            n = np.arange(np.floor((t.min() - t0) / period) - 1, np.ceil((t.max() - t0) / period) + 2)
            tcs = t0 + period * n
            real += event_shifts(t, c1, c2, tcs, dur)
            for f in fracs:
                null[f] += event_shifts(t, c1, c2, tcs + f * period, dur)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out["n_sectors_used"] = n_used
    comb = combine(real)
    if comb is None:
        out["status"] = "no_events" if n_used else "no_data"
        return out

    zs = []
    for f in fracs:
        c = combine(null[f])
        if c is not None:
            zs += [c["shift_col"] / c["err_col"], c["shift_row"] / c["err_row"]]
    null_rms = float(np.sqrt(np.mean(np.square(zs)))) if zs else np.nan

    shift = float(np.hypot(comb["shift_col"], comb["shift_row"]))
    depth = float(depth_ppm) / 1e6 if depth_ppm and np.isfinite(depth_ppm) else np.nan
    z = _z(comb)
    out.update(status="ok", n_events=comb["n_events"], n_in=comb["n_in"],
               shift_col=comb["shift_col"], shift_row=comb["shift_row"],
               err_col=comb["err_col"], err_row=comb["err_row"],
               cen_z=z, null_rms=null_rms, n_null=len(zs) // 2,
               cen_z_cal=z / max(null_rms, 1.0) if np.isfinite(null_rms) else z,
               cen_shift_mpx=1000.0 * shift)
    if depth and depth > 0:
        out["cen_offset_px"] = shift / depth
        out["cen_offset_err_px"] = float(np.sqrt((comb["err_col"] ** 2 + comb["err_row"] ** 2) / 2) / depth)
    return out


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default="centroid_features.csv")
    ap.add_argument("--coverage", default=None, help="spoc_coverage.csv; stars with 0 sectors are skipped")
    ap.add_argument("--tic-col", default="TIC_ID")
    ap.add_argument("--period-col", default="Period (days)")
    ap.add_argument("--epoch-col", default="Epoch (BJD)")
    ap.add_argument("--duration-col", default="Duration_hrs")
    ap.add_argument("--depth-col", default="Depth_ppm")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--tics", default=None, help="comma-separated TIC list to run only these")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None, help="random sample of N rows (use for a pilot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sectors", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    import lightkurve as lk

    df = pd.read_csv(a.labels)
    need = [a.tic_col, a.period_col, a.epoch_col, a.duration_col, a.depth_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"columns not found in {a.labels}: {missing}\navailable: {list(df.columns)}")
    if a.tics:
        df = df[df[a.tic_col].astype(int).isin([int(x) for x in a.tics.split(",")])]
    df = df.dropna(subset=[a.tic_col]).copy()
    if a.sample and a.sample < len(df):
        df = df.sample(n=a.sample, random_state=a.seed)
    if a.limit:
        df = df.head(a.limit)
    df["_key"] = df[a.tic_col].astype(int).astype(str) + "_" + df[a.period_col].astype(float).map("{:.6f}".format)

    nodata = set()
    if a.coverage and os.path.exists(a.coverage):
        cov = pd.read_csv(a.coverage)
        nodata = set(cov.loc[cov["n_sectors"] == 0, "tic"].astype(int))

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        keep = prev[~prev["status"].astype(str).str.startswith("error")]
        rows = keep.to_dict("records")
        done = set(keep["tic"].astype(int).astype(str) + "_" + keep["period"].map("{:.6f}".format))
        print(f"resuming: {len(keep)} done, {len(prev) - len(keep)} errored ones will be retried")

    todo = df[~df["_key"].isin(done)]

    def work(r):
        tic = int(r[a.tic_col])
        per = float(r[a.period_col])
        if tic in nodata:
            return dict(tic=tic, period=per, status="no_data")
        err = None
        for attempt in range(2):                         # one retry for network hiccups
            try:
                return process_star(lk, tic, per, r[a.epoch_col], r[a.duration_col], r[a.depth_col],
                                    a.max_sectors, a.workdir)
            except Exception as e:
                err = e
                time.sleep(5)
        return dict(tic=tic, period=per, status=f"error: {type(err).__name__}: {err}"[:120])

    t_start = time.time()

    def checkpoint():
        pd.DataFrame(rows).to_csv(a.out, index=False)

    def progress(i, n):
        el = time.time() - t_start
        eta = (n - i) * el / i
        print(f"{i}/{n}  {el / 60:.0f} min elapsed, about {eta / 60:.0f} min left", flush=True)

    recs = [r for _, r in todo.iterrows()]
    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(work, r) for r in recs]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 10 == 0:
                    checkpoint()
                    progress(i, len(recs))
    else:
        for i, r in enumerate(recs, 1):
            rows.append(work(r))
            if i % 10 == 0:
                checkpoint()
                progress(i, len(recs))
    checkpoint()

    res = pd.DataFrame(rows)
    print("\nstatus counts:\n" + res["status"].str.split(":").str[0].value_counts().to_string())
    if a.label_col and a.label_col in df.columns:
        m = df.drop_duplicates(a.tic_col).set_index(df[a.tic_col].astype(int))[a.label_col]
        res["label"] = res["tic"].map(m)
        print("\nusable (status ok) by class:\n" + res.groupby("label")["status"].apply(lambda s: (s == "ok").mean()).to_string())


if __name__ == "__main__":
    main()
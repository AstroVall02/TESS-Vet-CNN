import argparse
import os
import time

import pandas as pd

def search_one(lk, tic, retries=2):
    for attempt in range(retries + 1):
        try:
            res = lk.search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
            sectors = sorted({int(s) for s in res.table["sequence_number"]}) if len(res) else []
            return dict(tic=tic, n_sectors=len(sectors),
                        sectors=" ".join(map(str, sectors)), error="")
        except Exception as e:                      
            err = f"{type(e).__name__}: {e}"[:120]
            time.sleep(3 * (attempt + 1))
    return dict(tic=tic, n_sectors=-1, sectors="", error=err)


"""Count how many labelled stars have SPOC 2-minute light curves.
Only these carry the MOM_CENTR1 / MOM_CENTR2 centroid columns.

Usage:
    python check_spoc_coverage.py processed_labels_v2.csv --out spoc_coverage.csv
    python check_spoc_coverage.py processed_labels_v2.csv --limit 20      
Options: --tic-col TIC_ID   --label-col <class column>

"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default="spoc_coverage.csv")
    ap.add_argument("--tic-col", default="TIC_ID")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import lightkurve as lk

    labels = pd.read_csv(a.labels)
    tics = pd.unique(labels[a.tic_col].dropna().astype(int))
    if a.limit:
        tics = tics[: a.limit]

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        prev_ok = prev[prev["n_sectors"] >= 0]          
        rows = prev_ok.to_dict("records")
        done = set(prev_ok["tic"].astype(int))
        print(f"resuming: {len(done)} done, {len(prev) - len(prev_ok)} failed")

    todo = [t for t in tics if int(t) not in done]
    for i, tic in enumerate(todo, 1):
        rows.append(search_one(lk, int(tic)))
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{i}/{len(todo)} searched", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)

    cov = pd.DataFrame(rows)
    ok = cov[cov["n_sectors"] >= 0]
    print(f"\nsearched {len(cov)} stars, {len(cov) - len(ok)} failed")
    print(f"with >=1 SPOC 2-min sector: {(ok['n_sectors'] > 0).sum()} / {len(ok)} "
          f"({100 * (ok['n_sectors'] > 0).mean():.0f}%)")
    if a.label_col:
        m = labels.drop_duplicates(a.tic_col).set_index(a.tic_col)[a.label_col]
        ok = ok.assign(label=ok["tic"].map(m), has=ok["n_sectors"] > 0)
        print(ok.groupby("label")["has"].agg(["mean", "sum", "count"]))


if __name__ == "__main__":
    main()

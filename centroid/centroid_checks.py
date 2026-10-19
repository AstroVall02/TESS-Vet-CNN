import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry


def get_tpf(tic, sector, size=11, tries=4, wait=5):
    #TESScut download with retries (MAST drops connections now and then).
    import time
    for k in range(tries):
        try:
            return lk.search_tesscut(f"TIC {tic}", sector=sector).download(cutout_size=size)
        except Exception as e:
            print(f"  download attempt {k+1} failed: {type(e).__name__}")
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"TESScut download failed for TIC {tic} S{sector}")


def tic_position(tic):
    #RA, Dec (deg) and Tmag from the TIC.
    r = Catalogs.query_criteria(catalog="Tic", ID=int(tic))
    return float(r["ra"][0]), float(r["dec"][0]), float(r["Tmag"][0])


def locate(tic, sector, ra=None, dec=None):
    # Camera/CCD and CCD pixel position of the target in one sector.
    # The sector index is computed fresh from outSec (avoids the old sectorList-vs-outSec index mix-up).
    if ra is None:
        ra, dec, _ = tic_position(tic)
    out = tess_stars2px_function_entry(int(tic), ra, dec, aberrate=True)
    sec, cam, ccd, col, row = out[3], out[4], out[5], out[6], out[7]
    idx = list(sec).index(sector)
    return dict(tic=int(tic), sector=int(sec[idx]), camera=int(cam[idx]),
                ccd=int(ccd[idx]), col=float(col[idx]), row=float(row[idx]))


# CHECK 1: sign of dCol vs CCD
def ccd_table(measurements):
    """measurements: list of dicts {tic, sector, dCol, dRow}. Returns a DataFrame with camera/ccd addeded."""
    rows = []
    for m in measurements:
        loc = locate(m["tic"], m["sector"])
        rows.append({**loc, "dCol": m["dCol"], "dRow": m["dRow"]})
    df = pd.DataFrame(rows)
    df["rot180_group"] = np.where(df["ccd"].isin([1, 2]), "ccd 1/2", "ccd 3/4")
    return df


#CHECK 2: tess_stars2px vs cutout WCS
def wcs_crosscheck(tic, sector, size=11):
    ra, dec, _ = tic_position(tic)
    loc = locate(tic, sector, ra, dec)
    tpf = get_tpf(tic, sector, size)
    x_a = loc["col"] - tpf.column
    y_a = loc["row"] - tpf.row
    x_b, y_b = tpf.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    print(f"TIC {tic} S{sector} cam{loc['camera']} ccd{loc['ccd']}")
    print(f"  cutout origin (col,row): {tpf.column}, {tpf.row}")
    print(f"  (a) stars2px - origin : x={x_a:.3f}  y={y_a:.3f}")
    print(f"  (b) WCS               : x={float(x_b):.3f}  y={float(y_b):.3f}")
    print(f"  (a)-(b)               : dx={x_a - float(x_b):+.3f}  dy={y_a - float(y_b):+.3f}")
    return tpf, loc


# CHECK 3: fit gate
def gate(prf_fit_quality, thresh=0.5):
    """True if the fit is usable. Fits below ~0.5 = failed"""
    return prf_fit_quality >= thresh


# CHECK 4: ephemeris units + in-transit frames
def in_transit_count(time_btjd, period, epoch, dur_hr):
    #Number of frames inside transit. epoch may be BJD or BTJD.
    t0 = epoch - 2457000.0 if epoch > 2.4e6 else epoch
    phase = ((np.asarray(time_btjd) - t0 + 0.5 * period) % period) - 0.5 * period
    return int(np.sum(np.abs(phase) < dur_hr / 48.0))


# CHECK 5: direction of offset vs neighbour
def direction_check(tic, sector, d_col, d_row, size=11, radius_arcsec = 70, max_dmag = 5.0):
    ra, dec, tmag = tic_position(tic)
    loc = locate(tic, sector, ra, dec)
    tpf = get_tpf(tic, sector, size)
    target = SkyCoord(ra * u.deg, dec * u.deg)

    fit_x = loc["col"] + d_col - tpf.column
    fit_y = loc["row"] + d_row - tpf.row
    fit_sky = tpf.wcs.pixel_to_world(fit_x, fit_y)
    sep = target.separation(fit_sky).arcsec
    pa = target.position_angle(fit_sky).deg
    print(f"Fitted centroid: {sep:.1f} arcsec from target, PA {pa:.0f} deg (E of N)")

    nb = Catalogs.query_region(target, radius=radius_arcsec * u.arcsec, catalog = "Tic")
    nb = nb[(nb["ID"] != str(int(tic))) & (nb["Tmag"] < tmag + max_dmag)]
    rows = []
    for r in nb:
        c = SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg)
        rows.append(dict(TIC=r["ID"], 
                         Tmag=float(r["Tmag"]),
                         sep_arcsec=target.separation(c).arcsec,
                         PA_deg=target.position_angle(c).deg))
    df = pd.DataFrame(rows).sort_values("sep_arcsec")
    print(df.to_string(index=False))
    return df


#CHECK 6: per-CCD baseline fallback
def ccd_baseline(df_clean):
    return df_clean.groupby(["camera", "ccd"])[["dCol", "dRow"]].median()


def apply_baseline(df, baseline):
    #Subtract the per-CCD baseline; adds corrected dCol/dRow/px columns
    df = df.copy()
    b = baseline.reindex(pd.MultiIndex.from_frame(df[["camera", "ccd"]])).values
    df["dCol_c"] = df["dCol"].values - b[:, 0]
    df["dRow_c"] = df["dRow"].values - b[:, 1]
    df["px_c"] = np.hypot(df["dCol_c"], df["dRow_c"])
    df["arcsec_c"] = df["px_c"] * 21.0
    return df


# CHECK 7: tool-free difference image (independent of tdi)
def diff_image_check(tic, sector, period, epoch, dur_hr, size=11, win=2):
    """Build an out-of-transit minus in-transit image straight from the
    TESScut pixels and locate its centre of light with the WCS.
    No transit-diffImage, no PRF fit.  Positive diff = pixels that dim."""
    ra, dec, _ = tic_position(tic)
    tpf = get_tpf(tic, sector, size)
    t = tpf.time.value
    flux = np.asarray(tpf.flux.value, dtype=float)          # (n, ny, nx)
    ok = np.isfinite(t) & np.all(np.isfinite(flux), axis=(1, 2))
    t, flux = t[ok], flux[ok]

    t0 = epoch - 2457000.0 if epoch > 2.4e6 else epoch
    dur = dur_hr / 24.0
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    intr = np.abs(ph) < 0.35 * dur
    oot = (np.abs(ph) > 0.75 * dur) & (np.abs(ph) < 2.5 * dur)
    n_in, n_out = int(intr.sum()), int(oot.sum())
    if n_in < 10 or n_out < 10:
        print(f"TIC {tic}: too few frames (in={n_in}, oot={n_out})")
        return None

    diff = flux[oot].mean(0) - flux[intr].mean(0)
    err = np.sqrt(flux[oot].var(0, ddof=1) / n_out + flux[intr].var(0, ddof=1) / n_in)
    snr = diff / err
    py, px = np.unravel_index(np.argmax(diff), diff.shape)

    y0, y1 = max(py - win, 0), min(py + win + 1, diff.shape[0])
    x0, x1 = max(px - win, 0), min(px + win + 1, diff.shape[1])
    w = np.clip(diff[y0:y1, x0:x1], 0, None)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()

    tx, ty = tpf.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    dx, dy = cx - float(tx), cy - float(ty)
    print(f"TIC {tic} S{sector}: in={n_in} oot={n_out} frames, peak diff SNR={snr[py, px]:.1f}")
    print(f"  target (WCS) x={float(tx):.2f} y={float(ty):.2f} | diff-image centre x={cx:.2f} y={cy:.2f}")
    print(f"  offset dx={dx:+.2f} dy={dy:+.2f} px = {np.hypot(dx, dy):.2f} px "
          f"({np.hypot(dx, dy) * 21:.0f} arcsec)")
    return dict(dx=dx, dy=dy, px=float(np.hypot(dx, dy)), peak_snr=float(snr[py, px]),
                n_in=n_in, n_oot=n_out)


# single source of truth for the offset calculation
def offset_from_fit(tic, sector, fit_vector, prf_quality=None, min_quality=0.5):
    """Offset of a transit-diffImage PRF fit from the aberrated target position.
    fit_vector: the tool's fitVector (CCD column, CCD row, ...).
    Use this everywhere instead of re-deriving the reference position."""
    loc = locate(tic, sector)
    d_col = float(fit_vector[0]) - loc["col"]
    d_row = float(fit_vector[1]) - loc["row"]
    px = float(np.hypot(d_col, d_row))
    return dict(tic=int(tic), sector=int(sector), camera=loc["camera"], ccd=loc["ccd"],
                ref_col=loc["col"], ref_row=loc["row"],
                dCol=d_col, dRow=d_row, px=px, arcsec=px * 21.0,
                prf_quality=prf_quality,
                usable=(prf_quality is None) or (prf_quality >= min_quality))


# preferred: use the reference position the tool itself uses
def offset_from_catalog(fit_vector, catalog_data, pixel_data=None, tic=None,
                        prf_quality=None, min_quality=0.5):
    """Offset of a PRF fit from the target position that tessDiffImage draws
    as the magenta star: targetColPix - dCol, targetRowPix - dRow.
    That reference already includes aberration, proper-motion correction and
    the shift between the tess_stars2px frame and the cutout WCS frame.
    catalog_data / pixel_data come from the star's imageData pickle."""
    ref_col = float(np.atleast_1d(catalog_data["targetColPix"])[0]
                    - np.atleast_1d(catalog_data["dCol"])[0])
    ref_row = float(np.atleast_1d(catalog_data["targetRowPix"])[0]
                    - np.atleast_1d(catalog_data["dRow"])[0])
    d_col = float(fit_vector[0]) - ref_col
    d_row = float(fit_vector[1]) - ref_row
    px = float(np.hypot(d_col, d_row))
    out = dict(tic=tic, ref_col=ref_col, ref_row=ref_row,
               frame_dCol=float(np.atleast_1d(catalog_data["dCol"])[0]),
               frame_dRow=float(np.atleast_1d(catalog_data["dRow"])[0]),
               dCol=d_col, dRow=d_row, px=px, arcsec=px * 21.0,
               prf_quality=prf_quality,
               usable=(prf_quality is None) or (prf_quality >= min_quality))
    if pixel_data is not None:
        out.update(sector=int(pixel_data["sector"]), camera=int(pixel_data["camera"]),
                   ccd=int(pixel_data["ccd"]))
    return out


# which catalogue star does the fitted centroid land on?
def nearest_catalog_stars(fit_vector, catalog_data, n=6):
    """Catalogue stars closest to the fitted centroid, in the tool's own
    (dCol/dRow-corrected) frame. Assumes index 0 of catalog_data is the target."""
    dcol = float(np.atleast_1d(catalog_data["dCol"])[0])
    drow = float(np.atleast_1d(catalog_data["dRow"])[0])
    cols = np.asarray(catalog_data["ticColPix"], dtype=float) - dcol
    rows = np.asarray(catalog_data["ticRowPix"], dtype=float) - drow
    tmag = np.asarray(catalog_data["ticMag"], dtype=float)
    df = pd.DataFrame({
        "TIC": np.asarray(catalog_data["ticID"]).astype(int),
        "Tmag": tmag,
        "dTmag_vs_target": tmag - tmag[0],
        "sep_from_target_arcsec": np.asarray(catalog_data["separation"], dtype=float),
        "dist_fit_to_star_px": np.hypot(cols - float(fit_vector[0]), rows - float(fit_vector[1])),
    })
    return df.sort_values("dist_fit_to_star_px").head(n).reset_index(drop=True)


# stability
def _assign_to_transits(time, frame_idx, centers):
    """For each frame index, which transit centre (index into `centers`) is closest in time."""
    d = np.abs(time[frame_idx][:, None] - time[centers][None, :])
    return d.argmin(axis=1)


def split_half_fits(fit_func, make_diff, prf, catalog_data, pixel_data,
                    in_idx, out_idx, transit_index, split="alternate",
                    min_transits=2):
    """Fit the PRF centroid separately on two subsets of the transits.

    fit_func  : transitCentroids.tess_PRF_centroid
    make_diff : tdi.make_difference_image   (reuses the tool's own logic)
    in_idx, out_idx, transit_index : the arrays stored in the star's pickle
    split     : "alternate" (1st,3rd,.. vs 2nd,4th,..) or "halves" (first vs second half in time)

    Returns (DataFrame with one row per group, separation in px between the two fits)."""
    time = np.asarray(pixel_data["time"], dtype=float)
    centers = np.asarray(transit_index, dtype=int)
    in_idx = np.asarray(in_idx, dtype=int)
    out_idx = np.asarray(out_idx, dtype=int)
    in_owner = _assign_to_transits(time, in_idx, centers)
    out_owner = _assign_to_transits(time, out_idx, centers)

    # transits that actually contributed frames (the tool drops "bad" ones)
    good = [k for k in range(len(centers))
            if np.any(in_owner == k) and np.any(out_owner == k)]
    print(f"{len(good)} usable transits")

    if split == "alternate":
        groups = {"A (1st,3rd,..)": good[0::2], "B (2nd,4th,..)": good[1::2]}
    elif split == "halves":
        h = len(good) // 2
        groups = {"first half": good[:h], "second half": good[h:]}
    else:
        raise ValueError("split must be 'alternate' or 'halves'")

    rows, fits = [], {}
    for name, ks in groups.items():
        if len(ks) < min_transits:
            print(f"{name}: only {len(ks)} transit(s) (need >= {min_transits}), skipped")
            continue
        imgs = [make_diff(pixel_data, in_idx[in_owner == k], out_idx[out_owner == k])["diffImage"]
                for k in ks]
        img = np.nanmean(imgs, axis=0)
        out = fit_func(prf, catalog_data["extent"], img, catalog_data)
        off = offset_from_catalog(out[0], catalog_data, prf_quality=out[1])
        fits[name] = np.asarray(out[0][:2], dtype=float)
        rows.append(dict(group=name, n_transits=len(ks),
                         fit_col=float(out[0][0]), fit_row=float(out[0][1]),
                         dCol=off["dCol"], dRow=off["dRow"], px=off["px"],
                         arcsec=off["arcsec"], prf_quality=off["prf_quality"]))
    df = pd.DataFrame(rows)

    sep = None
    if len(fits) == 2:
        a, b = list(fits.values())
        sep = float(np.hypot(*(a - b)))
        verdict = ("stable" if sep < 0.5 else "marginal" if sep < 1.0 else "unstable")
        print(f"separation between the two fits: {sep:.2f} px ({sep * 21:.0f} arcsec) -> {verdict}")
        print("(rough guide only: each half has ~1/sqrt(2) of the full SNR, so some extra scatter is expected)")
    return df, sep


def control_epoch_fits(tdi, fit_func, prf, catalog_data, pixel_data, planet_data,
                       fractions=(0.25, 0.4, 0.6, 0.75), return_images=False):
    """Control: repeat the whole difference-image + PRF fit with the epoch shifted
    by a fraction of the period, so no real transit is in the 'in-transit' frames.
    Shows where the fit lands on pure noise for THIS star.
    Compare the real offset with these. Only meaningful if period >> duration.
    NOTE: if a bright or variable neighbour dominates the images, the control fits
    can reproduce the real fit (the signal is then NOT specific to the transit epoch).
    return_images=True also returns the list of control difference images."""
    rows, imgs = [], []
    for f in fractions:
        ctrl = dict(planet_data)
        ctrl["epoch"] = planet_data["epoch"] + f * planet_data["period"]
        in_i, out_i, tr_i, dimg = tdi.find_transits(pixel_data, ctrl)
        if (not dimg) or ("diffImage" not in dimg) or (not np.any(dimg["diffImage"])):
            print(f"shift {f}: no usable control transits")
            continue
        out = fit_func(prf, catalog_data["extent"], dimg["diffImage"], catalog_data)
        off = offset_from_catalog(out[0], catalog_data, prf_quality=out[1])
        rows.append(dict(shift_frac=f, n_transits=len(tr_i), dCol=off["dCol"],
                         dRow=off["dRow"], px=off["px"], prf_quality=off["prf_quality"]))
        imgs.append(np.asarray(dimg["diffImage"], dtype=float))
    df = pd.DataFrame(rows)
    return (df, imgs) if return_images else df


# ------------------------------- choosing validation targets from a table
def host_geometry(df, target_col="target_tic", host_col="host_tic",
                  depth_ppm_col=None, period_col=None):
    """Add geometry columns for (target, true host) pairs, from the TIC.

    df needs a target TIC column and a true-host TIC column (ints).
    Adds: target_Tmag, host_Tmag, dTmag (host - target), sep_arcsec, sep_px,
          pa_deg (position angle of host from target, E of N).
    If depth_ppm_col is given: max_dmag = 2.5*log10(1/depth) is the faintest host
          that could physically produce the dip (even with a total eclipse), and
          host_bright_enough = dTmag <= max_dmag.
    If period_col is given: transits_per_sector_est = 25 / period (rough)."""
    cache = {}

    def pos(tic):
        tic = int(tic)
        if tic not in cache:
            try:
                cache[tic] = tic_position(tic)
            except Exception:
                cache[tic] = (np.nan, np.nan, np.nan)
        return cache[tic]

    rows = []
    for _, r in df.iterrows():
        t_ra, t_dec, t_mag = pos(r[target_col])
        h_ra, h_dec, h_mag = pos(r[host_col])
        if np.all(np.isfinite([t_ra, t_dec, h_ra, h_dec])):
            t = SkyCoord(t_ra * u.deg, t_dec * u.deg)
            h = SkyCoord(h_ra * u.deg, h_dec * u.deg)
            sep, pa = t.separation(h).arcsec, t.position_angle(h).deg
        else:
            sep = pa = np.nan
        rows.append((t_mag, h_mag, h_mag - t_mag, sep, sep / 21.0, pa))
    geo = pd.DataFrame(rows, index=df.index,
                       columns=["target_Tmag", "host_Tmag", "dTmag", "sep_arcsec", "sep_px", "pa_deg"])
    out = pd.concat([df, geo], axis=1)
    if depth_ppm_col is not None:
        frac = out[depth_ppm_col].astype(float) / 1e6
        out["max_dmag"] = 2.5 * np.log10(1.0 / frac)
        out["host_bright_enough"] = out["dTmag"] <= out["max_dmag"]
    if period_col is not None:
        out["transits_per_sector_est"] = 25.0 / out[period_col].astype(float)
    return out


# ------------------------------------------- one label per star, fixed rules
def classify_measurement(full, halves_sep, ctrl_df, min_quality=0.5, min_offset_px=0.7,
                         max_halves_sep=0.5, max_offset_px=8.0, match_tol_px=0.5):
    """Apply the pre-registered rules and return (label, reason).

    full       : dict from offset_from_catalog (needs dCol, dRow, px, prf_quality)
    halves_sep : separation in px from split_half_fits (None if it could not run)
    ctrl_df    : DataFrame from control_epoch_fits (or None)

    Labels (prefixed "unverified_" when halves_sep is None, i.e. the split could not be run):
      no_measurement            - low fit quality, fit pinned near the cutout edge, or halves disagree
      on_target                 - good stable fit within min_offset_px of the target
      offset_epoch_specific     - good stable fit away from the target, no control epoch reproduces it
      offset_not_epoch_specific - good stable fit away from the target, but control epochs reproduce
                                  it (a bright/variable neighbour may be driving the image)"""
    q, px = full.get("prf_quality"), full["px"]
    if q is None or q < min_quality:
        return "no_measurement", f"full-fit quality {q} < {min_quality}"
    if px > max_offset_px:
        return "no_measurement", f"fit {px:.1f} px from target, likely pinned at the cutout edge"
    untested = halves_sep is None          # e.g. fewer than 4 usable transits
    if (not untested) and halves_sep > max_halves_sep:
        return "no_measurement", f"split-half fits disagree (sep={halves_sep:.2f} px)"
    prefix = "unverified_" if untested else ""
    note = " (split halves not testable)" if untested else ""
    if px < min_offset_px:
        return prefix + "on_target", f"offset {px:.2f} px{note}"
    n_same, n_ctrl = 0, 0
    if ctrl_df is not None and len(ctrl_df):
        n_ctrl = len(ctrl_df)
        d = np.hypot(ctrl_df["dCol"] - full["dCol"], ctrl_df["dRow"] - full["dRow"])
        n_same = int(((d < match_tol_px) & (ctrl_df["prf_quality"] >= min_quality)).sum())
    if n_same > 0:
        return prefix + "offset_not_epoch_specific", f"{n_same}/{n_ctrl} control epochs reproduce the fit{note}"
    return prefix + "offset_epoch_specific", f"offset {px:.2f} px, no control epoch reproduces it{note}"
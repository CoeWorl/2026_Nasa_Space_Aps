"""Run VALIDATION_PLAN.md Tests 2-7 on the extracted stack.

    uv run python -u scripts/validate_tests.py --stack DIR --out-dir DIR

Pre-registered choices (fixed in this file before any result was computed):
  * Pair for Tests 2 and 4: 2026-10-04 -> 2026-10-16 (Test 1's pair).
  * Land-cover proxy: backscatter level of the temporal median, in three
    terciles. This is NOT land cover; it is a brightness proxy.
  * Stable controls (Test 3): the 4 tiles with the highest temporal-median dB
    among tiles at or above 0 dB (built-up proxy), chosen by farthest-point
    sampling so they are geographically separated. Chosen from brightness only,
    never from change statistics.
  * Test 7 candidate: the non-control tile with the largest fraction of pixels
    whose temporal-median-shifted series shows a sustained step of at least
    3 robust sigma. Recorded before external evidence is looked at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from groundtruth.processing import log_ratio_db
from groundtruth.statistics import exceedance_fraction, robust_summary

TILE = 256
K = 3.0


def to_db(linear: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = valid & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok].astype(np.float64)).astype(np.float32)
    return out


def tiles(shape: tuple[int, int]) -> list[tuple[int, int, slice, slice]]:
    h, w = shape
    out = []
    for r0 in range(0, h, TILE):
        for c0 in range(0, w, TILE):
            out.append((r0, c0, slice(r0, min(r0 + TILE, h)), slice(c0, min(c0 + TILE, w))))
    return out


def block_mean_db(linear: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate linear intensity over k x k blocks with valid-pixel weights, then to dB."""
    h, w = linear.shape
    hh, ww = h // k * k, w // k * k
    lin = np.where(valid, linear, 0.0)[:hh, :ww].reshape(hh // k, k, ww // k, k)
    wgt = valid[:hh, :ww].astype(np.float64).reshape(hh // k, k, ww // k, k)
    num = (lin.astype(np.float64) * 1.0).sum(axis=(1, 3))
    den = wgt.sum(axis=(1, 3))
    ok = den >= 0.5 * k * k  # require at least half the block to be valid
    mean_lin = np.full(den.shape, np.nan)
    mean_lin[ok] = num[ok] / den[ok]
    db = np.full(mean_lin.shape, np.nan, dtype=np.float32)
    pos = ok & (mean_lin > 0)
    db[pos] = 10.0 * np.log10(mean_lin[pos]).astype(np.float32)
    return db, ok


def test2(db_a: np.ndarray, i4: int, i16: int, levels: np.ndarray) -> dict:
    change, valid = log_ratio_db(10 ** (db_a[i4] / 10), 10 ** (db_a[i16] / 10))
    vals = change[valid]
    summary = robust_summary(vals)
    skew = float(((vals - summary.median) ** 3).mean() / (summary.robust_scale ** 3 + 1e-12))

    tile_medians = []
    for _, _, rs, cs in tiles(change.shape):
        t = change[rs, cs][valid[rs, cs]]
        if t.size > 1000:
            tile_medians.append(float(np.median(t)))
    tile_medians = np.array(tile_medians)

    classes = {}
    for name, sel in {
        "low (tercile 1)": levels <= np.nanpercentile(levels, 33),
        "mid (tercile 2)": (levels > np.nanpercentile(levels, 33)) & (levels <= np.nanpercentile(levels, 66)),
        "high (tercile 3)": levels > np.nanpercentile(levels, 66),
    }.items():
        v = change[sel & valid]
        if v.size > 1000:
            s = robust_summary(v)
            classes[name] = {"median_db": s.median, "robust_scale_db": s.robust_scale,
                             "p05_db": s.p05, "p95_db": s.p95, "n": s.count}

    scale_by_k = {}
    for k in (1, 2, 4, 8):
        if k == 1:
            db_k, _ = db_a[i4], None
            db_k2 = db_a[i16]
            c = db_k2 - db_k
            vv = np.isfinite(c)
        else:
            d4, ok4 = block_mean_db(10 ** (db_a[i4] / 10), np.isfinite(db_a[i4]), k)
            d16, ok16 = block_mean_db(10 ** (db_a[i16] / 10), np.isfinite(db_a[i16]), k)
            c = d16 - d4
            vv = np.isfinite(c)
        scale_by_k[f"{10 * k} m"] = robust_summary(c[vv]).robust_scale

    return {
        "n_valid": int(vals.size),
        "median_db": summary.median,
        "mad_db": summary.mad,
        "robust_scale_db": summary.robust_scale,
        "p05_db": summary.p05,
        "p95_db": summary.p95,
        "skew_robust": skew,
        "tile_median_range_db": [float(tile_medians.min()), float(tile_medians.max())],
        "tile_median_std_db": float(tile_medians.std()),
        "by_brightness_class": classes,
        "robust_scale_by_resolution_db": scale_by_k,
        "numberOfLooks": "not extracted in this run",
    }


def tile_stats_controls(db: np.ndarray, rs: slice, cs: slice) -> np.ndarray:
    return db[:, rs, cs].reshape(db.shape[0], -1)


def choose_controls(temporal_median: np.ndarray, n: int = 4) -> list[tuple[int, int]]:
    """Pre-registered: brightest-tile candidates (>= 0 dB), farthest-point sampling."""
    cands = []
    for r0, c0, rs, cs in tiles(temporal_median.shape):
        block = temporal_median[rs, cs]
        if np.isfinite(block).sum() > 0.9 * block.size and np.nanmedian(block) >= 0.0:
            cands.append((float(np.nanmedian(block)), r0, c0))
    if not cands:
        return []
    cands.sort(reverse=True)
    chosen = [cands[0]]
    while len(chosen) < n and len(chosen) < len(cands):
        best, best_d = None, -1.0
        for cand in cands:
            if any(cand[1:] == c[1:] for c in chosen):
                continue
            d = min(((cand[1] - c[1]) ** 2 + (cand[2] - c[2]) ** 2) ** 0.5 for c in chosen)
            if d > best_d:
                best, best_d = cand, d
        if best is None:
            break
        chosen.append(best)
    return [(c[1], c[2]) for c in chosen]


def test3(db: np.ndarray, dates: np.ndarray, controls: list[tuple[int, int]]) -> dict:
    """Leave-one-out anomalies on control tiles; scale calibrated on the first half of dates."""
    n = db.shape[0]
    half = n // 2
    cal, test = list(range(half)), list(range(half, n))
    results = []
    for r0, c0 in controls:
        rs, cs = slice(r0, min(r0 + TILE, db.shape[1])), slice(c0, min(c0 + TILE, db.shape[2]))
        block = db[:, rs, cs]
        anomalies = []
        for t in range(n):
            others = np.delete(block, t, axis=0)
            base = np.nanmedian(others, axis=0)
            anomalies.append(block[t] - base)
        anomalies = np.stack(anomalies)
        cal_vals = anomalies[cal]
        cal_vals = cal_vals[np.isfinite(cal_vals)]
        s = robust_summary(cal_vals)
        test_vals = anomalies[test]
        fa = {}
        for k in (3.0, 4.0, 5.0):
            fa[f"{k:.0f}sigma"] = exceedance_fraction(test_vals, center=0.0, scale=s.robust_scale, k=k)
        results.append({"tile_row": r0, "tile_col": c0, "calibration_robust_scale_db": s.robust_scale,
                        "false_alarm_fraction_held_out": fa})
    all_fa = np.array([r["false_alarm_fraction_held_out"]["3sigma"] for r in results])
    return {
        "controls": results,
        "mean_false_alarm_3sigma": float(all_fa.mean()) if all_fa.size else None,
        "target_fraction": 0.01,
        "meets_target_3sigma": bool(all_fa.size and all_fa.mean() <= 0.01),
        "held_out_dates": [str(d) for d in dates[test]],
    }


def test4(db: np.ndarray, i4: int, i16: int) -> dict:
    a = db[i4]
    b = db[i16]
    ok = np.isfinite(a) & np.isfinite(b)
    a0 = np.where(ok, a - np.nanmean(np.where(ok, a, np.nan)), 0.0)
    b0 = np.where(ok, b - np.nanmean(np.where(ok, b, np.nan)), 0.0)
    best = (-2.0, 0, 0)
    ncc_zero = None
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            bs = np.roll(np.roll(b0, dy, axis=0), dx, axis=1)
            m = np.roll(np.roll(ok, dy, axis=0), dx, axis=1) & ok
            num = (a0 * bs)[m].sum()
            den = np.sqrt((a0[m] ** 2).sum() * (bs[m] ** 2).sum())
            ncc = float(num / den) if den > 0 else float("nan")
            if dy == 0 and dx == 0:
                ncc_zero = ncc
            if ncc > best[0]:
                best = (ncc, dy, dx)
    # Strong scatterer agreement: top 1% brightest pixels in each date
    ta = a[ok] >= np.quantile(a[ok], 0.99)
    tb = b[ok] >= np.quantile(b[ok], 0.99)
    overlap = float((ta & tb).sum() / max(ta.sum(), 1))
    return {
        "ncc_at_zero_shift": ncc_zero,
        "best_ncc": best[0],
        "best_shift_pixels_row_col": [best[1], best[2]],
        "shift_search_radius_pixels": 3,
        "top1pct_scatterer_overlap": overlap,
        "pass": bool(best[1] == 0 and best[2] == 0),
    }


def test5(db: np.ndarray, dates: np.ndarray) -> dict:
    """Leave-one-out anomaly per date, global robust scale, and per-pixel run structure."""
    n = db.shape[0]
    anom = np.empty_like(db)
    for t in range(n):
        others = np.delete(db, t, axis=0)
        anom[t] = db[t] - np.nanmedian(others, axis=0)
    flat = anom[np.isfinite(anom)]
    s = robust_summary(flat)
    flag = np.abs(anom) > K * s.robust_scale
    per_date = {str(dates[t]): float(np.nanmean(flag[t][np.isfinite(anom[t])])) for t in range(n)}

    # Per-pixel run structure over the valid pixels only.
    valid_all = np.all(np.isfinite(anom), axis=0)
    f = flag[:, valid_all]
    counts = f.sum(axis=0)
    # longest run of consecutive flagged dates
    longest = np.zeros(f.shape[1], dtype=np.int32)
    run = np.zeros(f.shape[1], dtype=np.int32)
    for t in range(n):
        run = np.where(f[t], run + 1, 0)
        longest = np.maximum(longest, run)
    total = max(f.shape[1], 1)
    categories = {
        "no_anomaly": float((counts == 0).sum() / total),
        "isolated_single_date": float((counts == 1).sum() / total),
        "sustained_run_3plus": float((longest >= 3).sum() / total),
    }
    return {
        "robust_scale_db": s.robust_scale,
        "per_date_flagged_fraction_3sigma": per_date,
        "pixel_categories": categories,
        "note": "Leave-one-out uses future dates in the baseline; this is exploratory, not a change-point test.",
    }


def test6(a_db: np.ndarray, a_valid: np.ndarray, b_db: np.ndarray, b_valid: np.ndarray,
          b_pixel_m: float, dates: np.ndarray, controls: list[tuple[int, int]]) -> dict:
    """Compare 80 m representations: A aggregated to 80 m vs native B."""
    n = a_db.shape[0]
    a80 = np.stack([block_mean_db(10 ** (a_db[t] / 10), a_valid[t], 8)[0] for t in range(n)])
    a80_valid = np.stack([np.isfinite(a80[t]) for t in range(n)])
    b_native = b_db
    bvalid = b_valid

    def coverage(valid):
        return [float(v.mean()) for v in valid]

    def control_scale(stack):
        scales = []
        for r0, c0 in controls:
            r80, c80 = r0 // 8, c0 // 8
            blk = stack[:, r80:r80 + TILE // 8, c80:c80 + TILE // 8]
            if blk.size == 0:
                continue
            anom_flat = []
            for t in range(stack.shape[0]):
                others = np.delete(blk, t, axis=0)
                anom_flat.append(blk[t] - np.nanmedian(others, axis=0))
            v = np.stack(anom_flat)
            v = v[np.isfinite(v)]
            if v.size > 20:
                scales.append(robust_summary(v).robust_scale)
        return scales

    # Broad-pattern agreement: temporal median maps on a common coarse grid (nearest).
    med_a = np.nanmedian(a80, axis=0)
    med_b_full = np.nanmedian(b_native, axis=0)
    scale_b = b_pixel_m / 80.0
    rows = np.clip((np.arange(med_a.shape[0]) * 1.0 / max(scale_b, 1e-9)).astype(int), 0, med_b_full.shape[0] - 1)
    cols = np.clip((np.arange(med_a.shape[1]) * 1.0 / max(scale_b, 1e-9)).astype(int), 0, med_b_full.shape[1] - 1)
    med_b = med_b_full[np.ix_(rows, cols)]
    ok = np.isfinite(med_a) & np.isfinite(med_b)
    corr = float(np.corrcoef(med_a[ok], med_b[ok])[0, 1]) if ok.sum() > 100 else None

    return {
        "a80_valid_fraction_per_date": dict(zip((str(d) for d in dates), coverage(a80_valid))),
        "b_valid_fraction_per_date": dict(zip((str(d) for d in dates), coverage(bvalid))),
        "control_robust_scale_db_a80": control_scale(a80),
        "control_robust_scale_db_b": control_scale(b_native),
        "broad_pattern_correlation_temporal_median": corr,
        "b_pixel_m_reported": b_pixel_m,
        "storage_bytes_a80_stack": int(a80.nbytes),
        "storage_bytes_b_stack": int(b_native.nbytes),
        "note": "Broad-pattern correlation uses nearest-neighbour resampling of B onto the A-80 m grid; B pixel size is from the file.",
    }


def test7(anom_by_pixel_sustained: np.ndarray, controls: list[tuple[int, int]]) -> dict:
    """Pre-registered candidate: tile with the largest fraction of sustained-step pixels, excluding controls."""
    best = None
    for r0, c0, rs, cs in tiles(anom_by_pixel_sustained.shape):
        if (r0, c0) in controls:
            continue
        block = anom_by_pixel_sustained[rs, cs]
        frac = float(np.nanmean(block)) if np.isfinite(block).any() else 0.0
        if best is None or frac > best["sustained_fraction"]:
            best = {"tile_row": r0, "tile_col": c0, "sustained_fraction": frac}
    return {"candidate": best, "selection_rule": "largest sustained-step pixel fraction, controls excluded"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.stack / "stack_aoi.json").read_text())
    z = np.load(args.stack / "stack_aoi.npz")
    dates = z["dates"]
    a_db = np.stack([to_db(z["A_values"][t], z["A_valid"][t]) for t in range(len(dates))])
    b_db = np.stack([to_db(z["B_values"][t], z["B_valid"][t]) for t in range(len(dates))])
    a_valid = np.isfinite(a_db)
    b_valid = np.isfinite(b_db)

    i4 = int(np.where(dates == "2026-10-04")[0][0])
    i16 = int(np.where(dates == "2026-10-16")[0][0])
    temporal_median = np.nanmedian(a_db, axis=0)
    levels = temporal_median

    controls = choose_controls(temporal_median, n=4)
    print(f"dates {len(dates)}; A {a_db.shape}; B {b_db.shape}; controls {controls}", flush=True)

    out = {"dates": [str(d) for d in dates], "pre_registered_controls": controls}
    out["test2_noise"] = test2(a_db, i4, i16, levels)
    print("test2 done", flush=True)
    out["test3_false_alarm"] = test3(a_db, dates, controls)
    print("test3 done", flush=True)
    out["test4_alignment"] = test4(a_db, i4, i16)
    print("test4 done", flush=True)
    out["test5_temporal"] = test5(a_db, dates)
    print("test5 done", flush=True)
    out["test6_scale_80m"] = test6(a_db, a_valid, b_db, b_valid, meta["B_pixel_m"], dates, controls)
    print("test6 done", flush=True)

    # Sustained-step fraction per pixel: a run of >=3 same-sign anomalies beyond 3 sigma.
    anom = np.stack([a_db[t] - np.nanmedian(np.delete(a_db, t, axis=0), axis=0) for t in range(len(dates))])
    s = robust_summary(anom[np.isfinite(anom)]).robust_scale
    pos = anom > K * s
    neg = anom < -K * s
    def sustained(mask):
        out_s = np.zeros(mask.shape[1:], dtype=bool)
        run = np.zeros(mask.shape[1:], dtype=np.int32)
        best = np.zeros(mask.shape[1:], dtype=np.int32)
        for t in range(mask.shape[0]):
            run = np.where(mask[t], run + 1, 0)
            best = np.maximum(best, run)
        return best >= 3
    step = sustained(pos) | sustained(neg)
    sust_tile = np.zeros(a_db.shape[1:], dtype=np.float32)
    sust_tile[:] = np.where(np.isfinite(temporal_median), step.astype(np.float32), np.nan)
    out["test7_candidate"] = test7(sust_tile, controls)
    print("test7 done", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "validation_results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out_dir / 'validation_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

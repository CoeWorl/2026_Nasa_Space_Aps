"""Follow-up checks on the temporal results.

    uv run python -u scripts/followup_checks.py --stack DIR --out-dir DIR

Section 1, robust slopes. OLS slopes flagged 6.4% of pixels at 3 sigma. Re-run
with the Theil-Sen median-of-pairwise-slopes estimator, a robust standard error
from the residual MAD, and with the two suspect dates (2026-01-25, 2026-02-06)
removed. Compare against the ~0.27% a Gaussian null would give at 3 sigma.

Section 2, water split. There is no land-cover product in this repo, so the split
is a brightness proxy: pixels whose temporal-median dB is below -15 dB. The
threshold is fixed here, before results, and sensitivity at -18 and -12 dB is
reported too. This is a proxy, not a water mask.

Section 3, injected-change sensitivity. Known multiplicative changes are added to
real, non-control tiles from a date index onward. A pixel is flagged when its
leave-one-out anomaly exceeds 3 sigma, with sigma calibrated on the four control
tiles from the first half of the dates (the same rule as Test 3). A patch is
detected when at least half its pixels are flagged. False-alarm rates come from
the unmodified control tiles on the second half of the dates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from groundtruth.statistics import robust_summary

TILE = 256
DAYS_PER_YEAR = 365.25
SUSPECT = {"2026-01-25", "2026-02-06"}
CONTROLS = [(1280, 256), (256, 1024), (512, 1024), (1536, 256)]  # from validate_tests.py
SIG_K = 3.0


def to_db(linear, valid):
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = valid & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok].astype(np.float64)).astype(np.float32)
    return out


def theil_sen_maps(db: np.ndarray, t: np.ndarray, rows_per_chunk: int = 64):
    """Per-pixel Theil-Sen slope, computed by chunks of rows to bound memory."""
    n, h, w = db.shape
    pairs = list(itertools.combinations(range(n), 2))
    i_idx = np.array([p[0] for p in pairs])
    j_idx = np.array([p[1] for p in pairs])
    dt = (t[j_idx] - t[i_idx]).astype(np.float32)
    slope = np.full((h, w), np.nan, dtype=np.float32)
    for r0 in range(0, h, rows_per_chunk):
        block = db[:, r0:r0 + rows_per_chunk, :]
        dy = block[j_idx] - block[i_idx]
        s = dy / dt[:, None, None]
        slope[r0:r0 + rows_per_chunk] = np.nanmedian(s, axis=0)
    return slope


def robust_se_and_t(db, t, slope):
    """Robust sigma from residual MAD; SE ~ 1.2533 * sigma / sqrt(Sxx), 1.2533 = sqrt(pi/2)."""
    n = db.shape[0]
    intercept = np.nanmedian(db - slope[None] * t[:, None, None], axis=0)
    resid = db - (intercept[None] + slope[None] * t[:, None, None])
    sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)
    tm = t.mean()
    sxx = ((t - tm) ** 2).sum()
    se = 1.2533 * sigma / np.sqrt(sxx)
    t_stat = np.where(se > 0, np.abs(slope) / se, np.nan)
    return t_stat


def ols_t(db, t):
    ok = np.isfinite(db)
    n = ok.sum(axis=0).astype(np.float64)
    tt = np.where(ok, t[:, None, None], np.nan)
    tm = np.nanmean(tt, axis=0)
    ym = np.nanmean(db, axis=0)
    sxx = np.nansum((tt - tm) ** 2, axis=0)
    sxy = np.nansum((tt - tm) * (db - ym), axis=0)
    slope = sxy / sxx
    resid = db - (ym + slope * (tt - tm))
    sigma = np.sqrt(np.nansum(resid ** 2, axis=0) / np.maximum(n - 2, 1))
    t_stat = np.abs(slope) / (sigma / np.sqrt(sxx))
    return np.where(n >= 15, t_stat, np.nan)


def section1(db, dates, t):
    out = {}
    slope_ts = theil_sen_maps(db, t)
    t_robust = robust_se_and_t(db, t, slope_ts)
    out["theil_sen_fraction_sig3"] = float(np.nanmean(t_robust > 3))

    t_ols = ols_t(db, t)
    out["ols_fraction_sig3_all_dates"] = float(np.nanmean(t_ols > 3))

    keep = np.array([str(d) not in SUSPECT for d in dates])
    t_ols_drop = ols_t(db[keep], t[keep])
    out["ols_fraction_sig3_without_suspect_dates"] = float(np.nanmean(t_ols_drop > 3))
    out["dropped_dates"] = sorted(SUSPECT)

    t_rob_drop = robust_se_and_t(db[keep], t[keep], theil_sen_maps(db[keep], t[keep]))
    out["theil_sen_fraction_sig3_without_suspect_dates"] = float(np.nanmean(t_rob_drop > 3))

    out["gaussian_null_fraction_sig3"] = 0.0027
    return out


def section2(db, level):
    spread = 1.4826 * np.nanmedian(np.abs(db - np.nanmedian(db, axis=0)), axis=0)
    out = {}
    for thr in (-18.0, -15.0, -12.0):
        water = level < thr
        land = (level >= thr) & np.isfinite(level)
        out[f"water_below_{thr:.0f}dB"] = {
            "water_fraction": float(np.nanmean(water[np.isfinite(level)])),
            "median_spread_water_db": float(np.nanmedian(spread[water])) if water.any() else None,
            "median_spread_land_db": float(np.nanmedian(spread[land])) if land.any() else None,
        }
    # Is spread higher in dim land than in bright land, within the land class?
    land = level >= -15
    lo = land & (level < np.nanpercentile(level[land], 50))
    hi = land & (level >= np.nanpercentile(level[land], 50))
    out["land_dim_half_spread_db"] = float(np.nanmedian(spread[lo]))
    out["land_bright_half_spread_db"] = float(np.nanmedian(spread[hi]))
    return out


def inject(db, r0, c0, size, delta, start_idx):
    out = db.copy()
    out[start_idx:, r0:r0 + size, c0:c0 + size] += delta  # dB addition == multiplicative in linear
    return out


def leave_one_out_anomaly(db):
    anom = np.empty_like(db)
    for t in range(db.shape[0]):
        anom[t] = db[t] - np.nanmedian(np.delete(db, t, axis=0), axis=0)
    return anom


def section3(db, dates, controls, sizes, deltas, durations):
    n = db.shape[0]
    half = n // 2
    cal = list(range(half))
    test = list(range(half, n))

    # Control anomalies: calibrate sigma on first half, measure FA on second half.
    # Edge control tiles can be narrower than TILE; each tile keeps its real shape.
    ctrl_anom = []
    for r0, c0 in controls:
        blk = db[:, r0:r0 + TILE, c0:c0 + TILE]
        ctrl_anom.append(leave_one_out_anomaly(blk))
    cal_vals = np.concatenate([a[cal][np.isfinite(a[cal])] for a in ctrl_anom])
    sigma = robust_summary(cal_vals).robust_scale
    test_vals = np.concatenate([a[test][np.isfinite(a[test])] for a in ctrl_anom])
    fa_pixel = float(np.mean(np.abs(test_vals) > SIG_K * sigma))
    # Patch-level FA on controls: non-overlapping patches of each size.
    patch_fa = {}
    for size in sizes:
        flags = []
        for a in ctrl_anom:
            ff = np.abs(a[test]) > SIG_K * sigma
            for t in range(ff.shape[0]):
                for r in range(0, ff.shape[1] - size + 1, size):
                    for c in range(0, ff.shape[2] - size + 1, size):
                        flags.append(ff[t, r:r + size, c:c + size].mean() >= 0.5)
        patch_fa[str(size)] = float(np.mean(flags)) if flags else None

    # Injected detection on non-control tiles, avoiding controls.
    cand = [(512, 512), (768, 768), (1024, 256), (1792, 768), (256, 256)]
    results = []
    for size, delta, dur in itertools.product(sizes, deltas, durations):
        start = n - dur
        pix_tp, patch_tp, patch_total = 0, 0, 0
        for r0, c0 in cand:
            blk = db[:, r0:r0 + TILE, c0:c0 + TILE]
            inj = inject(blk, 64, 64, size, delta, start)
            anom = leave_one_out_anomaly(inj)
            flag = np.abs(anom) > SIG_K * sigma
            injected_region = np.zeros(blk.shape[1:], dtype=bool)
            injected_region[64:64 + size, 64:64 + size] = True
            for t in range(start, n):
                f = flag[t][injected_region]
                pix_tp += f.sum()
                patch_total += 1
                if f.mean() >= 0.5:
                    patch_tp += 1
        total_pixels = len(cand) * dur * size * size
        results.append({
            "patch_side_px": size, "patch_side_m": 10 * size,
            "delta_db": delta, "duration_dates": dur,
            "pixel_tpr": float(pix_tp / max(total_pixels, 1)),
            "patch_detection_rate": float(patch_tp / max(patch_total, 1)),
        })
    return {
        "sigma_calibrated_db": float(sigma),
        "pixel_false_alarm_rate_controls_heldout": fa_pixel,
        "patch_false_alarm_rate_by_side_px": patch_fa,
        "sensitivity": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.stack / "stack_aoi.npz")
    dates = z["dates"]
    days = np.array([(np.datetime64(d) - np.datetime64(dates[0])).astype(int) for d in dates])
    t = (days / DAYS_PER_YEAR).astype(np.float64)
    db = np.stack([to_db(z["A_values"][i], z["A_valid"][i]) for i in range(len(dates))])
    level = np.nanmedian(db, axis=0)

    out = {"dates": [str(d) for d in dates]}
    print("section 1: robust slopes", flush=True)
    out["section1_robust_slopes"] = section1(db, dates, t)
    print(json.dumps(out["section1_robust_slopes"], indent=2), flush=True)

    print("section 2: water split", flush=True)
    out["section2_water_split"] = section2(db, level)
    print(json.dumps(out["section2_water_split"], indent=2), flush=True)

    print("section 3: injected changes", flush=True)
    out["section3_injection"] = section3(
        db, dates, CONTROLS,
        sizes=[4, 8], deltas=[1.0, 2.0, 3.0, 5.0], durations=[1, 3],
    )
    s3 = out["section3_injection"]
    print(f"sigma {s3['sigma_calibrated_db']:.2f} dB; pixel FA {s3['pixel_false_alarm_rate_controls_heldout']:.4f}; "
          f"patch FA {s3['patch_false_alarm_rate_by_side_px']}", flush=True)
    for r in s3["sensitivity"]:
        print(f"  {r['patch_side_m']} m  delta {r['delta_db']:+.0f} dB  dur {r['duration_dates']}  "
              f"pixel TPR {r['pixel_tpr']:.3f}  patch detect {r['patch_detection_rate']:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "followup_checks.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

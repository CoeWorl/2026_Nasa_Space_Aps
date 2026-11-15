"""Which places change more, or faster, over the 27-date stack?

    uv run python -u scripts/temporal_trends.py --stack DIR --out-dir DIR

For every 256-pixel tile:
  * temporal variability: median of each pixel's robust temporal spread (MAD of dB over dates);
  * linear trend: each pixel's OLS slope in dB per year, with its standard error;
  * significance: fraction of pixels whose |slope| / SE exceeds 3;
  * common-mode check: the same statistics after removing the scene-wide median
    anomaly for each date. If the tile trends are shared across the scene, they
    should shrink after this removal.

Nothing here is a change-point test. A linear slope over 27 noisy dates is
descriptive, and the significance is per pixel, not corrected for the number of pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from groundtruth.statistics import robust_summary

TILE = 256
DAYS_PER_YEAR = 365.25


def to_db(linear: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = valid & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok].astype(np.float64)).astype(np.float32)
    return out


def tiles(shape: tuple[int, int]):
    h, w = shape
    for r0 in range(0, h, TILE):
        for c0 in range(0, w, TILE):
            yield r0, c0, slice(r0, min(r0 + TILE, h)), slice(c0, min(c0 + TILE, w))


def pixel_trend(db: np.ndarray, t_years: np.ndarray):
    """Per-pixel OLS slope (dB/yr), its standard error, and residual sigma, NaN-aware."""
    y = db.astype(np.float64)
    ok = np.isfinite(y)
    n = ok.sum(axis=0).astype(np.float64)
    t = t_years[:, None, None] * np.ones_like(y)
    t = np.where(ok, t, np.nan)
    tm = np.nanmean(t, axis=0)
    ym = np.nanmean(y, axis=0)
    dt = t - tm
    dy = y - ym
    sxx = np.nansum(dt * dt, axis=0)
    sxy = np.nansum(dt * dy, axis=0)
    slope = np.where(sxx > 0, sxy / sxx, np.nan)
    intercept = ym - slope * tm
    resid = y - (intercept + slope * t)
    dof = np.maximum(n - 2, 1)
    sigma = np.sqrt(np.nansum(resid * resid, axis=0) / dof)
    se = np.where(sxx > 0, sigma / np.sqrt(sxx), np.nan)
    enough = n >= 15
    slope = np.where(enough, slope, np.nan)
    se = np.where(enough, se, np.nan)
    return slope, se, sigma


def temporal_spread(db: np.ndarray) -> np.ndarray:
    """Robust per-pixel spread over dates: 1.4826 x MAD of each pixel's dB series."""
    med = np.nanmedian(db, axis=0)
    mad = np.nanmedian(np.abs(db - med), axis=0)
    return 1.4826 * mad


def tile_summary(map_arr: np.ndarray) -> dict:
    vals = map_arr[np.isfinite(map_arr)]
    if vals.size < 1000:
        return {}
    return {"median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.stack / "stack_aoi.npz")
    dates = z["dates"]
    days = np.array([(np.datetime64(d) - np.datetime64(dates[0])).astype(int) for d in dates])
    t_years = days / DAYS_PER_YEAR
    db = np.stack([to_db(z["A_values"][t], z["A_valid"][t]) for t in range(len(dates))])

    # Common-mode removal: subtract each date's scene-wide median anomaly.
    scene_offset = np.nanmedian(db.reshape(db.shape[0], -1), axis=1)
    db_cm = db - scene_offset[:, None, None]

    slope, se, sigma = pixel_trend(db, t_years)
    slope_cm, se_cm, sigma_cm = pixel_trend(db_cm, t_years)
    spread = temporal_spread(db)
    sig = np.abs(slope) / se
    sig_cm = np.abs(slope_cm) / se_cm

    tiles_out = []
    for r0, c0, rs, cs in tiles(db.shape[1:]):
        sl = slope[rs, cs]
        sl_cm = slope_cm[rs, cs]
        fin = np.isfinite(sl)
        if fin.sum() < 1000:
            continue
        tiles_out.append({
            "row": r0, "col": c0,
            "median_spread_db": float(np.nanmedian(spread[rs, cs])),
            "median_slope_db_per_yr": float(np.nanmedian(sl)),
            "p10_slope": float(np.nanpercentile(sl, 10)),
            "p90_slope": float(np.nanpercentile(sl, 90)),
            "median_se_db_per_yr": float(np.nanmedian(se[rs, cs])),
            "fraction_pixels_sig3": float(np.nanmean(sig[rs, cs] > 3)),
            "median_slope_common_mode_removed": float(np.nanmedian(sl_cm)),
            "fraction_pixels_sig3_common_mode_removed": float(np.nanmean(sig_cm[rs, cs] > 3)),
        })

    rows = np.array([[t["median_slope_db_per_yr"], t["median_slope_common_mode_removed"],
                      t["median_se_db_per_yr"]] for t in tiles_out])
    slope_spread = float(np.std(rows[:, 0]))
    se_typical = float(np.median(rows[:, 2]))
    common_mode_slope_sd = float(np.std(rows[:, 0]) / max(np.std(rows[:, 1]), 1e-9))

    # Most and least variable tiles, by temporal spread, and fastest trends by median slope.
    by_spread = sorted(tiles_out, key=lambda x: -x["median_spread_db"])
    by_slope = sorted(tiles_out, key=lambda x: -abs(x["median_slope_db_per_yr"]))

    result = {
        "dates": [str(d) for d in dates],
        "scene_common_mode_offset_db": {str(d): float(o) for d, o in zip(dates, scene_offset)},
        "tiles": tiles_out,
        "summary": {
            "n_tiles": len(tiles_out),
            "scene_spread_db_median": float(np.median([t["median_spread_db"] for t in tiles_out])),
            "tile_slope_sd_db_per_yr": slope_spread,
            "typical_pixel_slope_se_db_per_yr": se_typical,
            "tile_slope_sd_over_common_mode_sd": common_mode_slope_sd,
            "fraction_pixels_sig3_all": float(np.nanmean(sig > 3)),
            "fraction_pixels_sig3_common_mode_removed": float(np.nanmean(sig_cm > 3)),
        },
        "most_variable_tiles": by_spread[:5],
        "largest_median_trend_tiles": by_slope[:5],
        "least_variable_tiles": sorted(tiles_out, key=lambda x: x["median_spread_db"])[:3],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "temporal_trends.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    np.savez_compressed(args.out_dir / "temporal_maps.npz", slope=slope.astype(np.float32),
                        se=se.astype(np.float32), spread=spread.astype(np.float32),
                        slope_common_mode=slope_cm.astype(np.float32))
    print(json.dumps(result["summary"], indent=2))
    print("\nmost variable:", [(t["row"], t["col"], round(t["median_spread_db"], 2)) for t in by_spread[:5]])
    print("largest trend:", [(t["row"], t["col"], round(t["median_slope_db_per_yr"], 2)) for t in by_slope[:5]])
    print(f"\nwrote {args.out_dir / 'temporal_trends.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

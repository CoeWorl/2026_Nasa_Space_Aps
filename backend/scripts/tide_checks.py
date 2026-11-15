"""Controls and holdout for the Jamaica Bay tide-backscatter result.

    uv run python -u scripts/tide_checks.py --stack DIR --tide-json FILE --polygons GEOJSON --out-dir DIR

Three checks, each prespecified:
  1. Weather covariates. Daily NOAA summaries for JFK (USW00094789): average wind (AWND, m/s),
     fastest 2-minute gust (WSF2, m/s), daily rainfall (PRCP, mm), and rainfall over the
     three days before the overpass. Hourly NOAA data were not returned by the API, so these are
     full-day values, not values at the 23:52 UTC overpass. Overpass is 19:52 local time, so the
     acquisition date matches the local day.
  2. Open-water contamination. A pixel counts as open water if its temporal-median backscatter is
     below -15 dB. Polygons with more than 10% open water are flagged and the test is repeated without them.
  3. Chronological holdout. Fit on the first 13 acquisitions, predict the last 13 with the
     water-level coefficient only, and report the out-of-sample correlation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import stats
from shapely.geometry import Polygon, shape

WATER_DB = -15.0
OPEN_WATER_MAX = 0.10
NOAA = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_noaa(station: str, start: str, end: str) -> dict[str, dict]:
    q = f"{NOAA}?dataset=daily-summaries&stations={station}&startDate={start}&endDate={end}&dataTypes=PRCP,AWND,WSF2&format=json&units=metric"
    with urllib.request.urlopen(q, timeout=300) as r:
        rows = json.load(r)
    return {row["DATE"]: row for row in rows}


def f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--tide-json", type=Path, required=True)
    ap.add_argument("--polygons", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    reg = json.loads(args.tide_json.read_text())
    per = reg["per_date"]
    dates = [p["date"] for p in per]
    h = np.array([p["h_m"] for p in per])
    y = np.array([p["marsh_minus_upland_db"] for p in per])
    doy = np.array([date.fromisoformat(d).timetuple().tm_yday for d in dates], float)

    # 1. weather
    d0 = date.fromisoformat(dates[0]) - timedelta(days=5)
    d1 = date.fromisoformat(dates[-1])
    noaa = fetch_noaa("USW00094789", d0.isoformat(), d1.isoformat())
    wind = np.array([f(noaa[d]["AWND"]) for d in dates])
    gust = np.array([f(noaa[d]["WSF2"]) for d in dates])
    rain = np.array([f(noaa[d].get("PRCP")) or 0.0 for d in dates])
    antecedent = np.array([sum(f(noaa.get((date.fromisoformat(d) - timedelta(days=k)).isoformat(), {}).get("PRCP")) or 0.0
                               for k in (1, 2, 3)) for d in dates])
    weather_rows = [{"date": d, "AWND_mps": w, "WSF2_mps": g, "PRCP_mm": r, "PRCP_3d_prior_mm": a,
                     "h_m": hh, "marsh_minus_upland_db": yy}
                    for d, w, g, r, a, hh, yy in zip(dates, wind, gust, rain, antecedent, h, y)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "weather_covariates.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(weather_rows[0].keys()))
        wr.writeheader()
        wr.writerows(weather_rows)

    def ols(columns: dict[str, np.ndarray], target: np.ndarray) -> dict:
        names = list(columns)
        X = np.column_stack([np.ones_like(target)] + [columns[k] for k in names])
        b, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ b
        n, k = X.shape
        s2 = (resid @ resid) / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        out = {"n": int(n), "R2": float(1 - (resid @ resid) / ((target - target.mean()) @ (target - target.mean())))}
        for i, nm in enumerate(["intercept"] + names, start=0):
            out[nm] = {"coef": float(b[i]), "se": float(se[i]), "t": float(b[i] / se[i])}
        return out

    seas = {"sin_doy": np.sin(2 * np.pi * doy / 365.25), "cos_doy": np.cos(2 * np.pi * doy / 365.25)}
    weather = {
        "water_only": ols({"h": h}, y),
        "water_plus_season": ols({"h": h, **seas}, y),
        "water_plus_wind_and_rain": ols({"h": h, "AWND": wind, "PRCP": rain, "PRCP_3d": antecedent}, y),
        "water_plus_season_wind_rain": ols({"h": h, **seas, "AWND": wind, "PRCP": rain, "PRCP_3d": antecedent}, y),
        "wind_alone_r": float(stats.pearsonr(wind, y)[0]),
        "rain_alone_r": float(stats.pearsonr(rain, y)[0]),
    }

    # 2. open-water contamination per polygon
    z = np.load(args.stack / "stack_aoi.npz")
    lin = z["A_values"]
    valid = z["A_valid"]
    db = np.where(valid & (lin > 0), 10 * np.log10(np.where(lin > 0, lin, 1.0)), np.nan)
    level = np.nanmedian(db, axis=0)
    transform = Affine(*z["A_transform"][0][:6])
    H, W = level.shape
    fc = json.loads(args.polygons.read_text())
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    shapes = []
    for ft in fc["features"]:
        g = shape(ft["geometry"])
        xs, ys = zip(*[to_utm.transform(x, yy) for x, yy in g.exterior.coords])
        shapes.append(Polygon(list(zip(xs, ys))))
    pid = rasterize([(s, k + 1) for k, s in enumerate(shapes)], out_shape=(H, W), transform=transform,
                    fill=0, dtype="int32")

    water_frac = {}
    for k in range(1, len(shapes) + 1):
        m = pid == k
        v = level[m]
        v = v[np.isfinite(v)]
        water_frac[k] = float(np.mean(v < WATER_DB)) if v.size >= 20 else None

    # per-polygon series, same dates as the regression
    dates_arr = z["dates"]
    keep_idx = [int(np.where(dates_arr == d)[0][0]) for d in dates]
    series = {}
    for k in range(1, len(shapes) + 1):
        m = pid == k
        vals = []
        for i in keep_idx:
            v = db[i][m]
            v = v[np.isfinite(v)]
            vals.append(np.median(v) if v.size >= 20 else np.nan)
        series[k] = np.array(vals)

    def corr_for(keys: list[int]) -> dict:
        rs = []
        for k in keys:
            s = series[k]
            ok = np.isfinite(s)
            if ok.sum() >= 10:
                rs.append(stats.pearsonr(h[ok], s[ok])[0])
        rs = np.array(rs)
        return {"polygons": int(rs.size), "median_r": float(np.median(rs)) if rs.size else None,
                "n_negative": int((rs < 0).sum()), "n_positive": int((rs > 0).sum())}

    clean = [k for k, wf in water_frac.items() if wf is not None and wf <= OPEN_WATER_MAX]
    flagged = [k for k, wf in water_frac.items() if wf is not None and wf > OPEN_WATER_MAX]
    water = {
        "threshold_db": WATER_DB,
        "max_open_water_fraction_kept": OPEN_WATER_MAX,
        "polygons_all": corr_for(list(water_frac)),
        "polygons_kept": corr_for(clean),
        "polygons_flagged_open_water": corr_for(flagged),
        "open_water_fraction_by_polygon": {str(k): wf for k, wf in water_frac.items()},
    }

    # 3. chronological holdout
    n = len(h)
    half = n // 2
    tr, te = slice(0, half), slice(half, n)
    b1, a1 = np.polyfit(h[tr], y[tr], 1)
    pred = a1 + b1 * h[te]
    r_out = float(stats.pearsonr(pred, y[te])[0])
    rmse_model = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
    rmse_null = float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
    holdout = {
        "train_dates": [dates[0], dates[half - 1]],
        "test_dates": [dates[half], dates[-1]],
        "train_slope_db_per_m": float(b1),
        "out_of_sample_r": r_out,
        "rmse_model_db": rmse_model,
        "rmse_mean_only_db": rmse_null,
        "skill_vs_mean_only": float(1 - rmse_model ** 2 / rmse_null ** 2),
    }

    result = {"weather": weather, "open_water": water, "holdout": holdout,
              "weather_source": "NOAA daily summaries, JFK (USW00094789)"}
    (args.out_dir / "tide_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("weather", "holdout")}, indent=2, default=float), flush=True)
    print(json.dumps({"open_water": {k: v for k, v in water.items() if k != "open_water_fraction_by_polygon"}},
                     indent=2), flush=True)
    print(f"\nwrote {args.out_dir / 'tide_checks.json'} and weather_covariates.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

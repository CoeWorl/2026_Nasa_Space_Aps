"""Quality-screened and fixed-footprint tide regressions, both tracks.

    uv run python -u scripts/tide_qc.py --out-dir DIR

POST HOC QUALITY CONTROL. The 90% coverage threshold was chosen after the ascending-track
outlier was seen, so these results are a sensitivity analysis, not preregistered evidence.
The unfiltered results are reported alongside.

For each track, per acquisition:
  * marsh coverage  = fraction of marsh-polygon pixels that are valid (pipeline validity rule)
  * upland coverage = fraction of upland-reference pixels that are valid
Analyses:
  A  all dates (original variable footprint)
  B  quality-screened: marsh and upland coverage both >= 0.90
  C  fixed footprint on the screened dates: marsh and upland medians over pixels valid on EVERY screened date
  D  fixed footprint on all dates: as C, over pixels valid on every date of the track
Each analysis reports OLS slope with its t-based 95% CI, Pearson, Spearman, Theil-Sen slope,
leave-one-date-out range, and the water levels of the dates used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import stats
from shapely.geometry import Polygon, shape

COVERAGE_MIN = 0.90
UPLAND_MIN_DB = -5.0
VERSION_RE = re.compile(r"_(\d{3})$")


def to_db(lin, valid):
    out = np.full(lin.shape, np.nan, dtype=np.float32)
    ok = valid & (lin > 0)
    out[ok] = 10.0 * np.log10(lin[ok].astype(np.float64)).astype(np.float32)
    return out


def load_track(stack: Path, tides_csv: Path, polygons: Path):
    z = np.load(stack / "stack_aoi.npz")
    dates_all = z["dates"]
    pids = z["product_ids"]
    keep = {}
    for i, (d, p) in enumerate(zip(dates_all, pids)):
        v = int(VERSION_RE.search(str(p)).group(1))
        if d not in keep or v > keep[d][1]:
            keep[d] = (i, v)
    sel = sorted((i for i, _ in keep.values()), key=lambda i: str(dates_all[i]))
    dates = dates_all[sel]
    tides = {r["date"]: r for r in csv.DictReader(tides_csv.open())}
    lin = z["A_values"][sel]
    valid = z["A_valid"][sel]
    db = np.stack([to_db(lin[t], valid[t]) for t in range(len(sel))])
    valid = np.isfinite(db)
    transform = Affine(*z["A_transform"][0][:6])
    H, W = db.shape[1:]

    fc = json.loads(polygons.read_text())
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    polys = []
    for f in fc["features"]:
        g = shape(f["geometry"])
        xs, ys = zip(*[to_utm.transform(x, y) for x, y in g.exterior.coords])
        polys.append(Polygon(list(zip(xs, ys))))
    pid = rasterize([(p, k + 1) for k, p in enumerate(polys)], out_shape=(H, W), transform=transform,
                    fill=0, dtype="int32")
    marsh = pid > 0
    level = np.nanmedian(db, axis=0)
    upland = (~marsh) & np.isfinite(level) & (level >= UPLAND_MIN_DB)

    h = np.array([0.5 * (float(tides[str(d)]["inwood_m"]) + float(tides[str(d)]["rockaway_m"])) for d in dates])
    cov_m = valid[:, marsh].mean(axis=1)
    cov_u = valid[:, upland].mean(axis=1)
    return {"dates": [str(d) for d in dates], "h": h, "db": db, "valid": valid,
            "marsh": marsh, "upland": upland, "cov_m": cov_m, "cov_u": cov_u,
            # extras for tide_adjust.py; the regressions above do not use them
            "sel": sel, "polygon_raster": pid, "features": fc["features"],
            "product_ids": [str(pids[i]) for i in sel],
            "dropped_duplicates": [str(p) for i, p in enumerate(pids) if i not in set(sel)]}


def regress(y, x):
    n = len(x)
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    ci = [lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    ts = stats.theilslopes(y, x)
    loo = []
    for j in range(n):
        m = np.ones(n, bool)
        m[j] = False
        loo.append(stats.pearsonr(x[m], y[m])[0])
    return {"n": int(n), "slope_db_per_m": float(lr.slope), "slope_ols_95ci": [float(ci[0]), float(ci[1])],
            "slope_ols_p": float(lr.pvalue), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
            "theil_sen_slope": float(ts[0]), "theil_sen_ci": [float(ts[2]), float(ts[3])],
            "loo_r_range": [float(min(loo)), float(max(loo))]}


def run(track: dict, idx: np.ndarray, fixed: bool):
    db, marsh, upland = track["db"][idx], track["marsh"], track["upland"]
    if fixed:
        vm = np.all(track["valid"][idx][:, marsh], axis=0)
        vu = np.all(track["valid"][idx][:, upland], axis=0)
        m_pix = marsh.copy()
        m_pix[marsh] = vm
        u_pix = upland.copy()
        u_pix[upland] = vu
    else:
        m_pix, u_pix = marsh, upland
    mdb = np.array([np.nanmedian(db[t][m_pix]) for t in range(len(idx))])
    udb = np.array([np.nanmedian(db[t][u_pix]) for t in range(len(idx))])
    y_rel = mdb - udb
    h = track["h"][idx]
    out = {
        "fixed_footprint": fixed,
        "marsh_pixels": int(m_pix.sum()),
        "upland_pixels": int(u_pix.sum()),
        "dates": [track["dates"][i] for i in idx],
        "h_range_m": [float(h.min()), float(h.max())],
        "marsh_minus_upland": regress(y_rel, h),
        "marsh_only": regress(mdb, h),
        "upland_only": regress(udb, h),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--polygons", type=Path, default=Path("data/processed/jamaica_bay/marsh_polygons_osm.geojson"))
    args = ap.parse_args()
    base = Path("data/processed")
    tracks = {
        "069D068_descending": load_track(base / "stack_bay_069D068", base / "jamaica_bay/tide_matches_27.csv", args.polygons),
        "133A022_ascending": load_track(base / "stack_bay_133A022", base / "jamaica_bay/tide_matches_133A022.csv", args.polygons),
    }
    result = {"coverage_threshold": COVERAGE_MIN, "post_hoc": True, "tracks": {}}
    for name, tr in tracks.items():
        screened = np.where((tr["cov_m"] >= COVERAGE_MIN) & (tr["cov_u"] >= COVERAGE_MIN))[0]
        all_idx = np.arange(len(tr["dates"]))
        coverage = [{"date": d, "h_m": float(h), "marsh_cov": float(cm), "upland_cov": float(cu),
                     "screened": bool(i in set(screened.tolist()))}
                    for i, (d, h, cm, cu) in enumerate(zip(tr["dates"], tr["h"], tr["cov_m"], tr["cov_u"]))]
        result["tracks"][name] = {
            "coverage_by_date": coverage,
            "n_all": len(all_idx), "n_screened": len(screened),
            "excluded_dates": [tr["dates"][i] for i in all_idx if i not in set(screened.tolist())],
            "A_all_variable_footprint": run(tr, all_idx, fixed=False),
            "B_screened_variable_footprint": run(tr, screened, fixed=False) if len(screened) >= 8 else None,
            "C_screened_fixed_footprint": run(tr, screened, fixed=True) if len(screened) >= 8 else None,
            "D_all_fixed_footprint": run(tr, all_idx, fixed=True),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tide_qc.json").write_text(json.dumps(result, indent=2, default=float) + "\n")

    for name, r in result["tracks"].items():
        print(f"\n== {name}: all {r['n_all']}, screened {r['n_screened']}, excluded {r['excluded_dates']}")
        for key in ("A_all_variable_footprint", "B_screened_variable_footprint",
                    "C_screened_fixed_footprint", "D_all_fixed_footprint"):
            v = r[key]
            if v is None:
                print(f"  {key}: too few dates")
                continue
            m = v["marsh_minus_upland"]
            print(f"  {key:30s} n={m['n']:2d} slope {m['slope_db_per_m']:+.2f} OLS CI [{m['slope_ols_95ci'][0]:+.2f},{m['slope_ols_95ci'][1]:+.2f}] "
                  f"r {m['pearson_r']:+.2f} (p {m['pearson_p']:.3f}) rho {m['spearman_rho']:+.2f} TS {m['theil_sen_slope']:+.2f} "
                  f"marsh px {v['marsh_pixels']}  upland r {v['upland_only']['pearson_r']:+.2f}")
    print(f"\nwrote {args.out_dir / 'tide_qc.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

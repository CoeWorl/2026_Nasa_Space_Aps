"""Test whether Jamaica Bay marsh backscatter follows water level at the overpass.

    uv run python -u scripts/tide_regression.py --stack DIR --tides CSV --polygons GEOJSON --out-dir DIR

Pre-registered (fixed before results):
  * Primary test: Y_t = median marsh HH (dB, all marsh polygons pooled) minus median upland HH (dB),
    regressed on h_t = mean of the two gauge levels (m NAVD88) at the overpass.
  * Upland reference: pixels outside every marsh polygon with temporal-median backscatter
    >= -5 dB (brightness proxy for land, not a land-cover product).
  * Duplicate acquisitions: when two granules share an acquisition time, the higher processing
    version is kept (_002 over _001).
  * Secondary: marsh alone and upland alone against h_t (to see whether the link is marsh-specific),
    Spearman as well as Pearson, bootstrap 95% CI over dates, leave-one-date-out range.
  * Exploratory: each marsh polygon separately. Not used for the primary conclusion.
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
from shapely.geometry import shape

UPLAND_MIN_DB = -5.0
VERSION_RE = re.compile(r"_(\d{3})$")


def to_db(linear, valid):
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = valid & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok].astype(np.float64)).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--tides", type=Path, required=True)
    ap.add_argument("--polygons", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    z = np.load(args.stack / "stack_aoi.npz")
    dates = z["dates"]
    pids = z["product_ids"]
    tides = {r["date"]: r for r in csv.DictReader(args.tides.open())}

    # Keep the highest processing version per acquisition date.
    keep = {}
    for i, (d, p) in enumerate(zip(dates, pids)):
        v = int(VERSION_RE.search(str(p)).group(1))
        if d not in keep or v > keep[d][1]:
            keep[d] = (i, v)
    idx = sorted(keep.values(), key=lambda x: str(dates[x[0]]))
    sel = [i for i, _ in idx]
    dates = dates[sel]
    print(f"acquisitions after duplicate removal: {len(sel)}", flush=True)

    lin = z["A_values"][sel]
    valid = z["A_valid"][sel]
    db = np.stack([to_db(lin[t], valid[t]) for t in range(len(sel))])
    transform = Affine(*z["A_transform"][0][:6])
    H, W = db.shape[1:]

    # Rasterise marsh polygons (OSM, lon/lat -> UTM 18N).
    fc = json.loads(args.polygons.read_text())
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    shapes = []
    for f in fc["features"]:
        g = shape(f["geometry"])
        xs, ys = zip(*[to_utm.transform(x, y) for x, y in g.exterior.coords])
        from shapely.geometry import Polygon

        shapes.append((Polygon(list(zip(xs, ys))), f["properties"].get("wetland", "")))
    poly_ids = rasterize([(s, k + 1) for k, (s, _) in enumerate(shapes)], out_shape=(H, W),
                         transform=transform, fill=0, dtype="int32")
    marsh = poly_ids > 0

    level = np.nanmedian(db, axis=0)
    upland = (~marsh) & np.isfinite(level) & (level >= UPLAND_MIN_DB)
    print(f"marsh pixels {int(marsh.sum())}, upland pixels {int(upland.sum())}, polygons {len(shapes)}", flush=True)

    rows = []
    for t, d in enumerate(dates):
        tr = tides[str(d)]
        h = 0.5 * (float(tr["inwood_m"]) + float(tr["rockaway_m"]))
        m_vals = db[t][marsh & np.isfinite(db[t])]
        u_vals = db[t][upland & np.isfinite(db[t])]
        rows.append({"date": str(d), "h_m": h,
                     "inwood_m": float(tr["inwood_m"]), "rockaway_m": float(tr["rockaway_m"]),
                     "marsh_db": float(np.median(m_vals)), "upland_db": float(np.median(u_vals)),
                     "n_marsh": int(m_vals.size), "n_upland": int(u_vals.size)})
    rows = np.array([(r["h_m"], r["marsh_db"], r["upland_db"]) for r in rows])
    h, mdb, udb = rows[:, 0], rows[:, 1], rows[:, 2]
    y_rel = mdb - udb

    def stats_for(y, x):
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        slope = stats.linregress(x, y).slope
        rng = np.random.default_rng(20261114)
        boots = []
        n = len(x)
        for _ in range(args.boot):
            k = rng.integers(0, n, n)
            if np.ptp(x[k]) > 0:
                boots.append(stats.linregress(x[k], y[k]).slope)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        loo = []
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            loo.append(stats.pearsonr(x[m], y[m])[0])
        return {"pearson_r": float(pr), "pearson_p": float(pp), "spearman_rho": float(sr),
                "spearman_p": float(sp), "slope_db_per_m": float(slope),
                "slope_bootstrap_95ci_db_per_m": [float(lo), float(hi)],
                "leave_one_date_out_r_range": [float(min(loo)), float(max(loo))]}

    result = {
        "n_acquisitions": len(h),
        "h_range_m": [float(h.min()), float(h.max())],
        "primary_marsh_minus_upland_vs_h": stats_for(y_rel, h),
        "secondary_marsh_only_vs_h": stats_for(mdb, h),
        "secondary_upland_only_vs_h": stats_for(udb, h),
        "critical_abs_r_p05_n": float(stats.t.ppf(0.975, len(h) - 2) /
                                       np.sqrt(stats.t.ppf(0.975, len(h) - 2) ** 2 + len(h) - 2)),
        "exploratory_per_polygon_r_vs_h": {},
        "per_date": [{"date": str(d), "h_m": float(a[0]), "marsh_db": float(a[1]),
                      "upland_db": float(a[2]), "marsh_minus_upland_db": float(a[1] - a[2])}
                     for d, a in zip(dates, rows)],
        "upland_definition": f"outside marsh polygons, temporal median >= {UPLAND_MIN_DB} dB",
        "duplicate_rule": "keep highest processing version per acquisition",
    }

    # Exploratory per-polygon correlations (not used for the conclusion).
    per_poly = []
    for k, (_, kind) in enumerate(shapes, start=1):
        mask = poly_ids == k
        series = []
        for t in range(len(dates)):
            v = db[t][mask & np.isfinite(db[t])]
            series.append(np.median(v) if v.size >= 20 else np.nan)
        series = np.array(series)
        ok = np.isfinite(series)
        if ok.sum() >= 10:
            r, p = stats.pearsonr(h[ok], series[ok])
            per_poly.append({"polygon": k, "kind": kind, "n": int(ok.sum()), "r": float(r), "p": float(p)})
    per_poly.sort(key=lambda x: x["p"])
    result["exploratory_per_polygon_r_vs_h"] = per_poly

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tide_regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n_acquisitions", "h_range_m", "primary_marsh_minus_upland_vs_h",
                                             "secondary_marsh_only_vs_h", "secondary_upland_only_vs_h")}, indent=2),
          flush=True)
    print(f"\nwrote {args.out_dir / 'tide_regression.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

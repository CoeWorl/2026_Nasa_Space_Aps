"""Geography for the explore page, all from real geometry and real radar.

    uv run --project backend python -u backend/scripts/make_basemap.py --out-dir data/processed/jamaica_bay/app

Run from the repository root. Writes basemap.json holding:
  image         temporal-median HH backscatter of the descending stack (dB), block-averaged
                by FACTOR and tinted. It is a radar picture of the bay, not an optical map and
                not a shoreline product.
  coverage      for every acquisition that fails the coverage screen, a hatched mask of the
                pixels with no valid observation on that date.
  regions       marsh polygon rings (OpenStreetMap) in image pixel coordinates.
  projects      restoration locator points that fall inside the window.
  gauges        the two USGS gauge positions (from the USGS monitoring-locations API).
  locator       simplified Brooklyn and Queens outlines (NYC Open Data borough boundaries,
                clipped to shoreline) and the study window, in lon/lat.
Images are embedded as data URIs so the page needs one JSON file and no new routes.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).parent))
from tide_adjust import TRACKS
from tide_qc import COVERAGE_MIN, load_track

FACTOR = 4  # 10 m pixels -> 40 m
DB_RANGE = (-24.0, 2.0)
WATER_RGB, LAND_RGB = (226, 232, 232), (74, 92, 102)  # pale water, ink land
HATCH_RGBA = (36, 56, 70, 150)
BOROUGHS_URL = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"
USGS_URL = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items/USGS-{}?f=json"
GAUGES = {"inwood": "01311850", "rockaway": "01311875"}


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "geo-client/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def png_uri(rgba: np.ndarray) -> str:
    """HxWxC uint8 -> PNG data URI."""
    bands = np.moveaxis(rgba, -1, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PNG has no georeferencing; that is intended
        with MemoryFile() as mem:
            with mem.open(driver="PNG", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                          dtype="uint8") as dst:
                dst.write(bands)
            data = mem.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def block(a: np.ndarray, fn) -> np.ndarray:
    h, w = (a.shape[0] // FACTOR) * FACTOR, (a.shape[1] // FACTOR) * FACTOR
    return fn(a[:h, :w].reshape(h // FACTOR, FACTOR, w // FACTOR, FACTOR), axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--polygons", type=Path, default=Path("data/processed/jamaica_bay/marsh_polygons_osm.geojson"))
    ap.add_argument("--projects", type=Path, default=Path("configs/restoration_projects.geojson"))
    args = ap.parse_args()
    base = Path("data/processed")

    tracks = {n: load_track(base / c["stack"], base / c["tides"], args.polygons) for n, c in TRACKS.items()}
    z = np.load(base / TRACKS["descending_069"]["stack"] / "stack_aoi.npz")
    a, b, c, d, e, f = z["A_transform"][0][:6]  # x = a*col + c, y = e*row + f (north-up)
    if b != 0 or d != 0:
        raise SystemExit("stack grid is rotated; this script assumes north-up")
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

    def to_px(lon: float, lat: float) -> list[float]:
        x, y = to_utm.transform(lon, lat)
        return [round((x - c) / a / FACTOR, 1), round((y - f) / e / FACTOR, 1)]

    # Radar picture of the bay: temporal median of the descending stack, in dB.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level = block(np.nanmedian(tracks["descending_069"]["db"], axis=0), np.nanmean)
    t = np.clip((np.nan_to_num(level, nan=DB_RANGE[0]) - DB_RANGE[0]) / (DB_RANGE[1] - DB_RANGE[0]), 0, 1) ** 0.8
    rgb = np.stack([(w + (ld - w) * t) for w, ld in zip(WATER_RGB, LAND_RGB)], axis=-1).astype(np.uint8)
    hgt, wid = level.shape

    # Missing-coverage masks, hatched, only for acquisitions that fail the screen.
    rr, cc = np.mgrid[0:hgt, 0:wid]
    hatch = ((rr + cc) % 6) < 2
    coverage = {}
    for name, tr in tracks.items():
        for i, date in enumerate(tr["dates"]):
            if tr["cov_m"][i] >= COVERAGE_MIN and tr["cov_u"][i] >= COVERAGE_MIN:
                continue
            missing = ~block(tr["valid"][i], np.any)
            rgba = np.zeros((hgt, wid, 4), np.uint8)
            rgba[missing & hatch] = HATCH_RGBA
            rgba[missing & ~hatch] = (243, 240, 232, 110)
            coverage[f"{name}_{date}"] = {
                "image": png_uri(rgba), "missing_fraction_of_window": float(missing.mean()),
                "marsh_coverage_fraction": float(tr["cov_m"][i]), "upland_coverage_fraction": float(tr["cov_u"][i])}

    regions = []
    for feat in tracks["descending_069"]["features"]:
        ring = [to_px(x, y) for x, y in feat["geometry"]["coordinates"][0]]
        regions.append({"region_id": f"osm_{feat['properties']['osm_id']}", "ring": ring})

    def inside(p: list[float]) -> bool:
        return 0 <= p[0] <= wid and 0 <= p[1] <= hgt

    projects = []
    for feat in json.loads(args.projects.read_text())["features"]:
        if feat["geometry"]:
            p = to_px(*feat["geometry"]["coordinates"])
            if inside(p):
                projects.append({"project_id": feat["properties"]["project_id"], "px": p,
                                 "geometry_kind": feat["properties"]["geometry_kind"]})

    gauges = {}
    for name, site in GAUGES.items():
        try:
            g = get_json(USGS_URL.format(site))
            gauges[name] = {"usgs_site": site, "name": g["properties"]["monitoring_location_name"],
                            "px": to_px(*g["geometry"]["coordinates"][:2])}
        except (OSError, KeyError, ValueError) as err:  # optional layer; the page works without it
            print(f"gauge {site} not located: {err}")

    locator = None
    try:
        outlines = []
        for feat in get_json(BOROUGHS_URL)["features"]:
            boro = feat["properties"].get("boroname") or feat["properties"].get("boro_name")
            if boro not in ("Brooklyn", "Queens"):
                continue
            geom = shape(feat["geometry"]).simplify(0.0015)
            polys = getattr(geom, "geoms", [geom])
            rings = [[[round(x, 4), round(y, 4)] for x, y in p.exterior.coords] for p in polys if p.area > 2e-6]
            outlines.append({"borough": boro, "rings": rings})
        from_utm = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
        corners = [from_utm.transform(c + a * FACTOR * i, f + e * FACTOR * j)
                   for i, j in ((0, 0), (wid, 0), (wid, hgt), (0, hgt))]
        locator = {"source": "NYC Open Data borough boundaries (clipped to shoreline), simplified",
                   "boroughs": outlines, "window": [[round(x, 4), round(y, 4)] for x, y in corners]}
    except (OSError, KeyError, ValueError) as err:
        print(f"borough outlines not fetched: {err}")

    out = {
        "width": wid, "height": hgt, "pixel_m": abs(a) * FACTOR, "crs": "EPSG:32618 (UTM 18N), north up",
        "image": png_uri(rgb),
        "image_note": "Temporal median of NISAR HH backscatter, descending track 069, "
                      f"{len(tracks['descending_069']['dates'])} dates, {abs(a) * FACTOR:.0f} m pixels. "
                      "Dark is radar-bright (buildings, vegetation); pale is radar-dark (open water).",
        "coverage": coverage, "regions": regions, "projects": projects, "gauges": gauges, "locator": locator,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "basemap.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB): image {wid}x{hgt}, {len(regions)} regions, "
          f"coverage masks {list(coverage)}, projects {[p['project_id'] for p in projects]}, gauges {list(gauges)}, "
          f"locator {'yes' if locator else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

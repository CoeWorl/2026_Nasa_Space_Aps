"""Match each NISAR overpass to the nearest USGS continuous water level (parameter 62620).

    uv run python -u scripts/match_tides.py --stack DIR --out-csv FILE [--window-min 60]

Gauges: USGS 01311850 (Jamaica Bay at Inwood), 01311875 (Rockaway Inlet near Floyd Bennett Field).
The OGC continuous-values API returns feet; values are converted to metres (x 0.3048).
Acquisition time comes from the granule name (UTC). Each match records the time offset,
the USGS approval status, and the reading used. No interpolation, no daily means.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

BASE = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items"
SITES = {"inwood": "01311850", "rockaway": "01311875"}
FT_TO_M = 0.3048
NAME_RE = re.compile(r"_(\d{8})T(\d{6})_")


def acquired_at(product_id: str) -> datetime:
    m = NAME_RE.search(product_id)
    if not m:
        raise ValueError(f"no start time in {product_id}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def fetch(site: str, start: datetime, end: datetime) -> list[dict]:
    params = {
        "monitoring_location_id": f"USGS-{site}",
        "parameter_code": "62620",
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "f": "json",
        "limit": 1000,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)
    out = [f["properties"] for f in data.get("features", [])]
    # follow pagination if the server offers a next link
    nxt = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
    while nxt:
        with urllib.request.urlopen(nxt, timeout=120) as r:
            data = json.load(r)
        out += [f["properties"] for f in data.get("features", [])]
        nxt = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
    return out


def nearest(props: list[dict], t: datetime, max_min: float) -> dict | None:
    best, best_dt = None, None
    for p in props:
        if p.get("value") in (None, ""):
            continue
        pt = datetime.fromisoformat(p["time"])
        dt = abs((pt - t).total_seconds()) / 60.0
        if dt <= max_min and (best_dt is None or dt < best_dt):
            best, best_dt = p, dt
    if best is None:
        return None
    return {"value_m": float(best["value"]) * FT_TO_M, "time_utc": best["time"],
            "offset_min": round((datetime.fromisoformat(best["time"]) - t).total_seconds() / 60.0, 2),
            "status": best.get("approval_status")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--window-min", type=float, default=60.0)
    args = ap.parse_args()

    z = np.load(args.stack / "stack_aoi.npz")
    rows = []
    for date, pid in zip(z["dates"], z["product_ids"]):
        t = acquired_at(str(pid))
        row = {"date": str(date), "acquired_utc": t.isoformat(), "product_id": str(pid)}
        for name, site in SITES.items():
            props = fetch(site, t - timedelta(minutes=args.window_min), t + timedelta(minutes=args.window_min))
            m = nearest(props, t, args.window_min)
            row[f"{name}_m"] = None if m is None else round(m["value_m"], 4)
            row[f"{name}_offset_min"] = None if m is None else m["offset_min"]
            row[f"{name}_status"] = None if m is None else m["status"]
            row[f"{name}_n_points_window"] = len(props)
        rows.append(row)
        print(f"{row['date']} {t:%H:%M}Z  inwood {row['inwood_m']} ({row['inwood_offset_min']} min, n={row['inwood_n_points_window']})  "
              f"rockaway {row['rockaway_m']} ({row['rockaway_offset_min']} min, n={row['rockaway_n_points_window']})", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    missing = sum(1 for r in rows if r["inwood_m"] is None or r["rockaway_m"] is None)
    print(f"\nwrote {args.out_csv}  ({len(rows)} dates, {missing} with a missing gauge within {args.window_min:.0f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

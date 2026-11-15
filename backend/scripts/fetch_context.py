"""Fetch what the environment was doing around every overpass, both tracks.

    uv run --project backend python -u backend/scripts/fetch_context.py --out-dir data/processed/jamaica_bay

Run from the repository root. Writes two files:

  hydrographs.json  Observed USGS water level (parameter 62620, ft converted to m NAVD88) from
                    both gauges for +/- HALF_WINDOW_H hours around each acquisition, at the gauge's
                    native sampling. Observed values only: no predicted tides, no interpolation.
  weather.json      NOAA daily summaries for JFK (USW00094789) on each acquisition's local date:
                    AWND, WSF2, PRCP and rainfall over the three days before. These are full-day
                    values, not values at the overpass.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from match_tides import FT_TO_M, SITES, fetch
from tide_checks import f, fetch_noaa

HALF_WINDOW_H = 13  # a little over one semidiurnal cycle each side
TIDES = {"descending_069": "tide_matches_27.csv", "ascending_133": "tide_matches_133A022.csv"}
NY = ZoneInfo("America/New_York")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = [(name, r) for name, fn in TIDES.items() for r in csv.DictReader((args.out_dir / fn).open())]
    local_dates = sorted({datetime.fromisoformat(r["acquired_utc"]).astimezone(NY).date() for _, r in rows})
    noaa = fetch_noaa("USW00094789", (local_dates[0] - timedelta(days=5)).isoformat(), local_dates[-1].isoformat())

    def rain(d: date) -> float:
        return f(noaa.get(d.isoformat(), {}).get("PRCP")) or 0.0

    hydro, weather = {}, {}
    for name, r in rows:
        t = datetime.fromisoformat(r["acquired_utc"])
        acq = f"{name}_{r['date']}"
        if acq in hydro:  # two processing versions of one acquisition share the same overpass
            continue
        entry = {"timestamp_utc": r["acquired_utc"], "units": "m NAVD88", "gauges": {}}
        for gauge, site in SITES.items():
            props = fetch(site, t - timedelta(hours=HALF_WINDOW_H), t + timedelta(hours=HALF_WINDOW_H))
            pts = sorted((round((datetime.fromisoformat(p["time"]) - t).total_seconds() / 60.0, 1),
                          round(float(p["value"]) * FT_TO_M, 4))
                         for p in props if p.get("value") not in (None, ""))
            entry["gauges"][gauge] = {"usgs_site": site, "offset_min": [p[0] for p in pts],
                                      "water_level_m": [p[1] for p in pts]}
        hydro[acq] = entry
        d = t.astimezone(NY).date()
        day = noaa.get(d.isoformat(), {})
        weather[acq] = {"local_date": d.isoformat(), "wind_avg_mps": f(day.get("AWND")),
                        "wind_gust_2min_mps": f(day.get("WSF2")), "precip_mm": rain(d),
                        "precip_3d_prior_mm": round(sum(rain(d - timedelta(days=k)) for k in (1, 2, 3)), 1)}
        print(f"{acq}  inwood n={len(entry['gauges']['inwood']['offset_min'])}  "
              f"rockaway n={len(entry['gauges']['rockaway']['offset_min'])}  wind {weather[acq]['wind_avg_mps']}",
              flush=True)

    (args.out_dir / "hydrographs.json").write_text(json.dumps(
        {"half_window_hours": HALF_WINDOW_H, "source": "USGS continuous water level, parameter 62620, observed",
         "acquisitions": hydro}) + "\n")
    (args.out_dir / "weather.json").write_text(json.dumps(
        {"source": "NOAA daily summaries, JFK (USW00094789)", "resolution": "daily, local date of the overpass",
         "acquisitions": weather}, indent=2) + "\n")
    print(f"\nwrote hydrographs.json and weather.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

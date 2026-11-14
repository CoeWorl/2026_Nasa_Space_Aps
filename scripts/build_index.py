"""Build the artifact index that the frontend reads.

One implementation, two consumers: the local development server
synthesises it per request, and `make index` writes it to a file for
upload to R2. Keeping the shape in a single function is what stops the
local contract and the Cloudflare contract drifting apart.

    make index     # writes data/interim/index.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Not an analysis product; it is the container selftest fixture.
EXCLUDED = {"selftest.tif"}


def build_index(data_root: Path) -> dict[str, Any]:
    """Describe every published raster under `data_root`.

    Fields come from the provenance sidecar that fetch-window writes, so
    the frontend learns the acquisition date, polarization and units
    without parsing granule filenames.
    """
    rasters: list[dict[str, Any]] = []

    for tif in sorted(data_root.glob("*.tif")):
        if tif.name in EXCLUDED:
            continue

        record: dict[str, Any] = {
            "name": tif.name,
            "url": f"/data/{tif.name}",
            "bytes": tif.stat().st_size,
        }

        sidecar = data_root / f"{tif.stem}.provenance.json"
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
            source = meta.get("source", {})
            read = meta.get("read", {})
            record.update(
                {
                    "acquired_at": source.get("acquired_at"),
                    "granule_id": source.get("granule_id"),
                    "track": source.get("track"),
                    "frame": source.get("frame"),
                    "orbit_direction": source.get("orbit_direction"),
                    "maturity": source.get("maturity"),
                    "polarization": read.get("polarization"),
                    "quantity": read.get("quantity"),
                    "units": read.get("units"),
                    "crs": (read.get("grid") or {}).get("crs"),
                    "valid_fraction": read.get("valid_fraction"),
                }
            )

        rasters.append(record)

    rasters.sort(key=lambda r: (r.get("acquired_at") or "", r["name"]))
    return {"schema_version": 1, "rasters": rasters}


def main() -> int:
    data_root = Path(os.environ.get("DATA_ROOT", "/data")) / "interim"
    if not data_root.is_dir():
        print(f"{data_root} is not mounted", file=sys.stderr)
        return 1

    index = build_index(data_root)
    out_path = data_root / "index.json"
    out_path.write_text(json.dumps(index, indent=2) + "\n")

    print(f"wrote {out_path}")
    for raster in index["rasters"]:
        date = (raster.get("acquired_at") or "?")[:10]
        print(f"  {date}  {raster.get('quantity', '?'):<18} {raster['name']}")
    if not index["rasters"]:
        print("  (no rasters - run 'make fetch DATE=...' first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

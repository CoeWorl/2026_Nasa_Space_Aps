"""Add HV backscatter and the GCOV numberOfLooks layer for the granules already in a bay stack.

    HOME=<scratch dir> backend/.venv/bin/python -u backend/scripts/extract_hv.py \
        --stack data/processed/stack_bay_069D068 --track 069 --direction D --frame 068

Reads exactly the product ids recorded in stack_aoi.npz (no new date selection), the same
lon/lat window, Frequency A, through the pipeline's own reader and validity rule. Writes
stack_hv.npz next to the HH stack with the dates in the same order:
    HV_values (linear intensity), HV_valid, looks (float16, NaN where the layer is absent).

authenticate() copies the netrc into $HOME, so point HOME at a scratch directory when
running outside Docker.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from audit_stack import BBOX, COLLECTIONS, TEMPORAL, parse

from groundtruth.geospatial.regions import reproject_bounds, window_from_bounds
from groundtruth.infrastructure.earthdata import (
    authenticate,
    open_granule,
    search_granules,
)
from groundtruth.products.nisar import BASE, GcovReader, full_grid_shape


def _read_one(granule: Any, ur: str, window: tuple[float, float, float, float]) -> dict[str, Any]:
    handle = open_granule(granule)
    with GcovReader(handle, source_uri=ur) as reader:
        tile = reader.read_window(window, "HV", frequency="frequencyA")
        full = reader.grid("frequencyA")
        win = window_from_bounds(full, reproject_bounds(window, "EPSG:4326", full.crs))
        layer = reader._file[f"{BASE}/grids/frequencyA"].get("numberOfLooks")
        if layer is not None and layer.shape == full_grid_shape(full):
            looks = np.asarray(layer[win.row_slice, win.col_slice], dtype=np.float16)
        else:
            looks = np.full(tile.values.shape, np.nan, dtype=np.float16)
    return {"ur": ur, "HV_values": tile.values, "HV_valid": tile.valid, "looks": looks}


def _init_worker() -> None:
    import earthaccess

    if not getattr(earthaccess.login(strategy="netrc", persist=False), "authenticated", False):
        raise RuntimeError("worker could not authenticate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--netrc", type=Path, default=Path("secrets/netrc"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.stack / "stack_aoi.npz")
    wanted = [str(p) for p in z["product_ids"]]
    shape = z["A_valid"].shape[1:]
    window = tuple(json.loads((args.stack / "stack_aoi.json").read_text())["window_lonlat"])

    authenticate(args.netrc)
    found: dict[str, Any] = {}
    for collection in COLLECTIONS:
        for granule in search_granules(collection, BBOX, TEMPORAL, count=500):
            meta = parse(granule)
            key = (args.track, args.direction, args.frame)
            if meta and (meta["track"], meta["direction"], meta["frame"]) == key and meta["ur"] in wanted:
                found[meta["ur"]] = granule
    missing = [u for u in wanted if u not in found]
    if missing:
        raise SystemExit(f"granules not found in CMR: {missing}")
    print(f"granules to read: {len(wanted)}", flush=True)

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_read_one, found[u], u, window) for u in wanted]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["HV_values"].shape != shape:
                raise SystemExit(f"{r['ur']}: HV window {r['HV_values'].shape} differs from HH {shape}")
            results[r["ur"]] = r
            print(f"  [{n:2d}/{len(wanted)}] {r['ur'][-45:]} valid {r['HV_valid'].mean():.4f} "
                  f"looks median {np.nanmedian(r['looks'].astype(np.float32)):.1f}", flush=True)

    out = args.stack / "stack_hv.npz"
    np.savez_compressed(
        out, dates=z["dates"], product_ids=z["product_ids"],
        HV_values=np.stack([results[u]["HV_values"] for u in wanted]),
        HV_valid=np.stack([results[u]["HV_valid"] for u in wanted]),
        looks=np.stack([results[u]["looks"] for u in wanted]),
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

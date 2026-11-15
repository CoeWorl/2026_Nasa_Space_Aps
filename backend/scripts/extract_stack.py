"""Extract AOI window stacks for the usable provisional dates (VALIDATION_PLAN.md, Tests 2-6).

Reads each granule's HH window at Frequency A (10 m) and Frequency B (native
grid), plus the GCOV numberOfLooks layer, through the pipeline's own reader.
Writes one compressed NPZ with every date stacked on the first axis.

Dates are chosen by the Test 0 audit, not by looking at the imagery: the
granules that share the usable processing configuration (CRID P05023,
collection PROVISIONAL_V1, valid fraction above the threshold) are kept.

    uv run python -u scripts/extract_stack.py --audit data/processed/stack_audit_069D068/scene_audit.csv --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from audit_stack import AOI_WINDOW, BBOX, COLLECTIONS, TEMPORAL, parse

from groundtruth.infrastructure.earthdata import (
    authenticate,
    open_granule,
    search_granules,
)
from groundtruth.products.nisar import GcovReader

USABLE_VALID_FRACTION = 0.5  # Test 0: dates below this have no usable AOI data.
CHANNEL = "HH"


def _read_one(granule: Any, meta: dict[str, str], window: tuple[float, float, float, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"date": meta["date"], "ur": meta["ur"]}
    handle = open_granule(granule)
    with GcovReader(handle, source_uri=meta["ur"]) as reader:
        a = reader.read_window(window, CHANNEL, frequency="frequencyA")
        out["A_values"] = a.values
        out["A_valid"] = a.valid
        out["A_transform"] = np.array(a.grid.transform)[:6]
    handle.seek(0)
    with GcovReader(handle, source_uri=meta["ur"]) as reader:
        b = reader.read_window(window, CHANNEL, frequency="frequencyB")
        out["B_values"] = b.values
        out["B_valid"] = b.valid
        out["B_transform"] = np.array(b.grid.transform)[:6]
        out["B_width_m"] = float(abs(b.grid.transform.a))
    return out


def _init_worker(netrc: str) -> None:
    import earthaccess

    if not getattr(earthaccess.login(strategy="netrc", persist=False), "authenticated", False):
        raise RuntimeError("worker could not authenticate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", type=Path, default=None,
                    help="Test 0 audit CSV. If omitted, dates are selected by CRID P05023 in PROVISIONAL_V1.")
    ap.add_argument("--track", default="069")
    ap.add_argument("--direction", default="D")
    ap.add_argument("--frame", default="068")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--netrc", type=Path, default=Path("secrets/netrc"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--window", type=float, nargs=4, default=list(AOI_WINDOW),
        metavar=("LON0", "LAT0", "LON1", "LAT1"),
        help="lon/lat window to extract (default: the Manhattan AOI)",
    )
    args = ap.parse_args()
    window = tuple(args.window)

    authenticate(args.netrc)

    if args.audit is not None:
        audit = list(csv.DictReader(args.audit.open()))
        usable = {
            r["product_id"]
            for r in audit
            if r["crid"] == "P05023"
            and r["collection"] == "NISAR_L2_GCOV_PROVISIONAL_V1"
            and float(r["valid_fraction"]) >= USABLE_VALID_FRACTION
        }
        print(f"usable dates from audit: {len(usable)}", flush=True)
    else:
        usable = None  # selected below by configuration

    candidates = []
    for collection in COLLECTIONS:
        for granule in search_granules(collection, BBOX, TEMPORAL, count=500):
            meta = parse(granule)
            if meta is None:
                continue
            if (meta["track"], meta["direction"], meta["frame"]) != (args.track, args.direction, args.frame):
                continue
            meta["collection"] = collection
            if usable is None:
                # Same configuration rule as the Manhattan stack: provisional collection, CRID P05023.
                if collection != "NISAR_L2_GCOV_PROVISIONAL_V1" or meta["crid"] != "P05023":
                    continue
            elif meta["ur"] not in usable:
                continue
            candidates.append((granule, meta))

    # One acquisition per date: keep the highest processing version (_002 over _001).
    by_date: dict[str, tuple[Any, dict[str, str]]] = {}
    for granule, meta in candidates:
        version = int(meta["ur"][-3:])
        if meta["date"] not in by_date or version > int(by_date[meta["date"]][1]["ur"][-3:]):
            by_date[meta["date"]] = (granule, meta)
    jobs = list(by_date.values())
    print(f"candidate granules {len(candidates)}, one per date {len(jobs)}", flush=True)
    jobs.sort(key=lambda j: j[1]["date"])
    print(f"granules to read: {len(jobs)}", flush=True)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(str(args.netrc),)
    ) as pool:
        futures = {pool.submit(_read_one, g, m, window): m for g, m in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            print(f"  [{n:2d}/{len(jobs)}] {res['date']} A{res['A_values'].shape} "
                  f"B{res['B_values'].shape} valid A {res['A_valid'].mean():.4f} "
                  f"B {res['B_valid'].mean():.4f}", flush=True)

    results.sort(key=lambda r: r["date"])
    a_shapes = {r["A_values"].shape for r in results}
    b_shapes = {r["B_values"].shape for r in results}
    if len(a_shapes) != 1 or len(b_shapes) != 1:
        raise SystemExit(f"Shapes differ across dates: A {a_shapes}, B {b_shapes}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / "stack_aoi.npz"
    np.savez_compressed(
        npz,
        dates=np.array([r["date"] for r in results]),
        product_ids=np.array([r["ur"] for r in results]),
        A_values=np.stack([r["A_values"] for r in results]),
        A_valid=np.stack([r["A_valid"] for r in results]),
        A_transform=np.stack([r["A_transform"] for r in results]),
        B_values=np.stack([r["B_values"] for r in results]),
        B_valid=np.stack([r["B_valid"] for r in results]),
        B_transform=np.stack([r["B_transform"] for r in results]),
    )
    meta_path = args.out_dir / "stack_aoi.json"
    meta_path.write_text(json.dumps({
        "window_lonlat": list(window),
        "channel": CHANNEL,
        "dates": [r["date"] for r in results],
        "A_shape": list(next(iter(a_shapes))),
        "B_shape": list(next(iter(b_shapes))),
        "B_pixel_m": results[0]["B_width_m"],
        "usable_valid_fraction_threshold": USABLE_VALID_FRACTION,
    }, indent=2) + "\n")
    print(f"\nwrote {npz}\nwrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

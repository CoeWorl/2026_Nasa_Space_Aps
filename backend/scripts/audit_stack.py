"""Audit the repeat-observation stack before treating it as one time series.

VALIDATION_PLAN.md, Test 0. For every granule in one acquisition stack
(track, direction, frame, polarization), read the metadata that decides
whether two dates are scientifically interchangeable, and write one row per
date to scene_audit.csv (plus a JSON copy).

    earthaccess search (public CMR)  ->  remote HDF5 range reads  ->  CSV/JSON

Performance properties:
  * One remote open per granule, reused for metadata and the window read.
  * Metadata is read from known paths only; no whole-file tree walk.
  * Granules run in a process pool. h5py serialises calls under one global
    lock, so threads would not overlap network waits; processes do.
  * Each finished row is appended to scene_audit.jsonl immediately, and rows
    already present are skipped, so an interrupted run resumes.
  * Only transient network errors are retried, with bounded backoff.

Reads come from the same GcovReader the pipeline uses, so the valid_fraction
column is exactly the validity rule the change products rely on.

Requires Earthdata credentials (secrets/netrc).

    uv run python -u scripts/audit_stack.py --out-dir DIR [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from groundtruth.geospatial.regions import Bounds
from groundtruth.infrastructure.earthdata import (
    authenticate,
    open_granule,
    search_granules,
)
from groundtruth.products.nisar import GcovReader

BBOX = (-74.30, 40.45, -73.65, 40.95)
TEMPORAL = ("2025-01-01", "2026-12-31")
AOI_WINDOW: Bounds = (-74.03, 40.70, -73.90, 40.88)
# Channel read for the change products. The CMR code "DHDH" is a collection
# filter, not a channel: the granules list HH and HV in frequencyA.
CHANNEL = "HH"
COLLECTIONS = ("NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1")

MAX_ATTEMPTS = 3
BACKOFF_S = 2.0

GRANULE_RE = re.compile(
    r"NISAR_L2_\w\w_(?P<product>\w+?)_(?P<cycle>\d+)_(?P<track>\d+)_"
    r"(?P<direction>[AD])_(?P<frame>\d+)_"
    r"(?P<mode>\w+?)_(?P<pol>\w+?)_A_(?P<date>\d{8})T\d{6}_\d{8}T\d{6}_"
    r"(?P<crid>P\d+)_N_(?P<letter>\w)_J_(?P<version>\d+)"
)

COLUMNS = [
    "date", "product_id", "collection", "maturity_letter", "processing_type",
    "crid", "product_version", "processing_datetime", "product_spec_version",
    "frequency", "polarization", "polarizations_freqA", "freqB_present",
    "polarizations_freqB", "resolution_m", "bandwidth", "valid_fraction",
    "valid_fraction_note", "track", "frame", "direction", "error",
]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        return ",".join(_text(v) for v in value.ravel()) if value.size else ""
    return str(value)


def _is_transient(exc: BaseException) -> bool:
    """Network-level failures that a retry can plausibly fix. Anything else is deterministic."""
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if any(tag in name for tag in ("ClientResponseError", "ClientConnection", "Timeout")):
        return any(code in text for code in ("502", "503", "504", "reset", "timed out"))
    return False


def _shallow_bandwidth(f: h5py.File) -> str:
    """Look for a bandwidth field at the few paths that could hold one.

    Deliberately non-recursive: walking the whole tree cost one remote read per
    object and was the bottleneck of the first run.
    """
    hits: list[str] = []
    for group_path in ("science/LSAR/identification", "science/LSAR/GCOV/grids/frequencyA"):
        group = f.get(group_path)
        if group is None:
            continue
        for key in group:
            item = group[key]
            if "bandwidth" in key.lower() and isinstance(item, h5py.Dataset) and item.size <= 4:
                hits.append(f"{group_path.rsplit('/', 1)[-1]}/{key}={_text(item[()])}")
    return "; ".join(hits) if hits else "not_found_shallow"


def audit_one(granule: Any, meta: dict[str, str]) -> dict[str, Any]:
    """Worker entry point. Runs in a child process, one granule at a time."""
    row: dict[str, Any] = {k: "" for k in COLUMNS}
    row.update(
        product_id=meta["ur"], date=meta["date"], track=meta["track"],
        frame=meta["frame"], direction=meta["direction"],
        collection=meta["collection"], maturity_letter=meta["letter"],
        polarization=CHANNEL, frequency="frequencyA", crid=meta["crid"],
    )
    last: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _read_granule(granule, meta, row)
            row["error"] = ""
            return row
        except Exception as exc:  # noqa: BLE001 - recorded per granule
            last = exc
            if not _is_transient(exc):
                break
            time.sleep(BACKOFF_S * attempt)
    row["error"] = f"{type(last).__name__}: {last}"[:300]
    return row


def _read_granule(granule: Any, meta: dict[str, str], row: dict[str, Any]) -> None:
    handle = open_granule(granule)  # one remote open per granule
    with h5py.File(handle, "r") as f:
        ident = f["science/LSAR/identification"]

        def scalar(name: str) -> str:
            return _text(ident[name][()]) if name in ident else "missing"

        row["product_id"] = scalar("granuleId")
        row["processing_type"] = scalar("processingType")
        row["crid"] = scalar("compositeReleaseId")
        row["product_version"] = scalar("productVersion")
        row["processing_datetime"] = scalar("processingDateTime")
        row["product_spec_version"] = scalar("productSpecificationVersion")
        row["bandwidth"] = _shallow_bandwidth(f)

        grids = f["science/LSAR/GCOV/grids"]
        row["polarizations_freqA"] = _text(grids["frequencyA/listOfPolarizations"][()])
        row["freqB_present"] = "frequencyB" in grids
        if row["freqB_present"] and "listOfPolarizations" in grids["frequencyB"]:
            row["polarizations_freqB"] = _text(grids["frequencyB/listOfPolarizations"][()])
        row["resolution_m"] = float(grids["frequencyA/xCoordinateSpacing"][()])

    # Reuse the same open handle: rewind, then let the pipeline reader use it.
    handle.seek(0)
    with GcovReader(handle, source_uri=row["product_id"]) as reader:
        tile = reader.read_window(AOI_WINDOW, CHANNEL)
    row["valid_fraction"] = round(tile.valid_fraction, 6)
    row["valid_fraction_note"] = "AOI window, pipeline validity rule"


def _init_worker(netrc: str) -> None:
    # The main process already copied the netrc into HOME. Workers must not
    # copy it again: four concurrent copies truncate the file under each other.
    import earthaccess

    if not getattr(earthaccess.login(strategy="netrc", persist=False), "authenticated", False):
        raise RuntimeError("worker could not authenticate from HOME/.netrc")


def parse(granule: Any) -> dict[str, str] | None:
    ur = granule["umm"].get("GranuleUR", "")
    m = GRANULE_RE.match(ur)
    if not m:
        return None
    g = m.groupdict()
    date = g["date"]
    return {
        "ur": ur,
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "track": g["track"], "frame": g["frame"], "direction": g["direction"],
        "pol": g["pol"], "crid": g["crid"], "letter": g["letter"],
        "collection": "",  # filled by caller
    }


def _load_done(jsonl: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                # Error rows are retried on resume, not treated as finished.
                if not row.get("error"):
                    done[row["_ur"]] = row
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--track", default="069")
    ap.add_argument("--direction", default="D")
    ap.add_argument("--frame", default="068")
    ap.add_argument("--pol", default="DHDH", help="CMR polarization code")
    ap.add_argument("--netrc", type=Path, default=Path("secrets/netrc"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    authenticate(args.netrc)

    found: list[tuple[Any, dict[str, str]]] = []
    for collection in COLLECTIONS:
        for granule in search_granules(collection, BBOX, TEMPORAL, count=500):
            meta = parse(granule)
            if meta is None:
                continue
            meta["collection"] = collection
            if (meta["track"], meta["direction"], meta["frame"], meta["pol"]) != (
                args.track, args.direction, args.frame, args.pol,
            ):
                continue
            found.append((granule, meta))
    found.sort(key=lambda item: item[1]["date"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "scene_audit.jsonl"
    done = _load_done(jsonl)
    pending = [(g, m) for g, m in found if m["ur"] not in done]
    print(f"stack {args.track}{args.direction}{args.frame} {args.pol}: "
          f"{len(found)} granule(s), {len(done)} already done, {len(pending)} to read, "
          f"{args.workers} worker(s)", flush=True)

    start = time.time()
    if pending:
        with jsonl.open("a") as sink, ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(args.netrc),),
        ) as pool:
            futures = {pool.submit(audit_one, g, m): m for g, m in pending}
            for n, fut in enumerate(as_completed(futures), 1):
                meta = futures[fut]
                row = fut.result()
                row["_ur"] = meta["ur"]
                sink.write(json.dumps(row, default=str) + "\n")
                sink.flush()
                done[meta["ur"]] = row
                status = "ERROR " + row["error"] if row["error"] else "ok"
                print(f"  [{n:2d}/{len(pending)}] {row['date']} {row['collection']:<28} "
                      f"{row['crid']} {row['processing_type']:<8} {status}  "
                      f"({time.time() - start:.0f}s)", flush=True)

    rows = [done[m["ur"]] for _, m in found if m["ur"] in done]
    csv_path = args.out_dir / "scene_audit.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})
    json_path = args.out_dir / "scene_audit.json"
    json_path.write_text(json.dumps({
        "stack": f"{args.track}{args.direction}{args.frame}_{args.pol}",
        "rows": [{k: v for k, v in r.items() if k != "_ur"} for r in rows],
    }, indent=2, default=str) + "\n")
    print(f"\nwrote {csv_path} ({len(rows)} rows)\nwrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

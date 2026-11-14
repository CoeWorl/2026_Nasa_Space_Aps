"""Open one NISAR granule by HTTP range reads and report its real structure.

Phase 0 exit criterion, per GROUNDTRUTH.md section 6.3: "The implementation
must not guess HDF5 paths." This prints the paths, units, grid geometry and
polarizations actually present in a granule, so the reader is written against
the file rather than against an assumption.

Nothing is downloaded in full. earthaccess.open() returns a file object that
fetches byte ranges on demand, and h5py reads only the chunks it needs.

    make inspect

Requires Earthdata credentials in secrets/netrc.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import earthaccess
import h5py
import numpy as np

BBOX = (-74.30, 40.45, -73.65, 40.95)
SHORT_NAME = "NISAR_L2_GCOV_PROVISIONAL_V1"
STACK = "_069_D_068_"  # track 069, descending, frame 068
TEMPORAL = ("2026-10-01", "2026-10-10")


def authenticate() -> None:
    """Make the mounted compose secret visible to earthaccess as ~/.netrc."""
    src = Path("/run/secrets/earthdata_netrc")
    if not src.is_file():
        raise SystemExit("No Earthdata netrc mounted at /run/secrets/earthdata_netrc")
    if "REPLACE_WITH" in src.read_text(errors="ignore"):
        raise SystemExit(
            "secrets/netrc still holds the placeholder. Put your Earthdata "
            "login in it: https://urs.earthdata.nasa.gov/users/new"
        )
    dst = Path(os.environ.get("HOME", "/tmp")) / ".netrc"
    shutil.copyfile(src, dst)
    dst.chmod(0o600)
    auth = earthaccess.login(strategy="netrc", persist=False)
    if not getattr(auth, "authenticated", False):
        raise SystemExit("Earthdata authentication failed")
    print("authenticated with Earthdata")


def decode(value):
    """HDF5 strings come back as bytes or arrays of bytes."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return [v.decode(errors="replace") for v in value.ravel()]
        if value.size <= 4:
            return value.tolist()
        return f"<array shape={value.shape} dtype={value.dtype}>"
    return value


def show_group_scalars(f: h5py.File, path: str, limit: int = 40) -> None:
    group = f.get(path)
    if group is None:
        print(f"  (no group at {path})")
        return
    shown = 0
    for name, obj in group.items():
        if not isinstance(obj, h5py.Dataset) or obj.size > 4:
            continue
        print(f"  {name:34s} {decode(obj[()])}")
        shown += 1
        if shown >= limit:
            break


def main() -> int:
    authenticate()

    granules = earthaccess.search_data(
        short_name=SHORT_NAME,
        bounding_box=BBOX,
        temporal=TEMPORAL,
        count=50,
    )
    stack = [g for g in granules if STACK in g["umm"]["GranuleUR"]]
    if not stack:
        raise SystemExit(f"No granule matching {STACK} in {TEMPORAL}")

    granule = stack[0]
    print(f"\ngranule: {granule['umm']['GranuleUR']}")

    t0 = time.time()
    handle = earthaccess.open([granule])[0]
    print(f"stream opened in {time.time() - t0:.1f}s ({type(handle).__name__})")

    t0 = time.time()
    with h5py.File(handle, "r") as f:
        print(f"HDF5 header read in {time.time() - t0:.1f}s\n")

        print("=== identification ===")
        show_group_scalars(f, "science/LSAR/identification")

        base = "science/LSAR/GCOV"
        for freq in ("frequencyA", "frequencyB"):
            path = f"{base}/grids/{freq}"
            grid = f.get(path)
            if grid is None:
                continue

            print(f"\n=== {path} ===")
            pols = grid.get("listOfPolarizations")
            if pols is not None:
                print(f"  polarizations: {decode(pols[()])}")

            for name, obj in grid.items():
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
                    continue
                units = decode(obj.attrs.get("units", b""))
                desc = decode(obj.attrs.get("description", b""))[:60]
                print(
                    f"  {name:26s} shape={obj.shape} dtype={obj.dtype} "
                    f"chunks={obj.chunks} units={units!r}"
                )
                if desc:
                    print(f"  {'':26s} {desc}")

            for coord in ("xCoordinates", "yCoordinates"):
                arr = grid.get(coord)
                if arr is None:
                    continue
                values = arr[:]
                spacing = float(values[1] - values[0]) if values.size > 1 else 0.0
                print(
                    f"  {coord:26s} n={values.size} "
                    f"{values.min():,.1f} .. {values.max():,.1f} step {spacing:,.1f}"
                )

            proj = grid.get("projection")
            if proj is not None:
                epsg = proj.attrs.get("epsg_code", proj[()])
                print(f"  projection                 EPSG:{decode(epsg)}")

    print("\nNothing was downloaded in full; only the bytes above were fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

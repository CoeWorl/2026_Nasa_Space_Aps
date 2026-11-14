"""Survey actual NISAR granule availability over the area of interest.

Phase 0 feasibility, per GROUNDTRUTH.md section 17: find out what really
exists before writing any processing against assumed products.

Granule *search* uses NASA's public CMR API and needs no credentials, so
anyone can reproduce this without an Earthdata account.

    make survey

A valid two-date pair must agree on track, direction, frame, polarization
AND processing maturity. The maturity flag varies *within* a single
collection, so it is parsed per granule rather than assumed.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

import earthaccess

# NYC pilot area: roughly Staten Island to the Bronx.
BBOX = (-74.30, 40.45, -73.65, 40.95)
TEMPORAL = ("2025-01-01", "2026-12-31")

# Candidate products for the two analysis paths in GROUNDTRUTH.md section 3.
COLLECTIONS = [
    # Path A, backscatter change.
    "NISAR_L2_GCOV_V1",
    "NISAR_L2_GCOV_PROVISIONAL_V1",
    # Path B, interferometric displacement.
    "NISAR_L2_GUNW_V1",
    "NISAR_L2_GUNW_PROVISIONAL_V1",
]

# NISAR_L2_PR_GCOV_<cycle>_<track>_<D|A>_<frame>_<mode>_<pol>_A_<start>T...
# ..._<processing>_N_<maturity>_J_<version>
GRANULE_RE = re.compile(
    r"NISAR_L2_\w\w_(?P<product>\w+?)_"
    r"(?P<cycle>\d+)_(?P<track>\d+)_(?P<direction>[AD])_(?P<frame>\d+)_"
    r"(?P<mode>\w+?)_(?P<pol>\w+?)_"
    r"(?:A_)?(?P<date>\d{8})T.*?_N_(?P<maturity>\w)_J_"
)


def granule_size_mb(granule) -> float:
    """`size` is an attribute in some earthaccess versions, a method in others."""
    raw = getattr(granule, "size", None)
    value = raw() if callable(raw) else raw
    return float(value or 0.0)


def survey(short_name: str) -> int:
    try:
        granules = earthaccess.search_data(
            short_name=short_name,
            bounding_box=BBOX,
            temporal=TEMPORAL,
            count=500,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        print(f"{short_name}: search failed: {exc}")
        return 0

    print(f"=== {short_name}: {len(granules)} granule(s) over the AOI")
    if not granules:
        print()
        return 0

    # Group by everything a two-date comparison must hold constant.
    stacks: dict[tuple[str, ...], set[tuple[str, float, str]]] = defaultdict(set)
    unparsed = 0
    for granule in granules:
        ur = granule["umm"].get("GranuleUR", "")
        match = GRANULE_RE.match(ur)
        if not match:
            unparsed += 1
            continue
        g = match.groupdict()
        key = (g["track"], g["direction"], g["frame"], g["pol"], g["maturity"])
        stacks[key].add((g["date"], granule_size_mb(granule), ur))

    if unparsed:
        print(f"  ({unparsed} granule name(s) did not match the expected pattern)")

    for key in sorted(stacks, key=lambda k: -len(stacks[k])):
        track, direction, frame, pol, maturity = key
        rows = sorted(stacks[key])
        per_granule = rows[-1][1]
        print(
            f"  track {track} {direction} frame {frame} pol {pol} "
            f"maturity {maturity}: {len(rows)} date(s), ~{per_granule:,.0f} MB each"
        )
        dates = [r[0] for r in rows]
        print(f"    {dates[0]} .. {dates[-1]}")
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            print(
                f"    newest valid pair: {a[0]} + {b[0]} "
                f"= {a[1] + b[1]:,.0f} MB if downloaded whole"
            )
            print(f"      earlier: {a[2]}")
            print(f"      later:   {b[2]}")
    print()
    return len(granules)


def main() -> int:
    print(f"area of interest: {BBOX}")
    print(f"time range:       {TEMPORAL[0]} .. {TEMPORAL[1]}")
    print()

    total = sum(survey(name) for name in COLLECTIONS)

    if total == 0:
        print("NOTHING FOUND. Widen the bbox or dates, or change study area.")
        return 1

    print("Reminder: a pair is only comparable when track, direction, frame,")
    print("polarization and maturity all match. Sizes above are whole granules;")
    print("streaming just the AOI window avoids downloading all of that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

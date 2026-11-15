"""Compute a two-date dB change raster and its QC report (VALIDATION_PLAN.md, Tests 1-2).

    uv run python scripts/validate_pair.py EARLIER.tif LATER.tif OUT_DIR

Reads two linear-intensity GeoTIFFs written by the extraction step, refuses
to proceed unless their grids match, writes change.tif (float32, NaN nodata)
and pair_qc.json, then spot-checks sample pixels by reading them through a
second, independent windowed read.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from groundtruth.processing import log_ratio_db
from groundtruth.statistics import robust_summary


def _is_url(ref: str) -> bool:
    return ref.startswith(("http://", "https://"))


def _name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _date_from_name(ref: str) -> str | None:
    """Acquisition date from a GCOV granule name, e.g. ...20261004T235223... -> 2026-10-04."""
    for token in _name(ref).split("_"):
        if len(token) >= 8 and token[:8].isdigit() and "T" in token:
            try:
                d = date(int(token[0:4]), int(token[4:6]), int(token[6:8]))
                return d.isoformat()
            except ValueError:
                return None
    return None


def _open(ref: str):
    # GDAL reads http(s) through /vsicurl/, which issues HTTP Range requests.
    # That is how the Worker's range behaviour gets exercised end to end.
    src = rasterio.open(f"/vsicurl/{ref}" if _is_url(ref) else ref)
    quantity = src.tags().get("quantity", "unknown")
    if quantity != "linear_intensity":
        src.close()
        raise SystemExit(
            f"{_name(ref)}: quantity tag is {quantity!r}, expected 'linear_intensity'. "
            "Refusing to compute a dB difference from non-intensity data."
        )
    return src


def _grid_signature(src) -> dict:
    return {
        "crs": src.crs.to_string() if src.crs else None,
        "transform": list(src.transform)[:6],
        "width": src.width,
        "height": src.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("earlier", help="local path or http(s) URL")
    parser.add_argument("later", help="local path or http(s) URL")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--samples", type=int, default=25, help="pixels to spot-check")
    args = parser.parse_args()

    with _open(args.earlier) as a, _open(args.later) as b:
        sig_a, sig_b = _grid_signature(a), _grid_signature(b)
        grid_ok = sig_a == sig_b
        if not grid_ok:
            raise SystemExit(
                "Grids differ; refusing to difference unaligned rasters.\n"
                f"earlier: {sig_a}\nlater:   {sig_b}"
            )

        e = a.read(1)
        l = b.read(1)
        nodata_a, nodata_b = a.nodata, b.nodata
        change, valid = log_ratio_db(e, l)

        profile = a.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=float("nan"), compress="DEFLATE",
            tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        change_path = args.out_dir / "change.tif"
        with rasterio.open(change_path, "w", **profile) as dst:
            dst.write(change, 1)
            dst.update_tags(
                measurement="gamma0_log_ratio",
                units="dB",
                sign="later minus earlier",
                earlier=_date_from_name(args.earlier) or "",
                later=_date_from_name(args.later) or "",
            )

        # Spot checks: read single pixels through a separate window read and
        # recompute the formula by hand in float64.
        rng = np.random.default_rng(20261016)
        valid_rc = np.argwhere(valid)
        picks = valid_rc[rng.choice(len(valid_rc), size=min(args.samples, len(valid_rc)), replace=False)]
        max_err = 0.0
        for r, c in picks:
            ie = float(a.read(1, window=Window(c, r, 1, 1))[0, 0])
            il = float(b.read(1, window=Window(c, r, 1, 1))[0, 0])
            expected = 10.0 * np.log10(il) - 10.0 * np.log10(ie)
            max_err = max(max_err, abs(expected - float(change[r, c])))

        width_m = abs(a.transform.a)
        height_m = abs(a.transform.e)
        pixel_m = (width_m, height_m)
        crs = a.crs.to_string()

    summary = robust_summary(change[valid]) if valid.any() else None
    total = valid.size
    report = {
        "earlier": _date_from_name(args.earlier),
        "later": _date_from_name(args.later),
        "measurement": "gamma0_log_ratio",
        "units": "dB",
        "sign_convention": "later minus earlier",
        "spatial_grid_compatible": grid_ok,
        "processing_compatible": None,
        "crs": crs,
        "pixel_size_m": pixel_m,
        "shape": [int(e.shape[0]), int(e.shape[1])],
        "nodata_earlier": nodata_a,
        "nodata_later": nodata_b,
        "valid_fraction": float(valid.sum()) / float(total),
        "valid_fraction_earlier": float(np.isfinite(e).sum()) / float(total),
        "valid_fraction_later": float(np.isfinite(l).sum()) / float(total),
        "median_change_db": summary.median if summary else None,
        "robust_scale_db": summary.robust_scale if summary else None,
        "mad_db": summary.mad if summary else None,
        "p05_db": summary.p05 if summary else None,
        "p95_db": summary.p95 if summary else None,
        "control_false_alarm_fraction": None,
        "spot_check": {
            "samples": len(picks),
            "max_abs_error_db": max_err,
        },
        "inputs": {
            "earlier": _name(args.earlier),
            "later": _name(args.later),
            "read_via": "http-range" if _is_url(args.earlier) else "local-file",
        },
    }
    qc_path = args.out_dir / "pair_qc.json"
    qc_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {change_path}\nwrote {qc_path}")


if __name__ == "__main__":
    main()

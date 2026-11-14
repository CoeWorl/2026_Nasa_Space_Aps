"""GROUNDTRUTH command line interface.

This phase covers environment and data plumbing only:

    groundtruth info        resolved settings and mount status
    groundtruth check-data  what is visible under the data root
    groundtruth selftest    prove the mount is writable and the stack works

`analyze-pair` is deliberately absent until the scientific kernel exists.
A command that pretends to analyse data it cannot read is worse than no
command at all.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from pathlib import Path

import typer

from groundtruth import __version__
from groundtruth.config import Settings, load_settings

app = typer.Typer(
    add_completion=False,
    help="NISAR surface-change analysis (GROUNDTRUTH).",
)

EXPECTED_SUBDIRS = (
    Path("raw") / "nisar",
    Path("raw") / "nyc",
    Path("interim"),
    Path("processed"),
)


# fsspec logs one line per HTTP range read at INFO, which buries our own
# output under hundreds of cache messages during a streamed window read.
NOISY_LOGGERS = ("fsspec", "fsspec.caching", "s3fs", "botocore", "urllib3", "h5py")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TiB"


def _mark(ok: bool) -> str:
    return "OK     " if ok else "MISSING"


@app.command()
def info() -> None:
    """Print resolved settings and whether the expected mounts are present."""
    settings = load_settings()
    _configure_logging(settings.log_level)

    typer.echo(f"groundtruth {__version__}")
    typer.echo(f"python     {sys.version.split()[0]} on {platform.machine()}")
    typer.echo("")
    typer.echo("settings")
    typer.echo(f"  app_env                 {settings.app_env}")
    typer.echo(f"  log_level               {settings.log_level}")
    typer.echo(f"  data_root               {settings.data_root}")
    typer.echo(f"  config_root             {settings.config_root}")
    typer.echo(f"  processing_tile_size    {settings.processing_tile_size}")
    typer.echo(f"  processing_max_workers  {settings.processing_max_workers}")
    typer.echo(f"  worker_api_url          {settings.worker_api_url or '(not set)'}")
    typer.echo(
        "  artifact upload         "
        + (
            "configured"
            if settings.upload_is_configured()
            else "not set (local-only output)"
        )
    )
    typer.echo("")

    typer.echo("mounts")
    data_ok = settings.data_root.is_dir()
    config_ok = settings.config_root.is_dir()
    typer.echo(f"  [{_mark(data_ok)}] {settings.data_root}   (data, read-write)")
    typer.echo(f"  [{_mark(config_ok)}] {settings.config_root}  (configs, read-only)")

    for sub in EXPECTED_SUBDIRS:
        path = settings.data_root / sub
        typer.echo(f"  [{_mark(path.is_dir())}] {path}")

    netrc = settings.earthdata_netrc_path
    if netrc.is_file():
        placeholder = "REPLACE_WITH_YOUR_EARTHDATA" in netrc.read_text(
            errors="ignore"
        )
        state = "placeholder (downloads will fail)" if placeholder else "present"
    else:
        state = "not mounted (downloads will fail)"
    typer.echo(f"  [       ] {netrc}  earthdata netrc: {state}")

    if not (data_ok and config_ok):
        typer.echo("")
        typer.echo(
            "One or more mounts are missing. Run through docker compose, "
            "not bare docker run.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command(name="check-data")
def check_data(
    max_entries: int = typer.Option(
        50, "--max-entries", help="Maximum files to list."
    ),
) -> None:
    """List what is visible under the data root, with sizes."""
    settings = load_settings()
    _configure_logging(settings.log_level)

    root = settings.data_root
    if not root.is_dir():
        typer.echo(f"data root {root} is not mounted", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"data root: {root}")
    total_files = 0
    total_bytes = 0
    shown = 0

    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == ".gitkeep":
            continue
        size = path.stat().st_size
        total_files += 1
        total_bytes += size
        if shown < max_entries:
            typer.echo(f"  {_human_bytes(size):>12}  {path.relative_to(root)}")
            shown += 1

    if total_files > shown:
        typer.echo(f"  ... and {total_files - shown} more")

    typer.echo("")
    typer.echo(f"{total_files} file(s), {_human_bytes(total_bytes)} total")

    if total_files == 0:
        typer.echo("")
        typer.echo(
            "No data yet. Put NISAR granules in ./data/raw/nisar/ on the host; "
            "they appear here immediately (bind mount, no rebuild needed)."
        )


def _run_selftest(settings: Settings) -> list[Path]:
    """Write and read back an HDF5 file and a GeoTIFF. Returns written paths."""
    import h5py
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from scipy import ndimage

    out_dir = settings.interim_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=20260301)
    intensity = rng.gamma(shape=4.0, scale=0.25, size=(256, 256)).astype(np.float32)

    h5_path = out_dir / "selftest.h5"
    with h5py.File(h5_path, "w") as handle:
        dataset = handle.create_dataset(
            "intensity", data=intensity, compression="gzip"
        )
        dataset.attrs["units"] = "linear_intensity"
    with h5py.File(h5_path, "r") as handle:
        read_back = handle["intensity"][:]
    if not np.allclose(read_back, intensity):
        raise RuntimeError("HDF5 round trip did not reproduce the input array")
    typer.echo(f"  [OK] h5py     wrote and read {h5_path} {read_back.shape}")

    # Exercise the numerical stack the way the real kernel will.
    decibels = (10.0 * np.log10(intensity)).astype(np.float32)
    smoothed = ndimage.uniform_filter(decibels, size=3)
    _labels, count = ndimage.label(smoothed > float(np.percentile(smoothed, 95)))
    typer.echo(
        f"  [OK] numpy     dB range "
        f"{decibels.min():.2f} .. {decibels.max():.2f}"
    )
    typer.echo(f"  [OK] scipy     {count} connected region(s) above the 95th pct")

    tif_path = out_dir / "selftest.tif"
    transform = from_origin(583_000.0, 4_510_000.0, 20.0, 20.0)
    crs = CRS.from_epsg(32618)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=decibels.shape[0],
        width=decibels.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(decibels, 1)
        dst.update_tags(units="dB", measurement="selftest_synthetic")

    with rasterio.open(tif_path) as src:
        if src.crs != crs:
            raise RuntimeError(f"CRS round trip failed: {src.crs} != {crs}")
        if src.transform != transform:
            raise RuntimeError("Geotransform round trip failed")
        if not np.allclose(src.read(1), decibels, equal_nan=True):
            raise RuntimeError("GeoTIFF round trip did not reproduce the raster")
        typer.echo(
            f"  [OK] rasterio  wrote and read {tif_path} "
            f"{src.width}x{src.height} {src.crs}"
        )

    return [h5_path, tif_path]


@app.command()
def selftest() -> None:
    """Prove the data mount is writable and the scientific stack works.

    Writes two files into <data_root>/interim and reads them back. Both
    must also appear on the host, which is what makes the bind mount
    verifiable from outside the container.
    """
    settings = load_settings()
    _configure_logging(settings.log_level)

    typer.echo(f"groundtruth {__version__} selftest")
    typer.echo(f"data root: {settings.data_root}")

    if not settings.data_root.is_dir():
        typer.echo(f"data root {settings.data_root} is not mounted", err=True)
        raise typer.Exit(code=1)

    try:
        written = _run_selftest(settings)
    except Exception as exc:
        typer.echo("")
        typer.echo(f"SELFTEST FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    typer.echo("SELFTEST PASSED")
    for path in written:
        typer.echo(f"  wrote {path} ({_human_bytes(path.stat().st_size)})")
    typer.echo("")
    typer.echo("These files should now exist on the host under ./data/interim/.")


@app.command(name="fetch-window")
def fetch_window(
    date: str = typer.Option(..., "--date", help="Acquisition date, YYYY-MM-DD."),
    bbox: str = typer.Option(
        "-74.03,40.70,-73.90,40.88",
        "--bbox",
        help="Area of interest as west,south,east,north in lon/lat.",
    ),
    polarization: str = typer.Option("HH", "--polarization"),
    collection: str = typer.Option(
        "NISAR_L2_GCOV_PROVISIONAL_V1", "--collection"
    ),
    stack: str = typer.Option(
        "_069_D_068_",
        "--stack",
        help="Substring selecting one track/direction/frame stack.",
    ),
    frequency: str = typer.Option("frequencyA", "--frequency"),
) -> None:
    """Stream one granule's area of interest to disk. No full download.

    Only the HDF5 chunks intersecting the window are fetched over HTTP, so
    a multi-gigabyte granule costs a few hundred megabytes of transfer and
    leaves a small GeoTIFF behind.
    """
    import json
    from datetime import UTC, datetime

    from groundtruth.geospatial.export import write_cog
    from groundtruth.infrastructure.earthdata import (
        EarthdataError,
        authenticate,
        open_granule,
        search_granules,
    )
    from groundtruth.products.nisar import GcovReader, NisarProductError

    settings = load_settings()
    _configure_logging(settings.log_level)

    try:
        west, south, east, north = (float(v) for v in bbox.split(","))
    except ValueError as exc:
        typer.echo(f"Could not parse --bbox {bbox!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    bounds = (west, south, east, north)

    try:
        authenticate(settings.earthdata_netrc_path)
    except EarthdataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc
    typer.echo("authenticated with Earthdata")

    granules = [
        g
        for g in search_granules(collection, bounds, (date, date), count=100)
        if stack in g["umm"].get("GranuleUR", "")
    ]
    if not granules:
        typer.echo(
            f"No {collection} granule on {date} matching stack {stack!r} "
            f"over {bounds}. Run 'make survey' to see what exists.",
            err=True,
        )
        raise typer.Exit(code=4)
    if len(granules) > 1:
        typer.echo(f"note: {len(granules)} granules matched; using the first")

    granule = granules[0]
    granule_id = granule["umm"]["GranuleUR"]
    typer.echo(f"granule: {granule_id}")

    started = time.time()
    handle = open_granule(granule)

    try:
        with GcovReader(handle, source_uri=granule_id) as reader:
            meta = reader.metadata(polarization=polarization)
            typer.echo(
                f"  track {meta.track} frame {meta.frame} "
                f"{meta.orbit_direction} maturity {meta.maturity}"
            )
            typer.echo(f"  acquired {meta.acquired_at.isoformat()}")
            typer.echo(f"  polarizations available: {reader.polarizations(frequency)}")

            full = reader.grid(frequency)
            typer.echo(f"  full grid {full.width} x {full.height} in {full.crs}")

            tile = reader.read_window(
                bounds, polarization, frequency=frequency
            )
    except NisarProductError as exc:
        typer.echo(f"product error: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    elapsed = time.time() - started
    typer.echo(
        f"  window {tile.grid.width} x {tile.grid.height} px, "
        f"{tile.valid_fraction:.1%} valid, read in {elapsed:.1f}s"
    )

    out_dir = settings.interim_dir
    stem = f"{granule_id}_{frequency}_{polarization}"
    raster_path = write_cog(
        tile,
        out_dir / f"{stem}.tif",
        tags={
            "granule_id": granule_id,
            "polarization": polarization,
            "units": "linear_intensity",
        },
    )

    provenance = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "software_version": __version__,
        "source": {
            "collection": collection,
            "granule_id": meta.product_id,
            "source_uri": meta.source_uri,
            "acquired_at": meta.acquired_at.isoformat(),
            "product_type": meta.product_type,
            "track": meta.track,
            "frame": meta.frame,
            "orbit_direction": meta.orbit_direction,
            "maturity": meta.maturity,
        },
        "read": {
            "frequency": frequency,
            "variable": f"{polarization}{polarization}",
            "polarization": polarization,
            "quantity": str(tile.quantity),
            "units": "linear_intensity (dimensionless covariance diagonal)",
            "requested_bounds_lonlat": list(bounds),
            "grid": {
                "crs": tile.grid.crs,
                "width": tile.grid.width,
                "height": tile.grid.height,
                "transform": list(tile.grid.transform)[:6],
                "bounds": list(tile.grid.bounds),
            },
            "valid_fraction": tile.valid_fraction,
            "validity_rule": (
                "finite AND > 0 AND mask != 255 AND inputDataExceptionMask == 0"
            ),
        },
    }
    provenance_path = out_dir / f"{stem}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    typer.echo("")
    typer.echo(f"wrote {raster_path} ({_human_bytes(raster_path.stat().st_size)})")
    typer.echo(f"wrote {provenance_path}")


if __name__ == "__main__":
    app()

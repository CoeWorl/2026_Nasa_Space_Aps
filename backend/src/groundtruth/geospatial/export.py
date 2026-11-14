"""Write georeferenced rasters. The only module that knows about GeoTIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from groundtruth.models import RasterTile


def write_cog(
    tile: RasterTile,
    path: Path,
    *,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a tiled, compressed, overviewed GeoTIFF.

    Invalid pixels are written as NaN and declared as the nodata value, so
    QGIS and ArcGIS Pro both leave them out of statistics and rendering
    rather than treating them as a measurement of zero.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.where(tile.valid, tile.values, np.nan).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": tile.grid.height,
        "width": tile.grid.width,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_string(tile.grid.crs),
        "transform": tile.grid.transform,
        "nodata": float("nan"),
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.update_tags(
            quantity=str(tile.quantity),
            valid_fraction=f"{tile.valid_fraction:.6f}",
            **(tags or {}),
        )
        # Overviews keep the browser and desktop GIS responsive on a
        # raster this size.
        factors = [f for f in (2, 4, 8, 16) if min(tile.grid.width, tile.grid.height) // f >= 256]
        if factors:
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    return path

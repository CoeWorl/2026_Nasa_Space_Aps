"""Area-of-interest geometry: pure functions, no I/O.

Everything here is testable without a network, a granule, or a container.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from numpy.typing import NDArray

from groundtruth.models import GridSpec, PixelWindow

# Geographic bounds are always (west, south, east, north).
Bounds = tuple[float, float, float, float]


def grid_from_coordinates(
    x_coords: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    crs: str,
) -> GridSpec:
    """Build a GridSpec from NISAR's pixel-CENTRE coordinate arrays.

    NISAR stores one coordinate per pixel centre. A geotransform describes
    pixel EDGES, so the origin is shifted by half a pixel. Getting this
    wrong offsets every output raster by 5 m, which is invisible on a map
    and fatal to a change comparison.
    """
    if x_coords.size < 2 or y_coords.size < 2:
        raise ValueError("Need at least two coordinates per axis to infer spacing")

    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    if dx == 0 or dy == 0:
        raise ValueError("Coordinate spacing must be non-zero")

    west = float(x_coords[0]) - dx / 2.0
    north = float(y_coords[0]) - dy / 2.0

    return GridSpec(
        width=int(x_coords.size),
        height=int(y_coords.size),
        crs=crs,
        transform=Affine.translation(west, north) @ Affine.scale(dx, dy),
    )


def window_from_bounds(grid: GridSpec, bounds: Bounds) -> PixelWindow:
    """Pixel window covering `bounds`, clipped to the grid.

    Raises if the area of interest does not intersect the grid, rather than
    silently returning an empty read.
    """
    west, south, east, north = bounds
    if west >= east or south >= north:
        raise ValueError(f"Degenerate bounds: {bounds}")

    inverse = ~grid.transform
    corners = [inverse @ (x, y) for x in (west, east) for y in (south, north)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]

    col_start = max(0, int(np.floor(min(cols))))
    row_start = max(0, int(np.floor(min(rows))))
    col_stop = min(grid.width, int(np.ceil(max(cols))))
    row_stop = min(grid.height, int(np.ceil(max(rows))))

    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError(
            f"Area of interest {bounds} does not intersect the granule "
            f"{grid.bounds} (CRS {grid.crs})"
        )

    return PixelWindow(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def subgrid(grid: GridSpec, window: PixelWindow) -> GridSpec:
    """The GridSpec describing just `window` of `grid`."""
    if window.col_off + window.width > grid.width:
        raise ValueError("Window exceeds grid width")
    if window.row_off + window.height > grid.height:
        raise ValueError("Window exceeds grid height")

    return GridSpec(
        width=window.width,
        height=window.height,
        crs=grid.crs,
        transform=grid.transform
        @ Affine.translation(float(window.col_off), float(window.row_off)),
    )


def reproject_bounds(bounds: Bounds, src_crs: str, dst_crs: str) -> Bounds:
    """Transform geographic bounds into the granule's CRS.

    Densified along the edges: a UTM box is not a lat/lon box, and taking
    only the four corners understates the extent.
    """
    from pyproj import Transformer

    if src_crs == dst_crs:
        return bounds

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    west, south, east, north = bounds

    steps = 21
    xs = np.linspace(west, east, steps)
    ys = np.linspace(south, north, steps)
    edge_x = np.concatenate([xs, xs, np.full(steps, west), np.full(steps, east)])
    edge_y = np.concatenate([np.full(steps, south), np.full(steps, north), ys, ys])

    out_x, out_y = transformer.transform(edge_x, edge_y)
    return (
        float(np.min(out_x)),
        float(np.min(out_y)),
        float(np.max(out_x)),
        float(np.max(out_y)),
    )

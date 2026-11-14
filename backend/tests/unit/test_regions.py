"""Area-of-interest geometry.

These are the calculations that silently corrupt a change comparison when
they are half a pixel wrong, so they are tested against hand-computed
numbers taken from a real GCOV granule's coordinate arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from groundtruth.geospatial.regions import (
    grid_from_coordinates,
    reproject_bounds,
    subgrid,
    window_from_bounds,
)
from groundtruth.models import GridSpec, PixelWindow

# Real values from NISAR_L2_PR_GCOV_032_069_D_068_... frequencyA:
# x centres start 438,485.0 step +10; y centres start 4,664,155.0 step -10.
X0, Y0, STEP = 438_485.0, 4_664_155.0, 10.0


def gcov_like_grid(width: int = 100, height: int = 80) -> GridSpec:
    x = X0 + STEP * np.arange(width, dtype=np.float64)
    y = Y0 - STEP * np.arange(height, dtype=np.float64)
    return grid_from_coordinates(x, y, "EPSG:32618")


def test_origin_is_shifted_by_half_a_pixel_from_centres() -> None:
    grid = gcov_like_grid()
    west, north = grid.transform @ (0, 0)
    # Pixel CENTRE 438,485 means the pixel EDGE is at 438,480.
    assert west == pytest.approx(X0 - STEP / 2)
    assert north == pytest.approx(Y0 + STEP / 2)


def test_pixel_centre_round_trips_through_the_transform() -> None:
    grid = gcov_like_grid()
    x, y = grid.transform @ (0.5, 0.5)
    assert (x, y) == pytest.approx((X0, Y0))


def test_grid_size_follows_the_coordinate_arrays() -> None:
    grid = gcov_like_grid(width=35352, height=34848)
    assert (grid.width, grid.height) == (35352, 34848)


def test_descending_y_axis_gives_negative_pixel_height() -> None:
    grid = gcov_like_grid()
    assert grid.transform.e == pytest.approx(-STEP)


def test_bounds_are_ordered_regardless_of_axis_direction() -> None:
    grid = gcov_like_grid()
    west, south, east, north = grid.bounds
    assert west < east
    assert south < north


def test_single_coordinate_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        grid_from_coordinates(np.array([1.0]), np.array([1.0, 2.0]), "EPSG:32618")


# -- windows -----------------------------------------------------------


def test_window_covers_the_requested_bounds() -> None:
    grid = gcov_like_grid(width=100, height=80)
    # Pixels 10..19 across, 5..14 down, in projected coordinates.
    west = X0 + 10 * STEP
    north = Y0 - 5 * STEP
    window = window_from_bounds(grid, (west, north - 10 * STEP, west + 10 * STEP, north))

    assert window.col_off <= 10
    assert window.row_off <= 5
    assert window.col_off + window.width >= 20
    assert window.row_off + window.height >= 15


def test_window_is_clipped_to_the_grid() -> None:
    grid = gcov_like_grid(width=100, height=80)
    huge = (X0 - 1e6, Y0 - 1e6, X0 + 1e6, Y0 + 1e6)
    window = window_from_bounds(grid, huge)

    assert window == PixelWindow(col_off=0, row_off=0, width=100, height=80)


def test_disjoint_area_of_interest_raises_rather_than_returning_empty() -> None:
    grid = gcov_like_grid()
    far_away = (X0 + 1e6, Y0 + 1e6, X0 + 1e6 + 100, Y0 + 1e6 + 100)
    with pytest.raises(ValueError, match="does not intersect"):
        window_from_bounds(grid, far_away)


def test_degenerate_bounds_are_rejected() -> None:
    grid = gcov_like_grid()
    with pytest.raises(ValueError, match="Degenerate"):
        window_from_bounds(grid, (10.0, 10.0, 10.0, 20.0))


# -- subgrid -----------------------------------------------------------


def test_subgrid_keeps_pixel_alignment_with_the_parent() -> None:
    grid = gcov_like_grid(width=100, height=80)
    window = PixelWindow(col_off=10, row_off=5, width=20, height=15)
    sub = subgrid(grid, window)

    assert (sub.width, sub.height) == (20, 15)
    assert sub.crs == grid.crs
    # The sub-grid's origin is the parent's pixel (10, 5) corner.
    assert sub.transform @ (0, 0) == pytest.approx(grid.transform @ (10, 5))
    # And a shared pixel maps to the same world coordinate in both.
    assert sub.transform @ (3, 4) == pytest.approx(grid.transform @ (13, 9))


def test_subgrid_rejects_a_window_outside_the_parent() -> None:
    grid = gcov_like_grid(width=10, height=10)
    with pytest.raises(ValueError, match="exceeds grid"):
        subgrid(grid, PixelWindow(col_off=5, row_off=0, width=10, height=5))


# -- reprojection ------------------------------------------------------


def test_reproject_bounds_is_identity_for_the_same_crs() -> None:
    bounds = (-74.03, 40.70, -73.90, 40.88)
    assert reproject_bounds(bounds, "EPSG:4326", "EPSG:4326") == bounds


def test_manhattan_reprojects_into_the_granule_extent() -> None:
    # The granule covers x 438,480..791,995 and y 4,315,685..4,664,160.
    west, south, east, north = reproject_bounds(
        (-74.03, 40.70, -73.90, 40.88), "EPSG:4326", "EPSG:32618"
    )
    assert 560_000 < west < 590_000
    assert 4_500_000 < south < 4_530_000
    assert west < east and south < north


def test_reprojected_box_is_densified_not_just_corners() -> None:
    """A lat/lon box is not a rectangle in UTM; corners alone understate it."""
    bounds = (-74.5, 40.0, -73.0, 41.0)
    densified = reproject_bounds(bounds, "EPSG:4326", "EPSG:32618")

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    corner_x, corner_y = transformer.transform(
        [bounds[0], bounds[0], bounds[2], bounds[2]],
        [bounds[1], bounds[3], bounds[1], bounds[3]],
    )
    corners_only = (min(corner_x), min(corner_y), max(corner_x), max(corner_y))

    # Densified bounds must contain the corner-only bounds.
    assert densified[0] <= corners_only[0]
    assert densified[1] <= corners_only[1]
    assert densified[2] >= corners_only[2]
    assert densified[3] >= corners_only[3]


# -- grid invariants ---------------------------------------------------


def test_empty_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GridSpec(width=0, height=10, crs="EPSG:32618", transform=Affine.identity())


def test_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PixelWindow(col_off=0, row_off=0, width=0, height=5)

"""Domain type invariants.

RasterTile is the contract every downstream calculation relies on: if a
pixel is marked valid, it is finite and its grid matches its shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from groundtruth.models import GridSpec, Quantity, RasterTile


def grid(width: int = 4, height: int = 3) -> GridSpec:
    return GridSpec(
        width=width,
        height=height,
        crs="EPSG:32618",
        transform=Affine.translation(500_000.0, 4_500_000.0) @ Affine.scale(10, -10),
    )


def tile(values: np.ndarray, valid: np.ndarray) -> RasterTile:
    return RasterTile(
        values=values,
        valid=valid,
        grid=grid(values.shape[1], values.shape[0]),
        quantity=Quantity.LINEAR_INTENSITY,
    )


def test_accepts_a_consistent_tile() -> None:
    values = np.ones((3, 4), dtype=np.float32)
    valid = np.ones((3, 4), dtype=bool)
    assert tile(values, valid).valid_fraction == 1.0


def test_nan_is_allowed_where_the_mask_says_invalid() -> None:
    values = np.full((3, 4), np.nan, dtype=np.float32)
    values[0, 0] = 1.0
    valid = np.zeros((3, 4), dtype=bool)
    valid[0, 0] = True

    result = tile(values, valid)
    assert result.valid_fraction == pytest.approx(1 / 12)


def test_nan_marked_valid_is_rejected() -> None:
    values = np.full((3, 4), np.nan, dtype=np.float32)
    valid = np.ones((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        tile(values, valid)


def test_infinity_marked_valid_is_rejected() -> None:
    values = np.ones((3, 4), dtype=np.float32)
    values[1, 1] = np.inf
    valid = np.ones((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        tile(values, valid)


def test_mask_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Mask shape"):
        tile(np.ones((3, 4), dtype=np.float32), np.ones((3, 5), dtype=bool))


def test_grid_shape_mismatch_is_rejected() -> None:
    values = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match grid"):
        RasterTile(
            values=values,
            valid=np.ones((3, 4), dtype=bool),
            grid=grid(width=9, height=9),
            quantity=Quantity.LINEAR_INTENSITY,
        )


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2D"):
        RasterTile(
            values=np.ones(4, dtype=np.float32),
            valid=np.ones(4, dtype=bool),
            grid=grid(),
            quantity=Quantity.LINEAR_INTENSITY,
        )


def test_valid_fraction_counts_only_valid_pixels() -> None:
    values = np.ones((2, 2), dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    assert tile(values, valid).valid_fraction == pytest.approx(0.5)

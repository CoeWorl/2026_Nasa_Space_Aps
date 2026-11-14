"""Core domain types.

Arrays for large numerical data, immutable objects for small metadata.
These types describe *what a raster is*; they deliberately do not decide
whether two scenes are scientifically comparable. That check belongs to
the workflow (GROUNDTRUTH.md section 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
from affine import Affine
from numpy.typing import NDArray


class Quantity(StrEnum):
    LINEAR_INTENSITY = "linear_intensity"
    BACKSCATTER_DB = "backscatter_db"
    LOS_DISPLACEMENT_M = "los_displacement_m"


@dataclass(frozen=True)
class GridSpec:
    """A regular grid: size, CRS and the affine transform to world coordinates."""

    width: int
    height: int
    crs: str
    transform: Affine

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Grid must be non-empty, got {self.width}x{self.height}")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in CRS units."""
        west, north = self.transform @ (0, 0)
        east, south = self.transform @ (self.width, self.height)
        return (min(west, east), min(south, north), max(west, east), max(south, north))


@dataclass(frozen=True)
class PixelWindow:
    """A rectangular read window in pixel coordinates."""

    col_off: int
    row_off: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.col_off < 0 or self.row_off < 0:
            raise ValueError("Window offsets must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Window must be non-empty, got {self.width}x{self.height}. "
                "An area of interest outside the granule produces this."
            )

    @property
    def row_slice(self) -> slice:
        return slice(self.row_off, self.row_off + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_off, self.col_off + self.width)


@dataclass(frozen=True)
class SceneMetadata:
    """Acquisition facts that compatibility checks are made of.

    Every field here exists because GROUNDTRUTH.md section 3.3 requires it
    to be recorded and checked, not because it was convenient to read.
    """

    product_id: str
    acquired_at: datetime
    product_type: str
    polarization: str | None
    track: str | None
    frame: str | None
    orbit_direction: str | None
    maturity: str
    crs: str
    source_uri: str


@dataclass(frozen=True)
class RasterTile:
    """A 2D measurement array with its validity mask and spatial reference.

    Frozen does not make the NumPy arrays immutable. Treat them as
    read-only by convention after construction.
    """

    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    grid: GridSpec
    quantity: Quantity

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Expected a 2D raster")
        if self.values.shape != self.valid.shape:
            raise ValueError(
                f"Mask shape {self.valid.shape} does not match "
                f"values shape {self.values.shape}"
            )
        expected = (self.grid.height, self.grid.width)
        if self.values.shape != expected:
            raise ValueError(
                f"Raster shape {self.values.shape} does not match grid {expected}"
            )
        if not np.isfinite(self.values[self.valid]).all():
            raise ValueError("Valid pixels must be finite")

    @property
    def valid_fraction(self) -> float:
        if self.valid.size == 0:
            return 0.0
        return float(self.valid.sum()) / float(self.valid.size)

"""Reader for NISAR L2 GCOV products.

One concrete reader for one product family, as GROUNDTRUTH.md section 6.4
prescribes. No plugin registry until a second product family proves it
needs one.

Every path, unit and fill value below was read out of an actual granule
with scripts/inspect_granule.py, not taken from a specification summary.
Product version this was verified against: GCOV 1.0.11, spec 1.5.0.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np

from groundtruth.geospatial.regions import (
    Bounds,
    grid_from_coordinates,
    reproject_bounds,
    subgrid,
    window_from_bounds,
)
from groundtruth.models import GridSpec, Quantity, RasterTile, SceneMetadata

BASE = "science/LSAR/GCOV"
IDENTIFICATION = "science/LSAR/identification"

# Verified against the granule: 255 marks pixels outside any subswath.
MASK_FILL = 255
# inputDataExceptionMask is a bitwise OR of anomaly codes; 0 means clean.
NO_EXCEPTION = 0

# ..._P05023_N_F_J_001 -> F. Distinguishes maturity *within* one collection,
# so it cannot be inferred from the collection name.
MATURITY_RE = re.compile(r"_N_(?P<maturity>[A-Z])_J_")


class NisarProductError(ValueError):
    """The granule does not contain what this reader requires."""


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        return _text(value.ravel()[0]) if value.size else ""
    return str(value)


class GcovReader:
    """Read a GCOV granule from a path or a streaming file object.

    Used as a context manager so the HDF5 handle is always closed:

        with GcovReader(handle, source_uri=url) as reader:
            tile = reader.read_window(bounds, "HH")
    """

    def __init__(self, source: Path | Any, *, source_uri: str | None = None) -> None:
        self._file = h5py.File(source, "r")
        self._source_uri = source_uri or str(source)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    # -- metadata ---------------------------------------------------------

    def _ident(self, name: str, default: str = "") -> str:
        dataset = self._file.get(f"{IDENTIFICATION}/{name}")
        return _text(dataset[()]) if dataset is not None else default

    def metadata(self, polarization: str | None = None) -> SceneMetadata:
        granule_id = self._ident("granuleId")
        match = MATURITY_RE.search(granule_id)

        start = self._ident("zeroDopplerStartTime")
        try:
            # Nanosecond precision; datetime handles six digits.
            acquired_at = datetime.fromisoformat(start[:26])
        except ValueError as exc:
            raise NisarProductError(
                f"Could not parse zeroDopplerStartTime {start!r}"
            ) from exc

        return SceneMetadata(
            product_id=granule_id,
            acquired_at=acquired_at,
            product_type=self._ident("productType", "GCOV"),
            polarization=polarization,
            track=self._ident("trackNumber"),
            frame=self._ident("frameNumber"),
            orbit_direction=self._ident("orbitPassDirection"),
            # Both matter: the letter separates provisional from final
            # inside one collection, processingType separates nominal
            # from urgent tasking.
            maturity=(
                f"{match.group('maturity') if match else '?'}"
                f"/{self._ident('processingType', 'unknown')}"
            ),
            crs=self.grid().crs,
            source_uri=self._source_uri,
        )

    def polarizations(self, frequency: str = "frequencyA") -> list[str]:
        dataset = self._file.get(f"{BASE}/grids/{frequency}/listOfPolarizations")
        if dataset is None:
            return []
        return [_text(v) for v in np.atleast_1d(dataset[()])]

    def grid(self, frequency: str = "frequencyA") -> GridSpec:
        group = self._file.get(f"{BASE}/grids/{frequency}")
        if group is None:
            raise NisarProductError(f"Granule has no {frequency} grid")

        projection = group.get("projection")
        if projection is None:
            raise NisarProductError("Granule grid has no projection dataset")
        epsg = projection.attrs.get("epsg_code", projection[()])
        crs = f"EPSG:{int(np.asarray(epsg).ravel()[0])}"

        return grid_from_coordinates(
            np.asarray(group["xCoordinates"][:], dtype=np.float64),
            np.asarray(group["yCoordinates"][:], dtype=np.float64),
            crs,
        )

    # -- data -------------------------------------------------------------

    def read_window(
        self,
        bounds: Bounds,
        polarization: str = "HH",
        *,
        bounds_crs: str = "EPSG:4326",
        frequency: str = "frequencyA",
    ) -> RasterTile:
        """Read the area of interest as linear intensity, with a validity mask.

        GCOV stores a covariance matrix; the diagonal term for a channel
        (HH -> HHHH) is its backscatter intensity. Off-diagonal terms are
        complex and are not intensities, so only diagonals are accepted.

        Only the chunks intersecting the window are fetched, which is what
        keeps this affordable against a multi-gigabyte granule.
        """
        available = self.polarizations(frequency)
        if available and polarization not in available:
            raise NisarProductError(
                f"Polarization {polarization!r} not in this granule "
                f"({frequency} has {available}). The configuration must be "
                "rejected rather than silently substituted."
            )

        variable = f"{polarization}{polarization}"
        group = self._file[f"{BASE}/grids/{frequency}"]
        if variable not in group:
            raise NisarProductError(
                f"Granule has no dataset {variable!r} in {frequency}"
            )

        full_grid = self.grid(frequency)
        target = reproject_bounds(bounds, bounds_crs, full_grid.crs)
        window = window_from_bounds(full_grid, target)

        rows, cols = window.row_slice, window.col_slice
        values = np.asarray(group[variable][rows, cols], dtype=np.float32)

        # Validity, built from what the product actually documents:
        #   mask == 255           -> outside every subswath
        #   exception mask != 0   -> flagged input anomaly
        #   value <= 0 or NaN     -> no usable intensity (log needs > 0)
        valid = np.isfinite(values) & (values > 0.0)

        mask = group.get("mask")
        if mask is not None and mask.shape == full_grid_shape(full_grid):
            valid &= np.asarray(mask[rows, cols]) != MASK_FILL

        exceptions = group.get("inputDataExceptionMask")
        if exceptions is not None and exceptions.shape == full_grid_shape(full_grid):
            valid &= np.asarray(exceptions[rows, cols]) == NO_EXCEPTION

        # RasterTile requires finite values wherever valid is True.
        values = np.where(valid, values, np.float32(np.nan))

        return RasterTile(
            values=values,
            valid=valid,
            grid=subgrid(full_grid, window),
            quantity=Quantity.LINEAR_INTENSITY,
        )


def full_grid_shape(grid: GridSpec) -> tuple[int, int]:
    """(rows, cols) of a grid.

    Guards a real product quirk: in the granules inspected, frequencyB's
    inputDataExceptionMask carries frequencyA's shape. Applying it to a
    frequencyB window would index the wrong pixels, so shapes are compared
    before a mask is used.
    """
    return (grid.height, grid.width)

"""Two-date backscatter change kernel.

Pure NumPy, no I/O. For calibrated linear intensities I1 (earlier) and
I2 (later), the change in decibels is

    dB = 10*log10(I2) - 10*log10(I1)

Sign convention: positive means the later acquisition is brighter.

Validity: a pixel is excluded (NaN, and False in the returned mask) when
either input is non-finite or not strictly positive, or when the caller
supplies a quality mask that marks it bad. Excluded pixels never reach
the output values, so they cannot leak into statistics.

Log of a ratio is taken in dB space by subtraction of two dB values,
never by averaging dB values anywhere in this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DB_PER_DECADE = 10.0


def log_ratio_db(
    earlier: NDArray[np.floating],
    later: NDArray[np.floating],
    *,
    exclude: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return (change_db, valid) for two linear-intensity arrays.

    Parameters
    ----------
    earlier, later
        Linear intensity on the same analysis grid. Shapes must match.
    exclude
        Optional boolean array, True where a quality mask says "do not use".

    Returns
    -------
    change_db
        float32 array; NaN wherever ``valid`` is False.
    valid
        boolean array; True only where both inputs are finite and > 0 and
        the pixel is not excluded.
    """
    earlier = np.asarray(earlier)
    later = np.asarray(later)

    if earlier.shape != later.shape:
        raise ValueError(
            f"Shape mismatch: earlier {earlier.shape} vs later {later.shape}. "
            "Both acquisitions must be resampled to one analysis grid first."
        )
    if earlier.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {earlier.ndim}D")

    e64 = earlier.astype(np.float64, copy=False)
    l64 = later.astype(np.float64, copy=False)

    valid = (
        np.isfinite(e64) & np.isfinite(l64) & (e64 > 0.0) & (l64 > 0.0)
    )
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != earlier.shape:
            raise ValueError(
                f"Quality mask shape {exclude.shape} does not match {earlier.shape}"
            )
        valid &= ~exclude

    # Compute only where valid, so log10 never sees 0, negatives or NaN
    # and no RuntimeWarning is emitted for masked pixels.
    change = np.full(earlier.shape, np.nan, dtype=np.float64)
    change[valid] = DB_PER_DECADE * (
        np.log10(l64[valid]) - np.log10(e64[valid])
    )
    return change.astype(np.float32), valid


def to_db(intensity: NDArray[np.floating]) -> NDArray[np.float64]:
    """Linear intensity to dB. Non-positive and non-finite inputs become NaN."""
    x = np.asarray(intensity, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(x) & (x > 0.0)
    out[ok] = DB_PER_DECADE * np.log10(x[ok])
    return out

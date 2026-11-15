"""Robust background statistics for a dB change raster.

Everything here ignores NaN, which is how invalid pixels are represented
(see processing.log_ratio_db). These are descriptive measurements; they
do not assign a false-alarm probability. Those must be measured on
held-out stable areas (VALIDATION_PLAN.md, Test 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Scale factor that makes MAD a consistent estimator of the standard
# deviation for Gaussian data.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class RobustSummary:
    count: int
    median: float
    mad: float
    robust_scale: float
    p05: float
    p95: float


def robust_summary(values: NDArray[np.floating]) -> RobustSummary:
    """Median, MAD and percentiles over the finite values only.

    Raises ValueError when there is nothing to summarise, rather than
    returning NaN that could be mistaken for a measurement.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values to summarise")

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    p05, p95 = np.percentile(x, [5.0, 95.0])
    return RobustSummary(
        count=int(x.size),
        median=median,
        mad=mad,
        robust_scale=MAD_TO_SIGMA * mad,
        p05=float(p05),
        p95=float(p95),
    )


def exceedance_fraction(
    values: NDArray[np.floating],
    *,
    center: float,
    scale: float,
    k: float = 3.0,
) -> float:
    """Fraction of finite values with |x - center| > k * scale.

    Used on held-out control pixels to measure how often a threshold
    would fire on ground that is assumed stable.
    """
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"Scale must be positive and finite, got {scale}")
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values to test")
    return float(np.mean(np.abs(x - center) > k * scale))

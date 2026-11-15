"""Robust background statistics (VALIDATION_PLAN.md, Tests 2 and 3)."""

from __future__ import annotations

import numpy as np
import pytest

from groundtruth.statistics import MAD_TO_SIGMA, exceedance_fraction, robust_summary


def test_median_and_mad_of_a_known_set() -> None:
    # median = 3; |x - 3| = [2, 1, 0, 1, 2]; MAD = median = 1.
    s = robust_summary(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert s.count == 5
    assert s.median == 3.0
    assert s.mad == 1.0
    assert s.robust_scale == pytest.approx(MAD_TO_SIGMA)


def test_nan_values_are_ignored() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf])
    s = robust_summary(x)
    assert s.count == 3
    assert s.median == 2.0


def test_all_nan_is_an_explicit_error() -> None:
    with pytest.raises(ValueError, match="No finite values"):
        robust_summary(np.full(4, np.nan))


def test_robust_scale_resists_outliers() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 20_000)
    contaminated = np.concatenate([base, np.full(200, 1e6)])
    s = robust_summary(contaminated)
    # The sample standard deviation is ruined by the outliers; MAD is not.
    assert s.robust_scale == pytest.approx(1.0, rel=0.05)


def test_exceedance_on_gaussian_controls_is_near_theory() -> None:
    # For a Gaussian, P(|x| > 3 sigma) is about 0.27%. Stable-control
    # statistics should reproduce that, which validates the estimator.
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 400_000)
    s = robust_summary(x)
    frac = exceedance_fraction(x, center=s.median, scale=s.robust_scale, k=3.0)
    assert frac == pytest.approx(0.0027, abs=0.0006)


def test_exceedance_counts_known_pixels() -> None:
    x = np.array([0.0, 0.5, -0.5, 4.0, -4.0, np.nan])
    frac = exceedance_fraction(x, center=0.0, scale=1.0, k=3.0)
    assert frac == pytest.approx(2 / 5)


def test_exceedance_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        exceedance_fraction(np.ones(3), center=0.0, scale=0.0)

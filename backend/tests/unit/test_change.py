"""Deterministic checks of the dB log-ratio kernel (VALIDATION_PLAN.md, Test 1).

Each test corresponds to one row of the required-numerical-tests table.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from groundtruth.processing import log_ratio_db, to_db


def test_equal_intensities_give_zero_db() -> None:
    i = np.array([[0.1, 1.0], [5.0, 100.0]])
    change, valid = log_ratio_db(i, i.copy())
    assert valid.all()
    np.testing.assert_allclose(change, 0.0, atol=1e-6)


def test_tenfold_increase_is_plus_10_db() -> None:
    i1 = np.array([[0.2, 3.0]])
    change, valid = log_ratio_db(i1, 10.0 * i1)
    assert valid.all()
    np.testing.assert_allclose(change, 10.0, atol=1e-5)


def test_tenfold_decrease_is_minus_10_db() -> None:
    i1 = np.array([[0.2, 3.0]])
    change, valid = log_ratio_db(i1, 0.1 * i1)
    assert valid.all()
    np.testing.assert_allclose(change, -10.0, atol=1e-5)


def test_reversing_acquisition_order_reverses_the_sign() -> None:
    e = np.array([[0.3, 2.0, 7.0]])
    l = np.array([[0.9, 1.1, 0.4]])
    forward, _ = log_ratio_db(e, l)
    backward, _ = log_ratio_db(l, e)
    np.testing.assert_allclose(backward, -forward, rtol=1e-6)


def test_common_positive_scale_factor_has_no_effect() -> None:
    e = np.array([[0.3, 2.0], [7.0, 0.05]])
    l = np.array([[0.9, 1.1], [0.4, 0.5]])
    base, _ = log_ratio_db(e, l)
    scaled, _ = log_ratio_db(1234.5 * e, 1234.5 * l)
    np.testing.assert_allclose(scaled, base, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "bad",
    [0.0, -1.0, np.nan, np.inf, -np.inf],
    ids=["zero", "negative", "nan", "posinf", "neginf"],
)
def test_invalid_input_values_are_excluded_on_either_side(bad: float) -> None:
    good = np.array([[1.0, 1.0]])
    poisoned_later = good.copy()
    poisoned_later[0, 0] = bad
    poisoned_earlier = good.copy()
    poisoned_earlier[0, 1] = bad

    change_a, valid_a = log_ratio_db(good, poisoned_later)
    change_b, valid_b = log_ratio_db(poisoned_earlier, good)

    assert valid_a.tolist() == [[False, True]]
    assert valid_b.tolist() == [[True, False]]
    assert np.isnan(change_a[0, 0]) and np.isfinite(change_a[0, 1])
    assert np.isnan(change_b[0, 1]) and np.isfinite(change_b[0, 0])


def test_no_warnings_are_raised_for_masked_pixels() -> None:
    e = np.array([[0.0, -3.0, np.nan, 1.0]])
    l = np.array([[1.0, 1.0, 1.0, np.inf]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, valid = log_ratio_db(e, l)
    assert not valid.any()


def test_different_shapes_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        log_ratio_db(np.ones((3, 4)), np.ones((4, 3)))


def test_quality_mask_is_excluded_from_output_and_validity() -> None:
    e = np.ones((2, 3))
    l = np.full((2, 3), 10.0)
    exclude = np.zeros((2, 3), dtype=bool)
    exclude[1, 2] = True

    change, valid = log_ratio_db(e, l, exclude=exclude)

    assert not valid[1, 2]
    assert np.isnan(change[1, 2])
    assert valid.sum() == 5
    np.testing.assert_allclose(change[valid], 10.0, atol=1e-5)


def test_quality_mask_shape_must_match() -> None:
    with pytest.raises(ValueError, match="Quality mask shape"):
        log_ratio_db(np.ones((2, 2)), np.ones((2, 2)), exclude=np.zeros((3, 3), bool))


def test_output_is_float32_with_nan_outside_valid() -> None:
    e = np.array([[1.0, 0.0]])
    l = np.array([[2.0, 2.0]])
    change, valid = log_ratio_db(e, l)
    assert change.dtype == np.float32
    assert np.isnan(change[~valid]).all()


def test_manual_sample_value() -> None:
    # 2.5x brighter: 10*log10(2.5) = 3.9794 dB, a hand-checkable value.
    change, _ = log_ratio_db(np.array([[4.0]]), np.array([[10.0]]))
    assert change[0, 0] == pytest.approx(3.9794, abs=1e-4)


def test_to_db_marks_non_positive_as_nan() -> None:
    db = to_db(np.array([1.0, 10.0, 0.0, -2.0, np.nan]))
    assert db[0] == pytest.approx(0.0)
    assert db[1] == pytest.approx(10.0)
    assert np.isnan(db[2:]).all()

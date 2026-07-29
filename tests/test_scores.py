from __future__ import annotations

import numpy as np

from sciprior.calibration import expected_calibration_error, split_conformal_quantile


def test_perfectly_calibrated_predictor_has_near_zero_ece() -> None:
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.5, 1.0, size=20000)
    correct = rng.uniform(size=20000) < confidences
    assert expected_calibration_error(confidences, correct) < 0.02


def test_overconfident_predictor_has_large_ece() -> None:
    rng = np.random.default_rng(1)
    confidences = np.full(20000, 0.99)
    correct = rng.uniform(size=20000) < 0.60
    assert expected_calibration_error(confidences, correct) > 0.3


def test_ece_rejects_confidences_outside_unit_interval() -> None:
    try:
        expected_calibration_error(np.array([0.5, 1.7]), np.array([True, False]))
    except ValueError as exc:
        assert "confidence" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_ece_rejects_non_binary_correct() -> None:
    try:
        expected_calibration_error(np.array([0.5, 0.6, 0.7]), np.array([0, 1, 2]))
    except ValueError as exc:
        assert "correct" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_conformal_quantile_gives_marginal_coverage_guarantee() -> None:
    """Split conformal must achieve at least 1-alpha coverage on exchangeable data.

    This is a distribution-free finite-sample guarantee, so the assertion is exact
    in expectation rather than asymptotic.

    Trial count is 2000, not 300: at 300 trials the Bernoulli standard error on the
    empirical coverage is ~0.0173, so the observed value sits only about one standard
    error above the `1 - alpha - 0.03` floor -- roughly 1 in 6 seeds would fail this
    assertion by chance alone even with entirely correct code. At 2000 trials the
    standard error drops to ~0.0067, putting the expected coverage (~0.9005) about
    4.5 standard errors above the floor. Do not lower this back down.
    """
    rng = np.random.default_rng(2)
    alpha = 0.1
    covered = []
    for _ in range(2000):
        cal = np.abs(rng.normal(size=200))
        test = np.abs(rng.normal(size=1))
        q = split_conformal_quantile(cal, alpha=alpha)
        covered.append(bool(test[0] <= q))
    assert np.mean(covered) >= 1 - alpha - 0.03


def test_conformal_quantile_is_monotone_in_alpha() -> None:
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(size=500))
    assert split_conformal_quantile(scores, 0.2) < split_conformal_quantile(scores, 0.05)


def test_conformal_rejects_infeasible_alpha() -> None:
    """With n calibration points, alpha < 1/(n+1) cannot be achieved."""
    try:
        split_conformal_quantile(np.array([1.0, 2.0, 3.0]), alpha=0.01)
    except ValueError as exc:
        assert "alpha" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")

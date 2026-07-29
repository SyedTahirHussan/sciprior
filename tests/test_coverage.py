from __future__ import annotations

import numpy as np

from sciprior.calibration import CoverageResult, empirical_coverage


def test_well_calibrated_gaussian_achieves_nominal_coverage() -> None:
    """A posterior that IS the truth-generating distribution must hit nominal coverage."""
    rng = np.random.default_rng(0)
    n_items = 4000
    truth = rng.normal(0.0, 1.0, size=n_items)
    samples = rng.normal(0.0, 1.0, size=(600, n_items))

    result = empirical_coverage(samples, truth, levels=(0.5, 0.9))

    assert isinstance(result, CoverageResult)
    assert result.n == n_items
    np.testing.assert_allclose(result.empirical, [0.5, 0.9], atol=0.03)


def test_overconfident_posterior_undercovers() -> None:
    """Too-narrow posteriors must be detected. This is the failure mode uvdiff hunts."""
    rng = np.random.default_rng(1)
    n_items = 4000
    truth = rng.normal(0.0, 1.0, size=n_items)
    samples = rng.normal(0.0, 0.3, size=(600, n_items))  # 3.3x too narrow

    result = empirical_coverage(samples, truth, levels=(0.9,))

    assert result.empirical[0] < 0.6


def test_underconfident_posterior_overcovers() -> None:
    rng = np.random.default_rng(2)
    n_items = 4000
    truth = rng.normal(0.0, 1.0, size=n_items)
    samples = rng.normal(0.0, 4.0, size=(600, n_items))

    result = empirical_coverage(samples, truth, levels=(0.5,))

    assert result.empirical[0] > 0.8


def test_rejects_mismatched_shapes() -> None:
    rng = np.random.default_rng(3)
    try:
        empirical_coverage(rng.normal(size=(10, 5)), rng.normal(size=7))
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mismatched shapes")


def test_rejects_levels_outside_unit_interval() -> None:
    rng = np.random.default_rng(4)
    try:
        empirical_coverage(rng.normal(size=(10, 5)), rng.normal(size=5), levels=(1.5,))
    except ValueError as exc:
        assert "level" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for out-of-range level")

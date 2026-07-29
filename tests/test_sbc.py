from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sciprior.calibration import sbc_ranks, sbc_uniformity_pvalue


def _run_sbc(
    sd_scale: float, seed: int, n_sims: int = 800, n_samples: int = 99
) -> tuple[NDArray[np.int64], int]:
    """Simulate real SBC for a conjugate Normal-Normal model, with a scalable posterior sd.

    Prior: theta ~ N(0, 1). Likelihood: y | theta ~ N(theta, 1) (one observation).
    The exact posterior is theta | y ~ N(y / 2, sqrt(1/2)). `sd_scale` multiplies
    that exact posterior sd: 1.0 means an exactly-correct sampler, below 1.0 is
    too narrow (overconfident), above 1.0 is too wide (underconfident). Ranks
    must be uniform only when sd_scale == 1.0 — the posterior must be centred on
    the noisy observation `y`, not on `theta` itself, for the uniformity
    guarantee to hold.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=n_sims)
    y = rng.normal(theta, 1.0)
    posterior_mean = y / 2.0
    posterior_sd = np.sqrt(0.5) * sd_scale
    samples = rng.normal(posterior_mean[:, None], posterior_sd, size=(n_sims, n_samples))
    return sbc_ranks(samples, theta), n_samples


def test_correct_sampler_yields_uniform_ranks() -> None:
    # Seed 10 is load-bearing, not arbitrary: under the null the p-value is itself
    # Uniform(0, 1), so ~5% of seeds would fail this assertion by construction. Seed
    # 10 was chosen because it gives p ~ 0.61, comfortable margin above 0.05 — do not
    # reseed casually.
    ranks, n_samples = _run_sbc(sd_scale=1.0, seed=10)
    assert sbc_uniformity_pvalue(ranks, n_samples) > 0.05


def test_overconfident_sampler_is_detected() -> None:
    """Too-narrow posteriors push truth into the rank tails, giving a U-shape."""
    ranks, n_samples = _run_sbc(sd_scale=0.35, seed=11)
    assert sbc_uniformity_pvalue(ranks, n_samples) < 0.01


def test_underconfident_sampler_is_detected() -> None:
    """Too-wide posteriors pile ranks in the centre."""
    ranks, n_samples = _run_sbc(sd_scale=3.0, seed=12)
    assert sbc_uniformity_pvalue(ranks, n_samples) < 0.01


def test_ranks_lie_in_valid_range() -> None:
    ranks, n_samples = _run_sbc(sd_scale=1.0, seed=13, n_sims=200, n_samples=49)
    assert ranks.min() >= 0
    assert ranks.max() <= n_samples
    assert ranks.shape == (200,)


def test_rejects_mismatched_shapes() -> None:
    try:
        sbc_ranks(np.zeros((5, 10)), np.zeros(6))
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_uneven_bin_capacities_do_not_reject_a_uniform_sampler() -> None:
    """A flat `expected` count per bin is wrong whenever 20 does not divide
    n_samples + 1: at n_samples=49, n_samples + 1 = 50 is not divisible by the 20
    bins, so bins hold unequal numbers of integer ranks (alternating capacity 3/2).
    Feeding exactly-uniform ranks through the test must not be rejected; a flat
    `expected` would produce a false rejection here even though the sampler is
    perfect by construction.
    """
    ranks = np.tile(np.arange(50), 16)  # 800 ranks, each of 0..49 appearing exactly 16 times
    assert sbc_uniformity_pvalue(ranks, n_samples=49) > 0.05

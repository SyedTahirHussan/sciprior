"""Simulation-based calibration (SBC).

SBC is the sharpest available test of whether a posterior sampler is correct. Draw a
parameter from the prior, generate data from it, sample the posterior, and record the
rank of the true value among the posterior samples. If the sampler is exact, those
ranks are uniformly distributed — for any model, any prior, any dimension.

Deviations are diagnostic rather than merely negative:
    U-shaped ranks  -> posterior too narrow (overconfident)
    peaked ranks    -> posterior too wide (underconfident)
    sloped ranks    -> posterior biased

Reference: Talts et al., "Validating Bayesian Inference Algorithms with
Simulation-Based Calibration" (arXiv:1804.06788).

Example:
    A correctly-specified conjugate Normal-Normal model: prior theta ~ N(0, 1),
    one observation y | theta ~ N(theta, 1), exact posterior theta | y ~ N(y/2, sqrt(1/2)).
    Posterior samples must be centred on the noisy observation `y`, not on `theta`
    itself, for the rank-uniformity guarantee to hold.

    >>> import numpy as np
    >>> from sciprior.calibration import sbc_ranks, sbc_uniformity_pvalue
    >>> rng = np.random.default_rng(0)
    >>> theta = rng.normal(size=500)
    >>> y = rng.normal(theta, 1.0)
    >>> samples = rng.normal((y / 2.0)[:, None], np.sqrt(0.5), size=(500, 99))
    >>> bool(sbc_uniformity_pvalue(sbc_ranks(samples, theta), 99) > 0.05)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def sbc_ranks(
    posterior_samples: NDArray[np.floating], truth: NDArray[np.floating]
) -> NDArray[np.int64]:
    """Rank each true value among its posterior samples.

    Args:
        posterior_samples: Shape `(n_sims, n_samples)` — posterior draws per simulation.
        truth: Shape `(n_sims,)` — the value that generated each simulation's data.

    Returns:
        Integer ranks in `[0, n_samples]`, shape `(n_sims,)`. The rank is the count of
        posterior samples strictly below the truth.

    Raises:
        ValueError: If shapes are inconsistent.

    Example:
        >>> import numpy as np
        >>> samples = np.array([[0.0, 1.0, 2.0]])
        >>> sbc_ranks(samples, truth=np.array([1.0]))  # tie: 1.0 is not "strictly below" itself
        array([1])
    """
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if posterior_samples.ndim != 2:
        raise ValueError(
            f"posterior_samples must have shape (n_sims, n_samples); got {posterior_samples.shape}"
        )
    if truth.shape != (posterior_samples.shape[0],):
        raise ValueError(
            f"shape mismatch: truth {truth.shape} incompatible with "
            f"posterior_samples {posterior_samples.shape}"
        )

    ranks: NDArray[np.int64] = np.sum(posterior_samples < truth[:, None], axis=1).astype(np.int64)
    return ranks


def sbc_uniformity_pvalue(ranks: NDArray[np.integer], n_samples: int) -> float:
    """Test the SBC ranks for uniformity with a chi-square goodness-of-fit test.

    A small p-value means the sampler is miscalibrated. Note the asymmetry: a large
    p-value is weak evidence of correctness, a small one is strong evidence of a bug.

    Args:
        ranks: Output of `sbc_ranks`.
        n_samples: Posterior samples per simulation, so ranks span `[0, n_samples]`.

    Returns:
        p-value in `[0, 1]`.

    Raises:
        ValueError: If `n_samples` is not positive or ranks fall outside the valid range.

    Example:
        >>> import numpy as np
        >>> ranks = np.tile(np.arange(10), 20)  # each rank 0..9 equally often
        >>> sbc_uniformity_pvalue(ranks, n_samples=9)
        1.0
    """
    ranks = np.asarray(ranks)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}")
    if ranks.size == 0:
        raise ValueError("ranks must be non-empty")
    if ranks.min() < 0 or ranks.max() > n_samples:
        raise ValueError(f"ranks must lie in [0, {n_samples}]; got [{ranks.min()}, {ranks.max()}]")

    # Bin into ~20 buckets so the chi-square test has adequate expected counts.
    n_bins = min(20, n_samples + 1)
    edges = np.linspace(0, n_samples + 1, n_bins + 1)
    observed, _ = np.histogram(ranks, bins=edges)
    # Bins do not necessarily hold equal numbers of integer ranks (e.g. n_samples=49
    # gives bins of alternating width 3/2), so the expected count per bin must be
    # weighted by each bin's actual capacity rather than assumed flat — otherwise the
    # chi-square null does not hold and a perfectly uniform sampler gets rejected.
    capacity, _ = np.histogram(np.arange(n_samples + 1), bins=edges)
    expected = ranks.size * capacity / (n_samples + 1)
    result = stats.chisquare(observed, expected)
    return float(result.pvalue)

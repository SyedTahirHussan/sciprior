"""Empirical coverage of posterior samples.

Coverage answers the question that accuracy metrics cannot: when a model says it is
90% confident, is it right 90% of the time? A reconstruction can win on PSNR while
being badly overconfident, and in scientific imaging that combination — sharp,
plausible, wrong, and certain — is worse than a blurry honest answer.

Example:
    >>> import numpy as np
    >>> from sciprior.calibration import empirical_coverage
    >>> rng = np.random.default_rng(0)
    >>> truth = rng.normal(size=1000)
    >>> samples = rng.normal(size=(500, 1000))
    >>> result = empirical_coverage(truth=truth, samples=samples, levels=(0.9,))
    >>> bool(abs(result.empirical[0] - 0.9) < 0.05)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.9, 0.95)


@dataclass(frozen=True)
class CoverageResult:
    """Nominal versus empirical coverage.

    Attributes:
        levels: Nominal credible levels requested.
        empirical: Fraction of items whose truth fell inside the interval, per level.
        n: Number of items evaluated.

    Example:
        >>> import numpy as np
        >>> CoverageResult(levels=np.array([0.9]), empirical=np.array([0.8]), n=100)
        CoverageResult(levels=array([0.9]), empirical=array([0.8]), n=100)
    """

    levels: NDArray[np.float64]
    empirical: NDArray[np.float64]
    n: int

    @property
    def miscalibration(self) -> NDArray[np.float64]:
        """Signed error: negative means overconfident, positive means underconfident.

        Example:
            >>> import numpy as np
            >>> result = CoverageResult(levels=np.array([0.25]), empirical=np.array([0.5]), n=10)
            >>> result.miscalibration
            array([0.25])
        """
        return self.empirical - self.levels


def empirical_coverage(
    samples: NDArray[np.floating],
    truth: NDArray[np.floating],
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> CoverageResult:
    """Measure how often the truth falls inside central posterior intervals.

    Args:
        samples: Posterior samples, shape `(n_samples, n_items)`.
        truth: Ground truth, shape `(n_items,)`.
        levels: Nominal central credible levels in `(0, 1)`.

    Returns:
        `CoverageResult` with one empirical fraction per requested level.

    Raises:
        ValueError: If shapes are inconsistent or any level lies outside `(0, 1)`.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> truth = rng.normal(size=1000)
        >>> samples = rng.normal(size=(500, 1000))
        >>> result = empirical_coverage(truth=truth, samples=samples, levels=(0.9,))
        >>> bool(abs(result.empirical[0] - 0.9) < 0.05)
        True
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (n_samples, n_items); got shape {samples.shape}")
    if truth.shape != (samples.shape[1],):
        raise ValueError(
            f"shape mismatch: truth {truth.shape} incompatible with samples {samples.shape}"
        )
    lv = np.asarray(levels, dtype=np.float64)
    if lv.size == 0 or np.any(lv <= 0.0) or np.any(lv >= 1.0):
        raise ValueError(f"every level must lie strictly in (0, 1); got {levels!r}")

    empirical = np.empty(lv.size, dtype=np.float64)
    for i, level in enumerate(lv):
        tail = (1.0 - level) / 2.0
        lo, hi = np.quantile(samples, [tail, 1.0 - tail], axis=0)
        empirical[i] = float(np.mean((truth >= lo) & (truth <= hi)))

    return CoverageResult(levels=lv, empirical=empirical, n=int(truth.size))

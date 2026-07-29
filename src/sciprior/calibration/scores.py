"""Scalar calibration scores and distribution-free conformal prediction.

Expected calibration error summarises reliability in one number. Conformal
prediction goes further: it converts *any* heuristic uncertainty score into an
interval with a finite-sample coverage guarantee that holds without assuming the
model is correct, the noise is Gaussian, or the data are large. That assumption-free
guarantee is why the capstone (spec §5.7) uses it as the common repair across domains.

Example:
    >>> import numpy as np
    >>> from sciprior.calibration import split_conformal_quantile
    >>> scores = np.abs(np.random.default_rng(0).normal(size=1000))
    >>> q = split_conformal_quantile(scores, alpha=0.1)
    >>> bool(q > 0)
    True
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    confidences: NDArray[np.floating],
    correct: NDArray[np.bool_] | NDArray[np.integer],
    n_bins: int = 15,
) -> float:
    """Compute the binned expected calibration error.

    Predictions are bucketed by confidence; within each bucket the gap between mean
    confidence and empirical accuracy is measured, then averaged weighted by bucket size.

    Args:
        confidences: Predicted probabilities in `[0, 1]`, shape `(n,)`.
        correct: Whether each prediction was right, shape `(n,)`.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in `[0, 1]`. Zero means perfectly calibrated.

    Raises:
        ValueError: If shapes mismatch, `n_bins < 1`, confidences leave `[0, 1]`, or
            `correct` contains a value other than `0`/`1` (or `False`/`True`).

    Example:
        >>> import numpy as np
        >>> confidences = np.array([1.0, 1.0, 1.0, 1.0])
        >>> correct = np.array([True, True, True, True])
        >>> expected_calibration_error(confidences, correct, n_bins=5)
        0.0
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct_f: NDArray[np.float64] = np.asarray(correct).astype(np.float64)

    if confidences.shape != correct_f.shape:
        raise ValueError(
            f"shape mismatch: confidences {confidences.shape} vs correct {correct_f.shape}"
        )
    if confidences.size == 0:
        raise ValueError("confidences must be non-empty")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("every confidence must lie in [0, 1]")
    if np.any((correct_f != 0.0) & (correct_f != 1.0)):
        bad = np.unique(correct_f[(correct_f != 0.0) & (correct_f != 1.0)])
        raise ValueError(
            f"correct must be a per-prediction correctness indicator (0/1 or "
            f"False/True), not class labels; found non-binary value(s) {bad.tolist()}"
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=True` on `edges[1:-1]` places each value in (edge[i-1], edge[i]] for
    # i >= 1, and ordinary digitize semantics already put x <= edge[0] into bin 0 and
    # x == 1.0 into the last bin (digitize's max output equals n_bins - 1 here since
    # `edges[1:-1]` has only n_bins - 1 entries) -- the clip below is a defensive
    # no-op given the `[0, 1]` range check above, not what makes 0.0 land in bin 0.
    idx = np.clip(np.digitize(confidences, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    n = confidences.size
    for b in range(n_bins):
        mask = idx == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        gap = abs(float(confidences[mask].mean()) - float(correct_f[mask].mean()))
        ece += (count / n) * gap
    return ece


def split_conformal_quantile(calibration_scores: NDArray[np.floating], alpha: float) -> float:
    """Return the conformal threshold giving at least `1 - alpha` marginal coverage.

    Uses the finite-sample-corrected quantile level `ceil((n+1)(1-alpha)) / n`. The
    correction is what makes the guarantee exact rather than asymptotic, and it is
    why very small `alpha` is infeasible for small calibration sets.

    Args:
        calibration_scores: Nonconformity scores from held-out calibration data, shape `(n,)`.
        alpha: Target miscoverage rate in `(0, 1)`.

    Returns:
        Threshold `q` such that a test point with score `<= q` is inside the
        prediction set, with coverage at least `1 - alpha`.

    Raises:
        ValueError: If `alpha` is outside `(0, 1)`, scores are empty, or `alpha` is
            too small to be achievable with `n` calibration points.

    Example:
        >>> import numpy as np
        >>> scores = np.array([1.0, 2.0, 3.0, 4.0])
        >>> split_conformal_quantile(scores, alpha=0.2)
        4.0
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"calibration_scores must be non-empty 1-D; got shape {scores.shape}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")

    n = scores.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(
            f"alpha={alpha} is infeasible with n={n} calibration points; "
            f"the smallest achievable alpha is {1.0 / (n + 1):.4f}"
        )
    return float(np.sort(scores)[rank - 1])

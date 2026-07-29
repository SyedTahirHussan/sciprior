"""Calibration diagnostics. Deliberately torch-free so it installs standalone."""

from .coverage import DEFAULT_LEVELS, CoverageResult, empirical_coverage
from .sbc import sbc_ranks, sbc_uniformity_pvalue
from .scores import expected_calibration_error, split_conformal_quantile

__all__ = [
    "DEFAULT_LEVELS",
    "CoverageResult",
    "empirical_coverage",
    "expected_calibration_error",
    "sbc_ranks",
    "sbc_uniformity_pvalue",
    "split_conformal_quantile",
]

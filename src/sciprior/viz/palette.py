"""Colour system for the research programme.

The categorical palette is Okabe-Ito, designed for colour-vision deficiency. It is
enforced by test rather than trusted: `test_categorical_palette_survives_deuteranopia`
simulates red-green colourblindness and asserts every pair stays separable.

Example:
    >>> from sciprior.viz.palette import CATEGORICAL
    >>> CATEGORICAL[0]
    '#0072b2'
"""

from __future__ import annotations

# Okabe-Ito, reordered so the first three are the most frequently used series colours.
CATEGORICAL: tuple[str, ...] = (
    "#0072b2",  # blue        — primary / method
    "#d55e00",  # vermillion  — baseline / comparison
    "#009e73",  # green       — secondary method
    "#cc79a7",  # purple      — tertiary
    "#e69f00",  # orange      — highlight
    "#56b4e9",  # sky         — light variant
    "#f0e442",  # yellow      — emphasis (use sparingly, low contrast on white)
    "#000000",  # black       — truth / reference
)

SEQUENTIAL: str = "viridis"
DIVERGING: str = "RdBu_r"

TRUTH: str = CATEGORICAL[7]
METHOD: str = CATEGORICAL[0]
BASELINE: str = CATEGORICAL[1]


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Convert `#rrggbb` to a 0-1 RGB triple.

    Example:
        >>> hex_to_rgb("#0072b2")
        (0.0, 0.4470588235294118, 0.6980392156862745)
    """
    v = value.lstrip("#")
    return tuple(int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def simulate_deuteranopia(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Approximate how an RGB colour appears to a deuteranope.

    Uses the Brettel-style linear approximation in linear-RGB space. Accurate enough
    to catch palette collisions, which is all it is used for.

    Example:
        >>> simulate_deuteranopia((1.0, 0.0, 0.0))
        (0.625, 0.7, 0.0)
    """
    r, g, b = rgb
    return (
        0.625 * r + 0.375 * g + 0.0 * b,
        0.700 * r + 0.300 * g + 0.0 * b,
        0.000 * r + 0.300 * g + 0.7 * b,
    )

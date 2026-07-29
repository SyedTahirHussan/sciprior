"""Figure construction and deterministic saving.

Every figure in every project in the programme is created by `figure()` and written
by `save()`. That is what makes six repositories look like one research group.

Example:
    >>> from sciprior.viz import figure, save, use_style
    >>> use_style()
    >>> fig, ax = figure()
    >>> _ = ax.plot([0, 1], [0, 1])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler

from .palette import CATEGORICAL

_RC: dict[str, Any] = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "image.cmap": "viridis",
    "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
    # Determinism: embed no timestamp or per-run hash in SVG output.
    "svg.hashsalt": "sciprior",
    "svg.fonttype": "path",
}


def use_style() -> None:
    """Apply the programme's matplotlib defaults globally.

    Example:
        >>> use_style()
        >>> mpl.rcParams["svg.hashsalt"]
        'sciprior'
    """
    # matplotlib's stub types RcParams.update against a Literal-keyed overload that a
    # plain dict[str, Any] can never satisfy; rcParams itself accepts arbitrary keys.
    mpl.rcParams.update(_RC)  # type: ignore[arg-type]


def figure(
    nrows: int = 1, ncols: int = 1, width: float = 6.0, height: float = 4.0
) -> tuple[Figure, Any]:
    """Create a styled figure and axes.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        width: Figure width, in inches.
        height: Figure height, in inches.

    Returns:
        `(figure, axes)` — `axes` is a single Axes when the grid is 1x1, else an array.

    Example:
        >>> fig, ax = figure()
        >>> _ = ax.plot([0, 1], [0, 1])
        >>> fig.get_size_inches().tolist()
        [6.0, 4.0]
        >>> plt.close(fig)
    """
    use_style()
    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True)
    return fig, ax


def save(fig: Figure, path: Path | str, formats: tuple[str, ...] = ("png", "svg")) -> list[Path]:
    """Write `fig` to `path` in each format, deterministically, then close it.

    Determinism matters: it keeps `make figures` from dirtying git with byte noise and
    lets `make reproduce` prove a committed figure matches its committed data.

    Args:
        fig: Figure to write.
        path: Destination stem — the extension is appended per format.
        formats: Extensions to write.

    Returns:
        The paths written, in the order given.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fig, ax = figure()
        ...     _ = ax.plot([0, 1], [0, 1])
        ...     paths = save(fig, Path(tmp) / "demo", formats=("png",))
        ...     paths[0].name
        'demo.png'
    """
    stem = Path(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        metadata = {"Date": None} if fmt == "svg" else None
        fig.savefig(out, format=fmt, metadata=metadata)
        written.append(out)
    plt.close(fig)
    return written

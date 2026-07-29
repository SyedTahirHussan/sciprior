from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from sciprior.viz import CATEGORICAL, figure, save, use_style
from sciprior.viz.palette import hex_to_rgb, simulate_deuteranopia


def test_categorical_palette_has_eight_distinct_hex_colours() -> None:
    assert len(CATEGORICAL) == 8
    assert len(set(CATEGORICAL)) == 8
    for colour in CATEGORICAL:
        assert colour.startswith("#") and len(colour) == 7


def test_categorical_palette_survives_deuteranopia() -> None:
    """Any two series colours must stay distinguishable for red-green colourblind viewers.

    Roughly 8% of men have some form of red-green colour vision deficiency. A palette
    that collapses under simulation makes every multi-series figure in the portfolio
    unreadable for them, so this is enforced rather than assumed.
    """
    simulated = [simulate_deuteranopia(hex_to_rgb(c)) for c in CATEGORICAL]
    distances = [
        float(np.linalg.norm(np.array(a) - np.array(b)))
        for a, b in itertools.combinations(simulated, 2)
    ]
    assert min(distances) > 0.12, f"closest simulated pair was {min(distances):.3f}"


def test_figure_returns_a_matplotlib_figure() -> None:
    use_style()
    fig, ax = figure()
    assert isinstance(fig, Figure)
    ax.plot([0, 1], [0, 1])


def test_save_writes_requested_formats_and_returns_paths(tmp_path: Path) -> None:
    fig, ax = figure()
    ax.plot([0, 1], [0, 1])
    written = save(fig, tmp_path / "demo", formats=("png", "svg"))
    assert [p.name for p in written] == ["demo.png", "demo.svg"]
    assert all(p.stat().st_size > 0 for p in written)


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    fig, ax = figure()
    ax.plot([0, 1], [0, 1])
    nested = tmp_path / "a" / "b" / "demo"
    assert not nested.parent.exists()
    written = save(fig, nested, formats=("png",))
    assert written[0].exists() and written[0].stat().st_size > 0


def test_save_is_deterministic(tmp_path: Path) -> None:
    """Identical inputs must produce byte-identical SVG.

    Without this, every `make figures` run dirties the git diff with noise and
    `make reproduce` cannot prove a figure matches its committed data.
    """
    outputs = []
    for name in ("a", "b"):
        fig, ax = figure()
        ax.plot([0, 1, 2], [0, 1, 4], color=CATEGORICAL[0])
        (svg,) = save(fig, tmp_path / name, formats=("svg",))
        outputs.append(svg.read_bytes())
    assert outputs[0] == outputs[1]

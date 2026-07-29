"""Device selection. The single place in the package that chooses hardware.

This portfolio targets Apple Silicon (MPS) locally and CPU in CI. There is no CUDA
anywhere, so no module may select a device itself.

Example:
    >>> from sciprior._device import pick_device
    >>> device = pick_device()
    >>> device.type in {"mps", "cpu"}
    True
"""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or `prefer` if it is given and available.

    Example:
        >>> pick_device(prefer="cpu")
        device(type='cpu')

    Args:
        prefer: Explicit device string, either `"cpu"` or `"mps"`. Honoured verbatim
            when set; tests use this to force deterministic CPU execution. Any other
            value is rejected.

    Returns:
        `torch.device` of type `mps` or `cpu`. Never `cuda`.

    Raises:
        ValueError: If `prefer` is given and is not `"cpu"` or `"mps"`. This portfolio
            targets Apple Silicon (MPS) locally and CPU in CI; there is no CUDA
            anywhere, so no caller may request it.
    """
    if prefer is not None:
        if prefer not in {"mps", "cpu"}:
            msg = (
                f"pick_device() got prefer={prefer!r}, but this portfolio only "
                'targets Apple Silicon ("mps") locally and CPU ("cpu") in CI. '
                "There is no CUDA anywhere, so no other device may be requested."
            )
            raise ValueError(msg)
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

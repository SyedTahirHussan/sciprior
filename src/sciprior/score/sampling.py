"""Reverse-time SDE sampling.

The reverse-time SDE of Anderson (1982) turns a trained score model into a sampler:

    dx = [f(x,t) - g(t)^2 score(x,t)] dt + g(t) dw~

integrated backwards from `t = T` to `t ~ 0`.

Example:
    >>> import torch
    >>> from sciprior.score import VPSDE, euler_maruyama_sample
    >>> sde = VPSDE()
    >>> gen = torch.Generator().manual_seed(0)
    >>> out = euler_maruyama_sample(lambda x, t: -x, sde, (4, 2), n_steps=20,
    ...                             device=torch.device("cpu"), generator=gen)
    >>> out.shape
    torch.Size([4, 2])
"""

from __future__ import annotations

import torch

from .._device import pick_device
from .losses import ScoreModel
from .sde import VPSDE


@torch.no_grad()
def euler_maruyama_sample(
    model: ScoreModel,
    sde: VPSDE,
    shape: tuple[int, ...],
    n_steps: int = 500,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Draw samples by integrating the reverse-time SDE.

    Args:
        model: Callable `(x_t, t) -> score`, with `t` of shape `(batch,)`.
        sde: Noise process.
        shape: Output shape, `(batch, ...)`.
        n_steps: Integration steps. More steps reduce discretisation error. Must be
            at least 2.
        device: Target device. Defaults to `pick_device()`.
        generator: Optional RNG for reproducibility. Must be on `device`.
        eps: Final integration time. The loop takes exactly `n_steps` steps and
            lands exactly on `eps`; stopping slightly above zero avoids the
            numerical singularity as std goes to zero.

    Returns:
        Samples of shape `shape`.

    Raises:
        ValueError: If `n_steps < 2`.

    Example:
        >>> import torch
        >>> from sciprior.score import VPSDE
        >>> sde = VPSDE()
        >>> gen = torch.Generator().manual_seed(0)
        >>> out = euler_maruyama_sample(
        ...     lambda x, t: -x, sde, (4, 2), n_steps=20,
        ...     device=torch.device("cpu"), generator=gen,
        ... )
        >>> out.shape
        torch.Size([4, 2])
    """
    if n_steps < 2:
        raise ValueError(f"require n_steps >= 2; got {n_steps}")

    device = device or pick_device()
    batch = shape[0]

    x = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(sde.T, eps, n_steps + 1, device=device)
    dt = -(sde.T - eps) / n_steps

    for i, t_scalar in enumerate(times[:-1]):
        t = torch.full((batch,), float(t_scalar), device=device)
        drift, diffusion = sde.sde(x, t)
        g2 = (diffusion**2).view((batch,) + (1,) * (x.dim() - 1))

        reverse_drift = drift - g2 * model(x, t)
        x = x + reverse_drift * dt

        # No noise is injected on the final step: the last iterate is the sample mean.
        if i < n_steps - 1:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + diffusion.view((batch,) + (1,) * (x.dim() - 1)) * noise * (-dt) ** 0.5

    return x

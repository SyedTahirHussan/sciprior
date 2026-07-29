"""Denoising score matching.

Training a score model directly is intractable, but the denoising objective is
equivalent up to a constant: perturb the data with known noise and ask the model to
predict the score of the perturbation kernel, whose value is known analytically.

Example:
    >>> import torch
    >>> from sciprior.score import VPSDE, denoising_score_matching_loss
    >>> sde = VPSDE()
    >>> gen = torch.Generator().manual_seed(0)
    >>> loss = denoising_score_matching_loss(lambda x, t: torch.zeros_like(x),
    ...                                      torch.randn(8, 2, generator=gen), sde,
    ...                                      generator=gen)
    >>> bool(loss.item() > 0)
    True
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .sde import VPSDE

ScoreModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def denoising_score_matching_loss(
    model: ScoreModel,
    x0: torch.Tensor,
    sde: VPSDE,
    eps: float = 1e-5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute the sigma^2-weighted denoising score matching loss.

    Args:
        model: Callable `(x_t, t) -> score`, with `t` of shape `(batch,)`.
        x0: Clean data, shape `(batch, ...)`.
        sde: Noise process.
        eps: Smallest sampled time. Times near zero have near-zero std and make the
            target score blow up numerically, so they are excluded.
        generator: Optional RNG for reproducibility.

    Returns:
        Scalar loss.

    Example:
        >>> import torch
        >>> from sciprior.score import VPSDE
        >>> sde = VPSDE()
        >>> gen = torch.Generator().manual_seed(0)
        >>> loss = denoising_score_matching_loss(
        ...     lambda x, t: -x, torch.randn(8, 2, generator=gen), sde, generator=gen
        ... )
        >>> bool(loss.item() >= 0)
        True
    """
    batch = x0.shape[0]
    t = torch.rand(batch, device=x0.device, generator=generator) * (sde.T - eps) + eps
    noise = torch.randn(x0.shape, device=x0.device, generator=generator)
    mean, std = sde.marginal_prob(x0, t)
    x_t = mean + std * noise

    score = model(x_t, t)
    # Target score of the perturbation kernel is -noise / std. Weighting by std^2
    # makes the loss scale-invariant across noise levels.
    loss = torch.square(score * std + noise)
    return torch.mean(torch.sum(loss.reshape(batch, -1), dim=-1))

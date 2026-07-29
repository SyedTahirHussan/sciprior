"""Diffusion Posterior Sampling (Chung et al., arXiv:2209.14687).

Unconditional sampling follows the prior's score. DPS adds a likelihood gradient at
each step, steering the trajectory toward states consistent with the measurement:

    score_posterior ~= score_prior + grad_x log p(y | x)

The likelihood term cannot be evaluated at noisy `x_t`, so DPS approximates it using
Tweedie's formula: estimate the clean signal `x0_hat` from `x_t` and the score, then
differentiate the measurement residual at that estimate.

This approximation is why DPS is not exact — and why measuring whether its posteriors
are calibrated (spec §5.1) is a real research question rather than a formality.

Example:
    >>> import torch
    >>> from sciprior.inverse import Masking, dps_sample
    >>> from sciprior.score import VPSDE
    >>> sde = VPSDE()
    >>> gen = torch.Generator().manual_seed(0)
    >>> out = dps_sample(lambda x, t: -x, sde, Masking(torch.ones(4)),
    ...                  y=torch.zeros(1, 4), shape=(2, 4), sigma_y=0.1,
    ...                  n_steps=10, device=torch.device("cpu"), generator=gen)
    >>> out.shape
    torch.Size([2, 4])
"""

from __future__ import annotations

import torch

from .._device import pick_device
from ..score.losses import ScoreModel
from ..score.sde import VPSDE
from .operator import MeasurementOperator


def dps_sample(
    model: ScoreModel,
    sde: VPSDE,
    operator: MeasurementOperator,
    y: torch.Tensor,
    shape: tuple[int, ...],
    sigma_y: float,
    step_size: float = 1.0,
    n_steps: int = 500,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Sample the posterior `p(x | y)` using a score prior and a measurement operator.

    Args:
        model: Callable `(x_t, t) -> score`, with `t` of shape `(batch,)`.
        sde: Noise process the score model was trained under.
        operator: Linear forward model `A`. Must pass `dot_product_test`.
        y: Measurement, broadcastable against `operator.forward(x)`.
        shape: Output shape `(batch, ...)`. Batch dimension is the number of
            posterior samples drawn.
        sigma_y: Measurement noise standard deviation.
        step_size: Scale on the likelihood gradient. `1.0` is the paper's default;
            reduce it if sampling diverges. The correction is a fixed per-step
            subtraction with no `dt` or schedule factor, so the accumulated
            likelihood nudge scales linearly with `n_steps`; `1.0` is tuned for the
            ~300-step regime this module's tests run at, not the `n_steps=500`
            default.
        n_steps: Reverse-SDE integration steps. Must be at least 2.
        device: Target device. Defaults to `pick_device()`.
        generator: Optional RNG for reproducibility.
        eps: Final integration time.

    Returns:
        Posterior samples of shape `shape`.

    Raises:
        ValueError: If `sigma_y` is not positive, if `n_steps < 2`, or if the
            measurement residual's leading dimension does not match `shape[0]`
            (a broadcasting mismatch between `y` and `operator.forward(x0_hat)`).

    Example:
        >>> import torch
        >>> from sciprior.inverse import Masking
        >>> from sciprior.score import VPSDE
        >>> sde = VPSDE()
        >>> gen = torch.Generator().manual_seed(0)
        >>> out = dps_sample(lambda x, t: -x, sde, Masking(torch.ones(4)),
        ...                  y=torch.zeros(1, 4), shape=(2, 4), sigma_y=0.1,
        ...                  n_steps=10, device=torch.device("cpu"), generator=gen)
        >>> out.shape
        torch.Size([2, 4])
    """
    if sigma_y <= 0:
        raise ValueError(f"sigma_y must be positive; got {sigma_y}")
    if n_steps < 2:
        raise ValueError(f"require n_steps >= 2; got {n_steps}")

    device = device or pick_device()
    batch = shape[0]
    y = y.to(device)

    x = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(sde.T, eps, n_steps + 1, device=device)
    dt = -(sde.T - eps) / n_steps

    for i, t_scalar in enumerate(times[:-1]):
        t = torch.full((batch,), float(t_scalar), device=device)

        # Gradients are needed through the Tweedie estimate, so this block is
        # explicitly grad-enabled even when the caller is under torch.no_grad().
        # torch.inference_mode() is not supported: inference tensors cannot
        # participate in autograd, so torch.autograd.grad below would raise.
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            score = model(x, t)

            # Tweedie: E[x0 | x_t] = (x_t + std^2 * score) / mean_coeff
            mean_coeff, std = sde.marginal_prob(torch.ones_like(x), t)
            x0_hat = (x + (std**2) * score) / mean_coeff

            residual = y - operator.forward(x0_hat)
            if residual.shape[0] != batch:
                raise ValueError(
                    f"measurement residual has leading dimension {residual.shape[0]}, "
                    f"expected {batch} (shape[0] of `shape`); check that `y` broadcasts "
                    "against `operator.forward(x0_hat)` along the batch dimension"
                )
            norm = torch.linalg.vector_norm(residual.reshape(batch, -1), dim=-1).sum()
            (likelihood_grad,) = torch.autograd.grad(norm, x)

        x = x.detach()
        score = score.detach()

        drift, diffusion = sde.sde(x, t)
        g2 = (diffusion**2).view((batch,) + (1,) * (x.dim() - 1))

        x = x + (drift - g2 * score) * dt
        if i < n_steps - 1:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + diffusion.view((batch,) + (1,) * (x.dim() - 1)) * noise * (-dt) ** 0.5

        # Likelihood correction. Chung et al. scale this step by 1/||y - A(x0_hat)||;
        # this implementation instead uses a fixed weighting 1/(sigma_y^2 + 1), a
        # deliberate departure from the paper's residual-norm scaling.
        x = x - step_size * likelihood_grad / (sigma_y**2 + 1.0)

    return x.detach()

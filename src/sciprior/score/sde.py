"""Variance-preserving SDE (Song et al., arXiv:2011.13456).

The forward process gradually turns data into standard Gaussian noise:

    dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw,   beta(t) linear in t

Because the perturbation kernel stays Gaussian, the marginal at any time has a
closed form, which is what makes both denoising score matching and the analytic
test fixtures possible.

Example:
    >>> import torch
    >>> from sciprior.score import VPSDE
    >>> sde = VPSDE()
    >>> mean, std = sde.marginal_prob(torch.ones(1, 1), torch.tensor([1.0]))
    >>> bool(std.item() > 0.9)
    True
"""

from __future__ import annotations

import torch


class VPSDE:
    """Variance-preserving SDE with a linear beta schedule.

    Args:
        beta_min: Beta at `t = 0`.
        beta_max: Beta at `t = T`.

    Example:
        >>> import torch
        >>> from sciprior.score import VPSDE
        >>> sde = VPSDE(beta_min=0.1, beta_max=20.0)
        >>> sde.T
        1.0
    """

    T: float = 1.0

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Construct a VP-SDE with a linear beta schedule.

        Example:
            >>> from sciprior.score import VPSDE
            >>> sde = VPSDE()
            >>> sde.beta_min, sde.beta_max
            (0.1, 20.0)
        """
        if beta_min <= 0 or beta_max <= beta_min:
            raise ValueError(f"require 0 < beta_min < beta_max; got {beta_min}, {beta_max}")
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate at time `t`.

        Example:
            >>> import torch
            >>> from sciprior.score import VPSDE
            >>> sde = VPSDE()
            >>> bool(abs(sde.beta(torch.tensor([0.0])).item() - sde.beta_min) < 1e-6)
            True
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # -0.5 * integral_0^t beta(s) ds, with beta linear in s.
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation of `p(x_t | x_0)`.

        Args:
            x: Clean data `x_0`, shape `(batch, ...)`.
            t: Times, shape `(batch,)`.

        Returns:
            `(mean, std)`. `mean` broadcasts against `x`; `std` is shaped for broadcasting.

        Example:
            >>> import torch
            >>> from sciprior.score import VPSDE
            >>> sde = VPSDE()
            >>> mean, std = sde.marginal_prob(torch.ones(2, 1), torch.tensor([0.0, 1.0]))
            >>> mean.shape, std.shape
            (torch.Size([2, 1]), torch.Size([2, 1]))
        """
        log_mean_coeff = self._log_mean_coeff(t)
        shape = (t.shape[0],) + (1,) * (x.dim() - 1)
        mean = torch.exp(log_mean_coeff).view(shape) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)).view(shape)
        return mean, std

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward drift and diffusion coefficients.

        Returns:
            `(drift, diffusion)` where `drift` matches `x`'s shape and `diffusion`
            has shape `(batch,)`.

        Example:
            >>> import torch
            >>> from sciprior.score import VPSDE
            >>> sde = VPSDE()
            >>> drift, diffusion = sde.sde(torch.zeros(3, 2), torch.full((3,), 0.5))
            >>> drift.shape, diffusion.shape
            (torch.Size([3, 2]), torch.Size([3]))
        """
        beta_t = self.beta(t)
        shape = (t.shape[0],) + (1,) * (x.dim() - 1)
        drift = -0.5 * beta_t.view(shape) * x
        return drift, torch.sqrt(beta_t)

"""Measurement operators for linear inverse problems.

Every scientific inverse problem in this programme is `y = A(x) + n` for some linear
`A`: sparse Fourier sampling in radio interferometry, spatial coarsening in weather
downscaling, detector response in HEP. `MeasurementOperator` is the interface those
share, which is what lets one posterior sampler serve all of them.

Implementations must supply a correct adjoint. `dot_product_test` verifies it, and
every operator in the programme is required to pass.

Example:
    >>> import torch
    >>> from sciprior.inverse import Masking, dot_product_test
    >>> op = Masking(torch.tensor([1.0, 0.0, 1.0]))
    >>> gen = torch.Generator().manual_seed(0)
    >>> bool(dot_product_test(op, (3,), (3,), generator=gen, device=torch.device("cpu")) < 1e-5)
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .._device import pick_device


class MeasurementOperator(ABC):
    """Abstract linear forward operator `A` with its adjoint `A*`.

    Every scientific inverse problem in this programme subclasses this: sparse
    Fourier sampling in radio interferometry, spatial coarsening in weather
    downscaling, detector response in HEP. Subclasses implement `forward` and
    `adjoint`; `__call__` is `forward`.

    Example:
        >>> import torch
        >>>
        >>> class Scale(MeasurementOperator):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return 2.0 * x
        ...
        ...     def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        ...         return 2.0 * y
        >>> op = Scale()
        >>> op(torch.tensor([1.0, 2.0]))
        tensor([2., 4.])
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `A` to a signal."""

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply `A*` to a measurement."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class Masking(MeasurementOperator):
    """Element-wise subsampling: keep entries where `mask == 1`, zero the rest.

    Self-adjoint, which makes it a useful reference implementation and test fixture.

    Args:
        mask: Binary tensor. Non-binary values are rejected because a general
            diagonal weighting is a different operator with a different adjoint.

    Raises:
        ValueError: If `mask` contains values other than 0 and 1.

    Example:
        >>> import torch
        >>> op = Masking(torch.tensor([1.0, 0.0, 1.0]))
        >>> op(torch.tensor([5.0, 5.0, 5.0]))
        tensor([5., 0., 5.])
    """

    def __init__(self, mask: torch.Tensor) -> None:
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary (0 or 1); use a Diagonal operator otherwise")
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out entries of `x` where the mask is 0.

        Example:
            >>> import torch
            >>> op = Masking(torch.tensor([1.0, 0.0, 1.0]))
            >>> op.forward(torch.tensor([5.0, 5.0, 5.0]))
            tensor([5., 0., 5.])
        """
        return x * self.mask.to(x.device)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply `A*`. Masking is self-adjoint, so this is identical to `forward`.

        Example:
            >>> import torch
            >>> op = Masking(torch.tensor([1.0, 0.0, 1.0]))
            >>> op.adjoint(torch.tensor([5.0, 5.0, 5.0]))
            tensor([5., 0., 5.])
        """
        return y * self.mask.to(y.device)


def dot_product_test(
    op: MeasurementOperator,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> float:
    """Verify the adjoint identity `<Ax, y> == <x, A*y>`.

    A wrong adjoint is the most common and most insidious bug in inverse-problem
    code: gradients point in slightly wrong directions, and the reconstruction still
    looks plausible while being incorrect. This test catches it in one line.

    Args:
        op: Operator to check.
        x_shape: Shape of a signal in the domain.
        y_shape: Shape of a measurement in the range.
        generator: Optional RNG for reproducibility.
        device: Target device. Defaults to `pick_device()`.

    Returns:
        Relative error `|<Ax,y> - <x,A*y>| / |<Ax,y>|`. Below `1e-5` means correct.

    Example:
        >>> import torch
        >>> op = Masking(torch.tensor([1.0, 0.0, 1.0]))
        >>> gen = torch.Generator().manual_seed(0)
        >>> error = dot_product_test(op, (3,), (3,), generator=gen, device=torch.device("cpu"))
        >>> bool(error < 1e-5)
        True
    """
    device = device or pick_device()
    x = torch.randn(x_shape, device=device, generator=generator)
    y = torch.randn(y_shape, device=device, generator=generator)

    lhs = torch.sum(op.forward(x) * y).item()
    rhs = torch.sum(x * op.adjoint(y)).item()
    denom = max(abs(lhs), 1e-12)
    return abs(lhs - rhs) / denom

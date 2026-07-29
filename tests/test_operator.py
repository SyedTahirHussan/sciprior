from __future__ import annotations

import torch

from sciprior.inverse import Masking, MeasurementOperator, dot_product_test


class BrokenAdjoint(MeasurementOperator):
    """An operator whose adjoint is deliberately wrong, to prove the test detects it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return 3.0 * y  # should be 2.0


class Slice(MeasurementOperator):
    """Rectangular operator R^n -> R^m: keep the first m entries; adjoint zero-pads back.

    Unlike `Masking`, `forward` and `adjoint` are genuinely different code paths
    (truncation vs. zero-padding) rather than the same expression, and the shapes
    differ (`x_shape != y_shape`), which is the case `dot_product_test`'s separate
    `x_shape`/`y_shape` parameters exist to support -- e.g. sparse Fourier sampling
    in radio interferometry.
    """

    def __init__(self, n: int, m: int) -> None:
        self.n = n
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[: self.m]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n, dtype=y.dtype, device=y.device)
        out[: self.m] = y
        return out


def test_masking_forward_zeroes_unobserved_entries() -> None:
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    op = Masking(mask)
    out = op.forward(torch.tensor([5.0, 5.0, 5.0, 5.0]))
    torch.testing.assert_close(out, torch.tensor([5.0, 0.0, 5.0, 0.0]))


def test_masking_passes_the_dot_product_test() -> None:
    """The adjoint identity <Ax, y> == <x, A*y> must hold to machine precision.

    Getting an adjoint subtly wrong is the single most common bug in inverse-problem
    code, and it produces reconstructions that look plausible while being incorrect.
    Every operator in the programme is required to pass this.

    Note: Masking is a degenerate case for this particular check. Its `forward` and
    `adjoint` are the same expression (`x * mask` vs `y * mask`), so the two sides of
    the dot product are bit-identical float32 computations and the measured error is
    exactly 0.0, not merely below 1e-5 -- this test does not exercise the numerical
    tolerance the 1e-5 threshold exists for. See `test_dot_product_test_detects_a_wrong_adjoint`
    (`BrokenAdjoint`) for the checker's true-negative path, and
    `test_slice_rectangular_operator_passes_the_dot_product_test` (`Slice`) for a
    genuinely different-code-path, non-self-adjoint, rectangular case.
    """
    torch.manual_seed(0)
    op = Masking(torch.randint(0, 2, (64,)).float())
    error = dot_product_test(op, x_shape=(64,), y_shape=(64,), device=torch.device("cpu"))
    assert error < 1e-5


def test_dot_product_test_detects_a_wrong_adjoint() -> None:
    torch.manual_seed(0)
    error = dot_product_test(
        BrokenAdjoint(), x_shape=(32,), y_shape=(32,), device=torch.device("cpu")
    )
    assert error > 0.1


def test_slice_rectangular_operator_passes_the_dot_product_test() -> None:
    """R^10 -> R^4: exercises dot_product_test with x_shape != y_shape.

    <Ax, y> = sum_{i<4} x_i y_i (forward truncates to the first 4 entries, then dots
    with y). <x, A*y> = sum_{i<4} x_i y_i (adjoint zero-pads y back to length 10, so
    only the first 4 entries of x contribute to the dot product). The two sides are
    computed via genuinely different code paths (slicing vs. zero-padding), unlike
    Masking's self-adjoint case above.

    Note: the measured error here is also exactly 0.0 -- zero-padding adds only exact
    float32 zeros, so this test does not exercise rounding tolerance either. What it
    demonstrates that Masking cannot is shape discrimination: x_shape != y_shape, so
    a rhs computed with the wrong operand shape (e.g. op.forward(y) instead of
    op.adjoint(y)) fails loudly with a shape-mismatch error rather than silently
    producing a plausible number.
    """
    torch.manual_seed(0)
    error = dot_product_test(Slice(10, 4), x_shape=(10,), y_shape=(4,), device=torch.device("cpu"))
    assert error < 1e-5


def test_masking_is_idempotent() -> None:
    mask = torch.tensor([1.0, 0.0, 1.0])
    op = Masking(mask)
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(op.forward(op.forward(x)), op.forward(x))


def test_operator_is_callable() -> None:
    op = Masking(torch.ones(3))
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(op(x), op.forward(x))


def test_masking_rejects_non_binary_mask() -> None:
    try:
        Masking(torch.tensor([0.0, 0.5, 1.0]))
    except ValueError as exc:
        assert "binary" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")

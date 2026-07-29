from __future__ import annotations

import math

import pytest
import torch

from sciprior.inverse import Masking, MeasurementOperator, dps_sample
from sciprior.score import VPSDE


class AnalyticGaussianScore:
    """Exact score for N(0, sigma0^2 I) data under a VP-SDE."""

    def __init__(self, sde: VPSDE, sigma0: float = 1.0) -> None:
        self.sde = sde
        self.sigma0 = sigma0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean_coeff, std = self.sde.marginal_prob(torch.ones_like(x), t)
        var = (mean_coeff**2) * (self.sigma0**2) + std**2
        return -x / var


def _setup(n_dim: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    mask = torch.zeros(n_dim)
    mask[: n_dim // 2] = 1.0  # observe the first half
    truth = torch.randn(n_dim)
    y = (truth * mask).unsqueeze(0)
    return mask, truth, y


def test_dps_reconstructs_observed_entries() -> None:
    """Where data exists, the posterior must agree with it.

    This is the weakest possible correctness bar for a posterior sampler, and a
    failure here means the likelihood gradient is not being applied at all.
    """
    sde = VPSDE()
    mask, truth, y = _setup()
    samples = dps_sample(
        AnalyticGaussianScore(sde),
        sde,
        Masking(mask),
        y=y,
        shape=(64, 16),
        sigma_y=0.05,
        n_steps=300,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    observed = mask.bool()
    error = (samples[:, observed].mean(dim=0) - truth[observed]).abs().mean().item()
    assert error < 0.35


def test_dps_falls_back_to_the_prior_where_there_is_no_data() -> None:
    """Unobserved coordinates must revert to the prior: mean ~0, spread ~sigma0.

    A sampler that returns confident values for coordinates it has no information
    about is exactly the hallucination failure mode uvdiff exists to measure.

    This test passes identically with the likelihood term disabled (`step_size=0.0`)
    -- with a diagonal prior the unobserved marginal is unchanged by conditioning.
    Discrimination of the likelihood term is carried by
    `test_dps_reconstructs_observed_entries` and
    `test_dps_uncertainty_is_larger_where_data_is_absent`.
    """
    sde = VPSDE()
    mask, _, y = _setup()
    samples = dps_sample(
        AnalyticGaussianScore(sde, sigma0=1.0),
        sde,
        Masking(mask),
        y=y,
        shape=(512, 16),
        sigma_y=0.05,
        n_steps=300,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(2),
    )
    unobserved = (1 - mask).bool()
    assert samples[:, unobserved].mean().abs().item() < 0.25
    assert 0.6 < samples[:, unobserved].std().item() < 1.5


def test_dps_uncertainty_is_larger_where_data_is_absent() -> None:
    """The central claim of posterior sampling: uncertainty tracks information.

    If this ordering does not hold, the uncertainty maps produced downstream are
    meaningless regardless of how good the reconstructions look.
    """
    sde = VPSDE()
    mask, _, y = _setup()
    samples = dps_sample(
        AnalyticGaussianScore(sde),
        sde,
        Masking(mask),
        y=y,
        shape=(512, 16),
        sigma_y=0.05,
        n_steps=300,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
    )
    per_coord_std = samples.std(dim=0)
    assert per_coord_std[(1 - mask).bool()].mean() > 2.0 * per_coord_std[mask.bool()].mean()


def test_dps_observed_std_stays_within_a_band_of_the_closed_form() -> None:
    """Pins the observed-coordinate posterior WIDTH against the closed-form Gaussian.

    For prior N(0,1) and y = x + n with noise std sigma_y, the true posterior on an
    observed coordinate is N(y/(1+sigma_y^2), sigma_y^2/(1+sigma_y^2)), i.e.
    std = sigma_y / sqrt(1 + sigma_y^2) -- computed here directly, not derived from
    the sampler. DPS is an approximate sampler and its likelihood weighting is a
    deliberate, settled departure from Chung et al.'s residual-norm scaling (see
    dps.py), so it is not expected to match the closed form exactly. Measured at
    these settings, DPS's observed std comes in at ~3.73x the closed form --
    consistent over-dispersion, not collapse.

    This band exists to pin over-dispersion against REGRESSION (in either
    direction), not to certify calibration -- that is spec Sec5.1's job. The upper
    bound catches uncertainty inflation; the lower bound catches uncertainty
    collapse toward false confidence, which is the hallucination failure mode this
    portfolio exists to measure.
    """
    sde = VPSDE()
    mask, _, y = _setup()
    sigma_y = 0.05
    samples = dps_sample(
        AnalyticGaussianScore(sde),
        sde,
        Masking(mask),
        y=y,
        shape=(512, 16),
        sigma_y=sigma_y,
        n_steps=300,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
    )
    observed = mask.bool()
    obs_std = samples.std(dim=0)[observed].mean().item()
    closed_form_std = sigma_y / math.sqrt(1.0 + sigma_y**2)
    ratio = obs_std / closed_form_std
    assert 2.0 < ratio < 6.0


def test_dps_is_reproducible_with_a_seeded_generator() -> None:
    sde = VPSDE()
    mask, _, y = _setup()
    kwargs = dict(
        sde=sde,
        operator=Masking(mask),
        y=y,
        shape=(8, 16),
        sigma_y=0.05,
        n_steps=40,
        device=torch.device("cpu"),
    )
    model = AnalyticGaussianScore(sde)
    a = dps_sample(model, generator=torch.Generator().manual_seed(9), **kwargs)
    b = dps_sample(model, generator=torch.Generator().manual_seed(9), **kwargs)
    torch.testing.assert_close(a, b)


def test_dps_rejects_non_positive_sigma_y() -> None:
    sde = VPSDE()
    mask, _, y = _setup()
    with pytest.raises(ValueError, match="sigma_y"):
        dps_sample(
            AnalyticGaussianScore(sde),
            sde,
            Masking(mask),
            y=y,
            shape=(2, 16),
            sigma_y=0.0,
            n_steps=5,
            device=torch.device("cpu"),
        )


def test_dps_rejects_too_few_steps() -> None:
    sde = VPSDE()
    mask, _, y = _setup()
    with pytest.raises(ValueError, match="n_steps"):
        dps_sample(
            AnalyticGaussianScore(sde),
            sde,
            Masking(mask),
            y=y,
            shape=(2, 16),
            sigma_y=0.05,
            n_steps=1,
            device=torch.device("cpu"),
        )


def test_dps_raises_on_residual_batch_mismatch() -> None:
    """A `y` that broadcasts to the wrong leading dimension must raise, not silently
    misgroup per-sample norms across `residual.reshape(batch, -1)`."""
    sde = VPSDE()
    mask = torch.ones(16)
    y = torch.zeros(2, 1, 16)  # broadcasts against forward output (3, 16) -> (2, 3, 16)
    with pytest.raises(ValueError, match="broadcast against the forward output"):
        dps_sample(
            AnalyticGaussianScore(sde),
            sde,
            Masking(mask),
            y=y,
            shape=(3, 16),
            sigma_y=0.05,
            n_steps=5,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
        )


class _SumOperator(MeasurementOperator):
    """Reduces the last dimension to a single scalar per sample: `(batch, n) -> (batch, 1)`.

    Used only to reproduce the broadcast blow-up in
    `test_dps_raises_on_residual_shape_blowup`: a realistic operator whose forward
    output has a trailing singleton dimension, the exact shape that a
    `shape[0]`-only guard fails to catch when `y` is `(batch,)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y.expand(-1, 16)


def test_dps_raises_on_residual_shape_blowup() -> None:
    """A per-sample scalar `y` of shape `(batch,)` against a `(batch, 1)` forward
    output must raise, not pass a `shape[0] == batch` guard and silently broadcast
    the residual to `(batch, batch)`.

    This is the exact hole a leading-dimension-only check leaves open:
    `y - operator.forward(x0_hat)` broadcasts `(batch,)` against `(batch, 1)` into
    `(batch, batch)`, which still has `shape[0] == batch`, so a guard that only
    inspects the leading dimension passes while `residual.reshape(batch, -1)` mixes
    unrelated samples into the same per-sample norm.
    """
    sde = VPSDE()
    batch = 4
    y = torch.zeros(batch)  # forward(x0_hat) is (batch, 1); this broadcasts to (batch, batch)
    with pytest.raises(ValueError, match="broadcast against the forward output"):
        dps_sample(
            AnalyticGaussianScore(sde),
            sde,
            _SumOperator(),
            y=y,
            shape=(batch, 16),
            sigma_y=0.05,
            n_steps=5,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
        )

from __future__ import annotations

import math

import torch

from sciprior.score import VPSDE, denoising_score_matching_loss, euler_maruyama_sample


class AnalyticGaussianScore:
    """Exact score for data distributed as N(0, sigma0^2 I) under a VP-SDE.

    Because the perturbation kernel of a Gaussian is Gaussian, the true score is
    available in closed form. That makes it possible to test the sampler against
    ground truth rather than against a golden file.

    Note: at `sigma0 = 1.0` this score reduces to exactly `-x` regardless of the
    schedule, because `var = mean_coeff**2 * sigma0**2 + std**2 = 1` identically for
    a VP-SDE (mean_coeff**2 + std**2 == 1 always). A test that only exercises this
    fixture at `sigma0 = 1.0` therefore does *not* exercise `marginal_prob`'s
    schedule at all -- see `test_marginal_prob_matches_the_closed_form_vp_sde_solution`
    for a direct schedule check, and the `sigma0 = 0.5` case of
    `test_sampler_recovers_the_data_distribution_moments` for an end-to-end check
    that genuinely depends on the schedule.
    """

    def __init__(self, sde: VPSDE, sigma0: float = 1.0) -> None:
        self.sde = sde
        self.sigma0 = sigma0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean_coeff, std = self.sde.marginal_prob(torch.ones_like(x), t)
        var = (mean_coeff**2) * (self.sigma0**2) + std**2
        return -x / var


def test_marginal_std_grows_monotonically() -> None:
    sde = VPSDE()
    x = torch.zeros(1, 1)
    stds = [sde.marginal_prob(x, torch.tensor([t]))[1].item() for t in (0.01, 0.3, 0.6, 1.0)]
    assert stds == sorted(stds)
    assert stds[0] < 0.15
    assert stds[-1] > 0.95


def test_marginal_mean_coefficient_decays_to_near_zero() -> None:
    sde = VPSDE()
    x = torch.ones(1, 1)
    m0, _ = sde.marginal_prob(x, torch.tensor([0.01]))
    m1, _ = sde.marginal_prob(x, torch.tensor([1.0]))
    assert m0.item() > 0.99
    assert m1.item() < 0.05


def test_marginal_prob_matches_the_closed_form_vp_sde_solution() -> None:
    """Pin `marginal_prob` against the closed-form VP-SDE solution, computed here
    with its own hard-coded arithmetic rather than by calling `VPSDE._log_mean_coeff`.

    This is deliberately independent of the implementation under test. The other
    tests in this module that exercise `marginal_prob` indirectly -- via
    `AnalyticGaussianScore` at `sigma0 = 1.0` -- cannot catch a schedule bug: at
    `sigma0 = 1.0`, `var = mean_coeff**2 + std**2` collapses to exactly `1`
    identically for *any* value the schedule produces, so a corrupted
    `_log_mean_coeff` (e.g. a stray factor of 2 in the beta integral) yields
    bit-identical sampler output and passes every other test in this file. Building
    the expectation from the closed-form formula, in the test body, closes that gap.
    """
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    x = torch.tensor([[2.0], [-1.5], [0.5]])
    for t_val in (0.1, 0.5, 1.0):
        t = torch.full((3,), t_val)
        # -0.5 * integral_0^t beta(s) ds for beta(s) = beta_min + s * (beta_max - beta_min).
        lmc = -0.5 * (0.1 * t_val + 0.5 * t_val**2 * (20.0 - 0.1))
        expected_mean = math.exp(lmc) * x
        expected_std = math.sqrt(1.0 - math.exp(2.0 * lmc)) * torch.ones(3, 1)

        mean, std = sde.marginal_prob(x, t)
        torch.testing.assert_close(mean, expected_mean, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(std, expected_std, rtol=1e-6, atol=1e-6)


def test_sde_returns_drift_and_diffusion_with_correct_shapes() -> None:
    sde = VPSDE()
    x = torch.randn(4, 3)
    drift, diffusion = sde.sde(x, torch.full((4,), 0.5))
    assert drift.shape == x.shape
    assert diffusion.shape == (4,)


def test_dsm_loss_is_lower_for_the_true_score_than_for_a_wrong_one() -> None:
    """The exact score must beat a deliberately wrong one. This catches sign and
    scale errors in the loss that a smoke test would silently accept."""
    torch.manual_seed(0)
    sde = VPSDE()
    x0 = torch.randn(512, 8)
    truth = AnalyticGaussianScore(sde, sigma0=1.0)
    wrong = lambda x, t: -truth(x, t)  # noqa: E731

    # Both calls use freshly, identically seeded generators so `t` and `noise` are
    # drawn identically for `good` and `bad` -- a genuinely paired comparison.
    good = denoising_score_matching_loss(
        truth, x0, sde, generator=torch.Generator().manual_seed(0)
    ).item()
    bad = denoising_score_matching_loss(
        wrong, x0, sde, generator=torch.Generator().manual_seed(0)
    ).item()

    assert good < bad


def test_sampler_recovers_the_data_distribution_moments() -> None:
    """Sampling with the analytic score must reproduce N(0, sigma0^2).

    The `sigma0 = 1.0` case is an end-to-end check of the reverse-time drift and
    the sampler, but *not* of the schedule: at `sigma0 = 1.0`,
    `AnalyticGaussianScore` reduces to exactly `-x` for any schedule (see the note
    on that class), so this case alone cannot catch a sign or scale error in
    `marginal_prob`. The `sigma0 = 0.5` case breaks that cancellation --
    `var = mean_coeff**2 * 0.25 + std**2` is no longer identically `1` -- and so is
    the genuine end-to-end check of the schedule together with the reverse-time
    drift and the sampler. If any of the three has a sign or scale error, the
    recovered standard deviation in the `sigma0 = 0.5` case will be visibly wrong.
    """
    torch.manual_seed(0)
    sde = VPSDE()
    sigma0 = 1.0
    model = AnalyticGaussianScore(sde, sigma0=sigma0)

    samples = euler_maruyama_sample(
        model, sde, shape=(4096, 1), n_steps=800, device=torch.device("cpu")
    )

    assert samples.shape == (4096, 1)
    assert abs(samples.mean().item()) < 0.12
    assert abs(samples.std().item() - sigma0) < 0.15

    # sigma0 != 1.0: the schedule no longer cancels out of the analytic score, so
    # this sub-case gives the test real discriminating power over the schedule.
    # Measured on this exact seed: mean error ~= 0.0045, std error ~= 0.0023.
    # Tolerances below give ~4-5x margin over those measured errors.
    torch.manual_seed(0)
    sde = VPSDE()
    sigma0 = 0.5
    model = AnalyticGaussianScore(sde, sigma0=sigma0)

    samples = euler_maruyama_sample(
        model, sde, shape=(4096, 1), n_steps=800, device=torch.device("cpu")
    )

    assert samples.shape == (4096, 1)
    assert abs(samples.mean().item()) < 0.02
    assert abs(samples.std().item() - sigma0) < 0.012


def test_sampler_is_reproducible_with_a_seeded_generator() -> None:
    sde = VPSDE()
    model = AnalyticGaussianScore(sde)
    kwargs = dict(shape=(16, 2), n_steps=50, device=torch.device("cpu"))

    a = euler_maruyama_sample(model, sde, generator=torch.Generator().manual_seed(7), **kwargs)
    b = euler_maruyama_sample(model, sde, generator=torch.Generator().manual_seed(7), **kwargs)

    torch.testing.assert_close(a, b)

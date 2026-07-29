import pytest
import torch

import sciprior
from sciprior._device import pick_device


def test_version_is_exposed() -> None:
    assert isinstance(sciprior.__version__, str)
    assert sciprior.__version__.count(".") == 2


def test_pick_device_returns_a_torch_device() -> None:
    assert isinstance(pick_device(), torch.device)


def test_pick_device_never_returns_cuda() -> None:
    # This portfolio targets Apple Silicon and CPU CI. A cuda device here means
    # someone hard-coded an assumption that will break on both machines.
    assert pick_device().type in {"mps", "cpu"}


def test_pick_device_honours_explicit_cpu_request() -> None:
    assert pick_device(prefer="cpu").type == "cpu"


def test_pick_device_rejects_cuda() -> None:
    # pick_device() is the only place in the package that selects hardware, and its
    # guarantee is "never cuda" -- torch.device("cuda") constructs fine even with no
    # CUDA present, so this must be rejected explicitly rather than trusted to torch.
    with pytest.raises(ValueError):
        pick_device(prefer="cuda")

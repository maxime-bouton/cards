r"""Testing torch interface to restart random number generators."""

from mcmc.backend import bm
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.torch

# FIXME: dirty test writing, to be revised thoroughly


@pytest.fixture
def shape():
    return (512, 512)


@pytest.fixture
def size():
    return 10


@pytest.fixture
def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture
def n_trials():
    return 10


@pytest.mark.env("serial-gpu")
def test_restart_gpu_rng(shape, n_trials, device):
    bm.set_backend("cupy")
    from mcmc.backend import xp

    rng = torch.Generator(device=device).manual_seed(1234)

    for i in range(n_trials):
        A = xp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng,
            )
        )

    rng2 = torch.Generator(device=device).manual_seed(1234)

    # TODO: new interface since torch=2.9: .set_state()/.get_state() on CPU, .graphsafe_get_state()/graphsafe_set_state() on GPU
    # see: https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html#torch.Generator
    # if device.type == "cpu":
    #     state = rng.get_state()
    #     rng2.set_state(state)
    # else:  # 'cuda'
    #     state = rng.graphsafe_get_state()
    #     rng2.graphsafe_set_state(state)

    state = rng.get_offset()
    rng2.set_offset(state)

    check = np.zeros(n_trials)

    for i in range(n_trials):
        A = xp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng,
            )
        )
        B = xp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng2,
            )
        )

        check[i] = xp.allclose(A, B)

    assert np.all(check)


if __name__ == "__main__":
    shape = (512, 512)
    n_trials = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_restart_gpu_rng(shape, n_trials, device)

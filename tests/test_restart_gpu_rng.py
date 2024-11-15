r"""Testing torch interface to restart random number generators."""

import cupy as cp
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.torch


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


def test_restart_gpu_rng(shape, n_trials, device):
    rng = torch.Generator(device=device).manual_seed(1234)

    for i in range(n_trials):
        A = cp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng,
            )
        )

    rng2 = torch.Generator(device=device).manual_seed(1234)

    offset = rng.get_offset()
    rng2.set_offset(offset)

    check = np.zeros(n_trials)

    for i in range(n_trials):
        A = cp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng,
            )
        )
        B = cp.asarray(
            torch.normal(
                torch.zeros(shape, device=device),
                torch.ones(shape, device=device),
                generator=rng2,
            )
        )

        check[i] = cp.allclose(A, B)

    assert np.all(check)


if __name__ == "__main__":
    shape = (512, 512)
    n_trials = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_restart_gpu_rng(shape, n_trials, device)

r"""Testing torch interface to restart random number generators."""

import cupy as cp
import numpy as np
import torch


import pytest


@pytest.fixture
def shape():
    return np.asarray([512, 512], dtype=int)


@pytest.fixture
def size():
    return 10


def test_restart_gpu_rng(shape, size):
    rng = torch.Generator(device="cuda").manual_seed(1234)

    shape = [512, 512]
    n = size

    A = cp.zeros(shape)

    for i in range(n):
        A = cp.asarray(
            torch.normal(
                torch.as_tensor(cp.zeros_like(A), device="cuda"),
                torch.as_tensor(cp.ones_like(A), device="cuda"),
                generator=rng,
            )
        )

    rng2 = torch.Generator(device="cuda").manual_seed(1234)

    offset = rng.get_offset()
    rng2.set_offset(offset)
    B = cp.zeros(shape)

    check = np.zeros(10)

    for i in range(10):
        A = cp.asarray(
            torch.normal(
                torch.as_tensor(cp.zeros_like(A), device="cuda"),
                torch.as_tensor(cp.ones_like(A), device="cuda"),
                generator=rng,
            )
        )
        B = cp.asarray(
            torch.normal(
                torch.as_tensor(cp.zeros_like(B), device="cuda"),
                torch.as_tensor(cp.ones_like(B), device="cuda"),
                generator=rng2,
            )
        )

        check[i] = cp.allclose(A, B)

    assert np.all(check)


if __name__ == "__main__":
    shape = np.asarray([512, 512], dtype=int)
    size = 10
    test_restart_gpu_rng(shape, size)

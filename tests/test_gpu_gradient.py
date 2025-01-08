r"""Test that checks that the computation done on GPU correspond to those on cpu for the gradient operator."""

import cupy as cp
import numpy as np
import pytest

from mcmc.operators.gpu.gradient import GpuGradient2d as gpu_grad
from mcmc.operators.gradient import Gradient2d as cpu_grad


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def dims():
    return np.asarray([150, 100], dtype=int)


def test_gpu_grad(seed, dims):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(dims)
    Y = rng.standard_normal((2, *dims))

    cpu_op = cpu_grad(dims)
    gpu_op = gpu_grad(dims)

    cpu_forward = cpu_op.forward(X)
    gpu_forward = gpu_op.forward(cp.asarray(X))

    cpu_adj = cpu_op.adjoint(Y)
    gpu_adj = gpu_op.adjoint(cp.asarray(Y))

    assert cp.allclose(cp.asarray(cpu_forward), gpu_forward)
    assert cp.allclose(cp.asarray(cpu_adj), gpu_adj)

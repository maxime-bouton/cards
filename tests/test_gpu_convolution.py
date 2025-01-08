r"""Test that checks if the computation done on GPU correspond to those done on CPU for the convolution product."""

import cupy as cp
import numpy as np

from mcmc.operators.serial_convolution import SerialConvolution
from mcmc.operators.gpu.convolution import GpuConvolution
import pytest


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def img_dims():
    return np.asarray([128, 64])


@pytest.fixture
def kernel_dims():
    return np.asarray([8, 8])


def test_cpu_vs_gpu(seed, img_dims, kernel_dims):
    rng = np.random.default_rng(seed)

    img = rng.standard_normal(img_dims)
    kernel = rng.standard_normal(kernel_dims)

    convo_dims = (
        np.asarray(img.shape) + np.asarray(kernel.shape) - np.ones_like(img_dims)
    )

    cpu_convo_handler = SerialConvolution(img_dims, kernel, convo_dims)
    gpu_convo_handler = GpuConvolution(img_dims, cp.asarray(kernel), tuple(convo_dims))

    cpu_convo = cpu_convo_handler.forward(img)
    gpu_convo = gpu_convo_handler.forward(cp.asarray(img))

    cpu_adjoint = cpu_convo_handler.adjoint(img)
    gpu_adjoint = gpu_convo_handler.adjoint(cp.asarray(img))

    assert cp.allclose(cp.asarray(cpu_convo), gpu_convo)
    assert cp.allclose(cp.asarray(cpu_adjoint), gpu_adjoint)

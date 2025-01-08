# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np
import pytest
import scipy.signal as sg

import mcmc.operators.serial_convolution as docv
from mcmc.operators.serial_convolution import generate_2d_gaussian_kernel
from mcmc.operators.padding import pad_array


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def image_size():
    return np.array([10, 7], "i")


@pytest.fixture
def x(image_size, rng):
    return (1 + 1j) * rng.normal(loc=0.0, scale=1.0, size=image_size)


@pytest.fixture
def kernel():
    return generate_2d_gaussian_kernel(3, 0.1)


@pytest.fixture
def data_size(image_size, kernel):
    return image_size + np.array(kernel.shape, dtype="i") - 1


@pytest.fixture
def lc_fft_kernel(kernel, data_size):
    return np.fft.rfftn(kernel, data_size, axes=range(len(kernel.shape)))


@pytest.fixture
def lc_fft_kernel_c(kernel, data_size):
    return np.fft.fftn(kernel, data_size, axes=range(len(kernel.shape)))


def test_fft_conv_linear_convolution_1d(x, kernel, image_size, data_size):
    """Check :func:`dsgs.operators.convolutions.fft_conv` against
    `scipy.signal.convolve` on a 1d array."""
    ft_kernel = np.fft.fftn(
        kernel[0, :], s=[data_size[1]], axes=range(len(kernel[0, :].shape))
    )
    Hx = docv.fft_conv(x[0, :], ft_kernel, shape=data_size[1])
    Hx_scipy = sg.convolve(x[0, :], kernel[0, :], mode="full")
    assert np.allclose(Hx, Hx_scipy)


def test_fft_conv_linear_convolution_real(
    x, kernel, lc_fft_kernel, image_size, data_size
):
    """Check :func:`dsgs.operators.convolutions.fft_conv` against
    `scipy.signal.convolve` on a real-type array."""
    Hx = docv.fft_conv(np.real(x), lc_fft_kernel, shape=data_size)
    Hx_scipy = sg.convolve(np.real(x), kernel, mode="full")
    assert np.allclose(Hx, Hx_scipy)


def test_fft_conv_linear_convolution_complex(
    x, kernel, lc_fft_kernel_c, image_size, data_size
):
    """Check :func:`dsgs.operators.convolutions.fft_conv` against
    `scipy.signal.convolve` on a complex-type array."""
    Hx = docv.fft_conv(x, lc_fft_kernel_c, shape=data_size)
    Hx_scipy = sg.convolve(x, kernel, mode="full")
    assert np.allclose(Hx, Hx_scipy)


def test_fft_conv_linear_convolution_adjoint(
    x, kernel, lc_fft_kernel_c, image_size, data_size, rng
):
    """Check the correctness of the adjoint convolution operator (linear
    convolution).
    """
    Hx = docv.fft_conv(x, lc_fft_kernel_c, shape=data_size)
    y = (1 + 1j) * rng.standard_normal(data_size)
    Hadj_y = docv.fft_conv(y, np.conj(lc_fft_kernel_c), shape=data_size)[
        : image_size[0], : image_size[1]
    ]

    hp1 = np.sum(np.conj(Hx) * y)
    hp2 = np.sum(np.conj(x) * Hadj_y)
    assert np.isclose(hp1, hp2)


def test_fft_conv_linear_convolution_adjoint_real(
    x, kernel, lc_fft_kernel, image_size, data_size, rng
):
    """Check :func:`dsgs.operators.convolutions.fft_conv` against
    `scipy.signal.convolve` on a real-type array."""
    Hx = docv.fft_conv(np.real(x), lc_fft_kernel, shape=data_size)
    Hx_scipy = sg.convolve(np.real(x), kernel, mode="full")
    assert np.allclose(Hx, Hx_scipy)

    y = rng.standard_normal(data_size)
    Hadj_y = docv.fft_conv(y, np.conj(lc_fft_kernel), shape=data_size)[
        : image_size[0], : image_size[1]
    ]

    hp1 = np.sum(np.conj(Hx) * y)
    hp2 = np.sum(np.conj(np.real(x)) * Hadj_y)
    assert np.isclose(hp1, hp2)


def test_fft2_conv_throws_exceptions():
    """Check exceptions returned by
    :func:`dsgs.operators.convolutions.fft2_conv`.
    """
    # * shape has the wrong number of elements (1 instead of 2)
    with pytest.raises(ValueError) as excinfo:
        docv.fft2_conv(np.ones((5, 2)), np.ones((2, 2)), shape=[5])
    assert "x.shape and shape must have the same length" in str(excinfo.value)

    # * kernel has the wrong number of dimensions (1 instead of 2)
    with pytest.raises(ValueError) as excinfo:
        docv.fft2_conv(np.ones((5, 2)), np.ones(2), shape=[5, 6])
    assert "x.shape and h.shape must have the same length" in str(excinfo.value)


def test_fft2_conv_linear_convolution(x, kernel, image_size, data_size):
    """Check consistency between :func:`dsgs.operators.convolutions.fft2_conv`
    and :func:`dsgs.operators.convolutions.fft_conv` for a linear convolution.
    """
    # * complex case
    Hxc, fft_kernel_c = docv.fft2_conv(x, kernel, shape=data_size)
    Hxc_ref = docv.fft_conv(x, fft_kernel_c, shape=data_size)
    assert Hxc.dtype.kind == "c"
    assert np.allclose(Hxc, Hxc_ref)

    # * real case
    x_ = np.real(x)
    padded_kernel = pad_array(kernel, data_size, padmode="after")
    Hx, fft_kernel = docv.fft2_conv(x_, padded_kernel, shape=data_size)
    Hx_ref = docv.fft_conv(x_, fft_kernel, shape=data_size)
    assert not Hx.dtype.kind == "c"
    assert np.allclose(Hx, Hx_ref)


def test_fft2_conv_circular_convolution(x, kernel, image_size, rng):
    """Check consistency between :func:`dsgs.operators.convolutions.fft2_conv`
    and :func:`dsgs.operators.convolutions.fft_conv` for a circular convolution.
    """
    # * complex case
    Hxc, fft_kernel_c = docv.fft2_conv(x, kernel, shape=None)
    Hxc_ref = docv.fft_conv(x, fft_kernel_c, shape=image_size)
    assert Hxc.shape == x.shape
    assert Hxc.dtype.kind == "c"
    assert np.allclose(Hxc, Hxc_ref)

    # * real case
    x_ = np.real(x)
    padded_kernel = pad_array(kernel, image_size, padmode="after")
    Hx, fft_kernel = docv.fft2_conv(x_, padded_kernel, shape=image_size)
    Hx_ref = docv.fft_conv(x_, fft_kernel, shape=image_size)
    assert Hx.shape == x.shape
    assert not Hx.dtype.kind == "c"
    assert np.allclose(Hx, Hx_ref)


def test_linear_convolution(x, kernel, image_size, data_size, rng):
    """Check the correctness of the adjoint convolution operator between
    :func:`dsgs.operators.convolutions.linear_convolution` and :func:`dsgs.operators.convolutions.adjoint_linear_convolution`
    """
    Hx = docv.linear_convolution(x, kernel, mode="symmetric")
    y = (1 + 1j) * rng.standard_normal(data_size)
    Hadj_y = docv.adjoint_linear_convolution(y, kernel, mode="symmetric")

    hp1 = np.sum(np.conj(Hx) * y)
    hp2 = np.sum(np.conj(x) * Hadj_y)
    assert np.isclose(hp1, hp2)

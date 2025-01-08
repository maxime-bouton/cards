"""FFT-based convolution operator and helper functions."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

# TODO: cleanup examples / playground with boundary conditions
# FIXME: keep kernel out of the class? (e.g., for blind deconvolution)

import numpy as np
import scipy

from mcmc.operators.linear_operator import LinearOperator
from mcmc.operators.padding import adjoint_padding, pad_array_nd


def generate_2d_gaussian_kernel(kernel_size, kernel_std):
    r"""Generate a square normalized 2D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of one dimension of the kernel.
    kernel_std : float
        Standard deviation of the Gaussian kernel.

    Note
    ----
    Equivalent to the ``fspecial('gaussian', ...)`` function in Matlab.

    Returns
    -------
    h : numpy.ndarray
        Square Gaussian kernel with :math:`\|h\|_1 = 1`.
    """
    # equivalent to fspecial('gaussian', ...) in Matlab
    w = scipy.signal.windows.gaussian(kernel_size, kernel_std)
    h = w[:, np.newaxis] * w[np.newaxis, :]
    h = h / np.sum(h)
    return h


def fft2_conv(x, h, shape=None):
    r"""FFT-based 2d convolution.

    Convolve the array ``x`` with the 2d kernel ``h`` using the FFT algorithm.
    Performs linear or circular convolution depending on the padding needed to
    reach the desired size ``shape``.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (of size :math:`N`).
    h : numpy.ndarray
        Input convolution kernel (of size :math:`M`).
    shape : tuple, int, optional
        Desired convolution size (:math:`K \geq \max \{ N, M \}`), by default
        None.

    Returns
    -------
    y : numpy.ndarray
        Output convolution.
    fft_h : numpy.ndarray
        Fourier transform of the convolution kernel ``h`` (of size :math:`K`).

    Raises
    ------
    ValueError
        ``x.shape`` and ``shape`` must have the same length.
    ValueError
        ``x.shape`` and ``h.shape`` must have the same length.

    Note
    ----
    This function does not allow the adjoint convolution operator to be easily
    encoded. See :func:`dsgs.operators.convolutions.fft_conv` instead.
    """
    if shape is None:
        shape = x.shape

    if not len(x.shape) == len(shape):
        raise ValueError("x.shape and shape must have the same length")

    if not len(h.shape) == len(shape):
        raise ValueError("x.shape and h.shape must have the same length")

    if (x.dtype.kind == "c") or (h.dtype.kind == "c"):
        fft_h = np.fft.fft2(h, shape)
        y = np.fft.ifft2(fft_h * np.fft.fft2(x, shape))  # cropping handled separately
    else:
        fft_h = np.fft.rfft2(h, shape)
        y = np.fft.irfft2(fft_h * np.fft.rfft2(x, shape), shape)

    return y, fft_h


def fft_conv(x, fft_h, shape):
    r"""FFT-based nd convolution.

    Convolve the array ``x`` with the kernel of Fourier transform ``fft_h``
    using the FFT. Performs linear or circular convolution depending on
    the 0-padding initially adopted for ``fft_h``.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (of size :math:`N`).
    fft_h : numpy.ndarray
        Input kernel (of size
        :math:`\lfloor K/2 \rfloor + 1` if real, :math:`K` otherwise).
    shape : tuple[int]
        Full shape of the convolution (referred to as :math:`K` above).

    Returns
    -------
    y : numpy.ndarray
        Convolution results.
    """
    # turn shape into a list if only given as a scalar
    if np.isscalar(shape):
        shape_ = [shape]
    else:
        shape_ = shape
    if x.dtype.kind == "c":
        y = np.fft.ifftn(fft_h * np.fft.fftn(x, shape_, axes=range(len(shape_))))
    else:  # assuming h is a real kernel as well
        y = np.fft.irfftn(
            fft_h * np.fft.rfftn(x, shape_, axes=range(len(shape_))),
            shape_,
            axes=range(len(shape_)),
        )

    return y


def linear_convolution(x, h, mode="constant"):
    """Multi-dimensional linear convolution (i.e., with zero-padding
    boundary condition).

    Parameters
    ----------
    x : numpy.ndarray
        Input array (of size :math:`N`).
    h : numpy.ndarray
        Input kernel (of size :math:`M`).

    Returns
    -------
    y : numpy.ndarray
        Convolution result (of size :math:`M + N - 1`).

    Note
    ----
    The function `scipy.ndimage.convolve` produces an output of the same
    size as the input (truncation implictly operated).
    """
    lsize = np.zeros(len(h.shape), dtype="i")
    rsize = np.array(h.shape, dtype="i") - 1
    y = pad_array_nd(x, lsize, rsize, mode="constant")
    return scipy.ndimage.convolve(y, h, mode="constant", cval=0.0)
    # y = pad_array_nd(x, lsize, rsize, mode=mode)
    # return scipy.ndimage.convolve(y, h, mode="constant", cval=0.0)


def adjoint_linear_convolution(y, h, mode="constant"):
    """Adjoint multi-dimensional linear convolution (i.e., with zero-padding
    boundary condition).

    Parameters
    ----------
    y : numpy.ndarray
        Input array (of size :math:`M + N - 1`).
    h : numpy.ndarray
        Input kernel (of size :math:`M`).

    Returns
    -------
    x : numpy.ndarray
        Adjoint convolution result (of size :math:`N`).
    """
    lsize = np.zeros(len(h.shape), dtype="i")
    rsize = 2 * (np.array(h.shape, dtype="i") - 1)
    yp = pad_array_nd(y, lsize, np.array(h.shape, dtype="i") - 1, mode="constant")
    x = scipy.ndimage.convolve(
        yp, np.conj(np.flip(h, axis=None)), mode="constant", cval=0.0
    )
    # ! np.flip(h, axis=None) flips all axes
    return adjoint_padding(x, lsize, rsize, mode="constant")

    # ! symmetric bd condition
    # ! not functional for now
    # lsize = (np.array(h.shape, dtype="i") - 1)
    # rsize = lsize
    # yp = pad_array_nd(y, np.zeros(len(h.shape), dtype="i"), rsize, mode="constant")
    # x = scipy.ndimage.convolve(yp, np.conj(np.flip(h, axis=None)), mode='constant', cval=0.0)
    # # ! np.flip(h, axis=None) flips all axes
    # return adjoint_padding(x, lsize, rsize, mode=mode)


class SerialConvolution(LinearOperator):
    r"""Serial (FFT-based) convolution operator.

    Attributes
    ----------
    image_size : numpy.ndarray[int], of size ``d``
        Full image size.
    kernel : numpy.ndarray
        Input kernel. The array should have ``d`` axis, such that
        ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
    data_size : empty numpy.ndarray[int], of size ``d``
        Full data size.
        - If ``data_size == image_size``: circular convolution;
        - If ``data_size == image_size + kernel_size - 1``: linear convolution.
    """

    # TODO: make sure the interface works for both linear and circular
    # TODO- convolutions
    def __init__(
        self,
        image_size,
        kernel,
        data_size,
    ):
        r"""SerialConvolution constructor.

        Parameters
        ----------
        image_size : numpy.ndarray[int], of size ``d``
            Full image size.
        kernel : numpy.ndarray[float]
            Input kernel. The array should have ``d`` axis, such that
            ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
        data_size : numpy.ndarray[int], of size ``d``
            Full data size.
            - If ``data_size == image_size``: circular convolution;
            - If ``data_size == image_size + kernel_size - 1``: linear convolution.
        fft_kernel : numpy.ndarray
            Fourier transform of the known convolution kernel.
        valid_coefficients : Slice
            Slice object to retrieve valid coefficients after applying the
            adjoint convolution operator.

        Raises
        ------
        ValueError
            ``image_size`` and ``data_size`` must have the same number of
            elements.
        ValueError
            ``kernel`` should have ``ndims = len(image_size)`` dimensions.
        TypeError
            Only real-valued kernel supported.

        Note
        ----
        Setting ``data_size`` to the same value as ``image_size`` results in a
        circular convolution.
        """
        if not image_size.size == data_size.size:
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )
        super(SerialConvolution, self).__init__(image_size, data_size)

        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")

        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")

        self.kernel = kernel
        self.fft_kernel = np.fft.rfftn(
            self.kernel, self.data_size, axes=range(len(self.data_size))
        )
        self.valid_coefficients = tuple(
            [np.s_[: self.image_size[d]] for d in range(self.ndims)]
        )

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : numpy.ndarray[float]
            Input array (image space).

        Returns
        -------
        numpy.ndarray
            Convolution result (direct operator).
        """
        return fft_conv(input_image, self.fft_kernel, self.data_size)

    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : numpy.ndarray[float]
            Input array (data space).

        Returns
        -------
        numpy.ndarray
            Convolution result (adjoint operator).
        """
        return fft_conv(input_data, np.conj(self.fft_kernel), self.data_size)[
            self.valid_coefficients
        ]

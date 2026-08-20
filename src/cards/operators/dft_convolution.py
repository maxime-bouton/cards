"""Serial implementation of an FFT-based convolution operator."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from collections.abc import Sequence

import cards.backend as xp
from cards.operators.linear_operator import LinearOperator


def fft_conv(x: xp.ndarray, fft_h: xp.ndarray, shape) -> xp.ndarray:
    r"""FFT-based multi-dimensional convolution.

    Convolve the array ``x`` with the kernel of Fourier transform ``fft_h``
    using the FFT. Performs linear or circular convolution depending on
    the 0-padding initially adopted for ``fft_h``.

    Parameters
    ----------
    x : xp.ndarray
        Input array (of size :math:`N`).
    fft_h : xp.ndarray
        Input kernel (of size
        :math:`\lfloor K/2 \rfloor + 1` if real, :math:`K` otherwise).
    shape : tuple[int, ...]
        Full shape of the convolution (referred to as :math:`K` above).

    Returns
    -------
    y : xp.ndarray
        Convolution result.
    """
    # turn shape into a list if only given as a scalar
    if xp.isscalar(shape):
        shape_ = [shape]
    else:
        shape_ = shape
    if x.dtype.kind == "c":
        y = xp.fft.ifftn(fft_h * xp.fft.fftn(x, shape_, axes=range(len(shape_))))
    else:  # assuming h is a real kernel as well
        y = xp.fft.irfftn(
            fft_h * xp.fft.rfftn(x, shape_, axes=range(len(shape_))),
            shape_,
            axes=range(len(shape_)),
        )

    return y


class DftConvolution(LinearOperator):
    r"""FFT-based convolution operator.

    Attributes
    ----------
    kernel : xp.ndarray
        Input kernel. The array should have ``d`` axis, such that
        ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
    fft_kernel : ndarray
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
    The class implements either a linear or a circular comvolution operaotr, depending on the value speficied for ``data_shape``
        - if ``data_shape[k] == image_shape[k] for k in range(len(image_shape))``: circular convolution;
        - if ``data_shape[k] == image_shape[k] + kernel.shape[k] - 1 for k in range(len(image_shape))``: linear convolution.
    See :func:`cards.operators.dft_convolution.fft_conv`
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        data_shape: Sequence[int],
        kernel,
    ):
        super().__init__(image_shape, data_shape)
        if not len(self.image_shape) == len(self.data_shape):
            raise ValueError(
                "image_shape and data_shape must have the same number of elements"
            )

        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")

        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")

        self.kernel = kernel
        self.fft_kernel = xp.fft.rfftn(
            a=self.kernel, s=self.data_shape, axes=range(len(self.image_shape))
        )
        self.valid_coefficients = tuple(
            [xp.s_[: self.image_shape[d]] for d in range(self.ndims)]
        )

    def forward(self, image: xp.ndarray, op=None) -> xp.ndarray:
        return fft_conv(image, self.fft_kernel, self.data_shape)

    def adjoint(self, data: xp.ndarray, adjoint_op=None) -> xp.ndarray:
        return fft_conv(data, xp.conj(self.fft_kernel), self.data_shape)[
            self.valid_coefficients
        ]

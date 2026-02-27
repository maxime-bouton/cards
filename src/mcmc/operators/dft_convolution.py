"""Implementation of the convolution product as a linear operator, computed via the DFT.
The computations can be done either on CPU or GPU depending on the settings.
"""

from mcmc.backend import xp
from mcmc.operators.linear_operator import LinearOperator
from mcmc.operators.utils_convolution import fft_conv


class DftConvolution(LinearOperator):
    r"""Convolution operator on GPU (FFT-based).

    Attributes
    ----------
    image_size : numpy.ndarray[int], of size ``d``
        Full image size.
    kernel : cupy.ndarray
        Input kernel. The array should have ``d`` axis, such that
        ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
    data_size : tuple, of size ``d``
        Full data size.
        - If ``data_size == image_size``: circular convolution;
        - If ``data_size == image_size + kernel_size - 1`` for the spatial axis: linear convolution.
    """

    def __init__(
        self,
        image_size,
        kernel,
        data_size,
    ):
        r"""GpuConvolution constructor.

        Parameters
        ----------
        image_size : ndarray[int], of size ``d``
            Full image size.
        kernel : ndarray[float]
            Input kernel. The array should have ``d`` axis, such that
            ``kernel.shape[i] < image_size[i]`` for ``i in range(d)``.
        data_size : ndarray[int] | tuple[int], of size ``d``
            Full data size.
            - If ``data_size == image_size``: circular convolution;
            - If ``data_size == image_size + kernel_size - 1``: linear convolution.
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
        Setting ``data_size`` to the same value as ``image_size`` results in a
        circular convolution.
        """
        # FIXME: size expected as a numpy array, not a tuple (given the current use throughout the library)! (currently inconsistent with what is required from image_size)
        if not isinstance(data_size, tuple):
            data_size = tuple(data_size)
        if not image_size.size == len(data_size):
            raise ValueError(
                "image_size and data_size must have the same number of elements"
            )
        super(DftConvolution, self).__init__(image_size, data_size)

        if not len(kernel.shape) == self.ndims:
            raise ValueError("kernel should have ndims = len(image_size) dimensions")

        if kernel.dtype.kind == "c":
            raise TypeError("only real-valued kernel supported")

        self.kernel = kernel
        self.fft_kernel = xp.fft.rfftn(self.kernel, self.data_size)
        self.valid_coefficients = tuple(
            [xp.s_[: self.image_size[d]] for d in range(self.ndims)]
        )

    def forward(self, input_image: xp.ndarray) -> xp.ndarray:
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space).

        Parameters
        ----------
        input_image : ndarray[float]
            Input array (image space).

        Returns
        -------
        ndarray
            Convolution result (direct operator).
        """
        return fft_conv(input_image, self.fft_kernel, self.data_size)

    def adjoint(self, input_data: xp.ndarray) -> xp.ndarray:
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space).

        Parameters
        ----------
        input_data : ndarray[float]
            Input array (data space).

        Returns
        -------
        numpy.ndarray
            Convolution result (adjoint operator).
        """
        return fft_conv(input_data, xp.conj(self.fft_kernel), self.data_size)[
            self.valid_coefficients
        ]

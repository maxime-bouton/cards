from mcmc.backend import xp


def fft_conv(x: xp.ndarray, fft_h: xp.ndarray, shape: tuple[int]) -> xp.ndarray:
    r"""FFT-based nd convolution.

    Convolve the array ``x`` with the kernel of Fourier transform ``fft_h``
    using the FFT. Performs linear or circular convolution depending on
    the 0-padding initially adopted for ``fft_h``.

    Parameters
    ----------
    x : ndarray
        Input array (of size :math:`N`).
    fft_h : ndarray
        Input kernel (of size
        :math:`\lfloor K/2 \rfloor + 1` if real, :math:`K` otherwise).
    shape : tuple[int]
        Full shape of the convolution (referred to as :math:`K` above).

    Returns
    -------
    ndarray
        Convolution results.
    """
    # turn shape into a list if only given as a scalar
    shape_ = [shape] if xp.isscalar(shape) else shape

    # use the appropriate FFT/IFFT function based on the data type
    fft = xp.fft.fftn if x.dtype.kind == "c" else xp.fft.rfftn
    ifft = xp.fft.ifftn if x.dtype.kind == "c" else xp.fft.irfftn
    return ifft(fft_h * fft(x, shape_), shape_)

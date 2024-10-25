"""Helper functions to implement the FFT-based convolution operator."""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np
import scipy

from dsgs.operators.padding import adjoint_padding, pad_array_nd


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
        y = np.fft.ifftn(fft_h * np.fft.fftn(x, shape_))
    else:  # assuming h is a real kernel as well
        y = np.fft.irfftn(fft_h * np.fft.rfftn(x, shape_), shape_)

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


# TODO: write a generic version
# direct: padding, conv. in valid mode
# Hy_ = sg.convolve2d(y_, h, boundary="fill", mode="full")
# Hadj_y_ = adjoint_padding(Hy_, ext_size, ext_size, mode="symmetric")

# same kind for symmetric when not based on fft (overlap-add for distributed version)


# ! to be made more generic (quick test for now)
# TODO: adjoint of a convolution operator with any boundary extension involves
# the adjoint of the circular convolution and the adjoint of the padding
# operator
# def adjoint_conv(x, h, shape):
#     """Adjoint of the linear convolution operator.

#     Parameters
#     ----------
#     x : _type_
#         _description_
#     h : _type_
#         _description_
#     shape : _type_
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     Hx = sg.convolve2d(x, np.flip(h, axis=None), boundary="fill", mode="full")
#     # ! np.flip(m, axis=None) flips all axes
#     s = tuple([np.s_[: shape[k]] for k in range(len(shape))])
#     return Hx[s]


if __name__ == "__main__":
    # # TODO: structure the example better, make sure this is included in a
    # # TODO: unit-test
    # import matplotlib.pyplot as plt
    # import scipy.signal as sg
    # from PIL import Image

    # from dsgs.operators.linear_convolution import SerialConvolution
    # from dsgs.operators.padding import adjoint_padding, pad_array, pad_array_nd

    # # Generate 2D Gaussian convolution kernel
    # vr = 1
    # M = 3
    # if np.mod(M, 2) > 0:  # M odd
    #     n = np.arange(-(M - 1) // 2, (M - 1) // 2 + 1)
    # else:
    #     n = np.arange(-M // 2, M // 2)
    # h = np.exp(-(n**2 + n[:, np.newaxis] ** 2) / (2 * vr))

    # # plt.imshow(h, cmap=plt.cm.gray)
    # # plt.show()

    # # x = np.asarray(Image.open("img/cameraman.png", "r"), dtype="d")
    # rng = np.random.default_rng(1234)
    # x = rng.standard_normal((4, 4))
    # N = x.shape
    # M = h.shape

    # * version 1: circular convolution
    # K = N
    # hpad = pad_array(h, K, padmode="after")  # after, using fft convention for center
    # yc, H = fft2_conv(x, hpad, K)

    # circ_conv = SerialConvolution(np.array(K, dtype="i"), h, np.array(K, dtype="i"))
    # yc2 = circ_conv.forward(x)
    # print("yc2 == yc ? {0}".format(np.allclose(yc2, yc)))

    # # plt.imshow(yc, cmap=plt.cm.gray)
    # # plt.show()

    # # check adjoint operator (circular convolution)
    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # Hx_ = circ_conv.forward(x_)
    # y_ = rng.standard_normal(K)
    # Hadj_y_ = circ_conv.adjoint(y_)
    # hp1 = np.sum(Hx_ * y_)
    # hp2 = np.sum(x_ * Hadj_y_)

    # print(
    #     "Correct adjoint operator (circular convolution)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # # * version 1.2: circular convolution w/o Fourier
    # K = N
    # # with fft
    # hpad = pad_array(h, K, padmode="after")  # after, using fft convention for center
    # yc, H = fft2_conv(x, hpad, K)

    # # w/o fft (pad signal, linear convolution with x w/o additional border effect) -> ok
    # pad_width = [[M[n] - 1, 0] for n in range(len(M))]
    # xp = np.pad(x, pad_width, mode="wrap")
    # yc2 = sg.convolve2d(xp, h, boundary="fill", mode="valid")
    # print("yc2 == yc? {0}".format(np.allclose(yc2, yc)))
    # #
    # # yc3 = sg.convolve2d(xp, h, boundary="fill", mode="full")
    # # yc3 = yc3[M[0]-1:-(M[0]-1), M[1]-1:-(M[1]-1)]
    # # print("yc3 == yc2? {0}".format(np.allclose(yc2, yc3)))

    # # check adjoint operator (circular convolution, w/o Fourier)
    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # Hx_ = np.pad(x_, pad_width, mode="wrap")
    # Hx_ = sg.convolve2d(Hx_, h, boundary="fill", mode="valid")

    # y_ = rng.standard_normal(K)
    # Hadj_y_ = sg.convolve2d(y_, np.conj(np.flip(h)), boundary="fill", mode="full")
    # # ! manual adjoint padding (wrap condition)
    # # Hadj_y_[-(M[0]-1):, :] += Hadj_y_[:M[0]-1, :]
    # # Hadj_y_[:, -(M[1]-1):] += Hadj_y_[:, :M[1]-1]
    # # Hadj_y_ = Hadj_y_[M[0]-1:, M[1]-1:]
    # Hadj_y_ = adjoint_padding(
    #     Hadj_y_, [M[n] - 1 for n in range(len(M))], 2 * [0], mode="wrap"
    # )

    # hp1 = np.sum(Hx_ * y_)
    # hp2 = np.sum(x_ * Hadj_y_)
    # print(
    #     "Correct adjoint operator (circular convolution)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # # * version 1.3: circular convolution using a linear convolution,
    # # * pre-padding, using fft
    # # ! add adjoint operator here as well
    # K = N
    # yc, _ = fft2_conv(x, h, K)
    # # pad signal, then linear convolution w/o additional border effect
    # pad_width = [[M[n]-1, 0] for n in range(len(M))]
    # conv_shape = [N[n] + 2*(M[n] - 1) for n in range(len(M))]
    # conv_slice = np.s_[M[0]-1:-(M[0]-1), M[1]-1:-(M[1]-1)]
    # xp = np.pad(x, pad_width, mode="wrap")
    # yl, _ = fft2_conv(xp, h, conv_shape)
    # yc2 = yl[conv_slice]
    # print("yc2 == yc? {0}".format(np.allclose(yc2, yc)))

    # # check adjoint operator (circular convolution, w/o Fourier)
    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # Hx_ = np.pad(x_, pad_width, mode="wrap")
    # Hx_, H = fft2_conv(Hx_, h, shape=conv_shape)
    # Hx_ = Hx_[conv_slice]
    # y_ = rng.standard_normal(K)
    # Hadj_y_ = pad_array_nd(
    #     y_, [M[n] - 1 for n in range(len(M))], [M[n] - 1 for n in range(len(M))], mode="constant"
    # )
    # Hadj_y_ = fft_conv(Hadj_y_, np.conj(H), shape=conv_shape)
    # Hadj_y_ = Hadj_y_[:-(M[0]-1), :-(M[1]-1)]
    # Hadj_y_ = adjoint_padding(
    #     Hadj_y_, [M[n] - 1 for n in range(len(M))], 2 * [0], mode="wrap"
    # )
    # hp1 = np.sum(Hx_ * y_)
    # hp2 = np.sum(x_ * Hadj_y_)
    # print(
    #     "Correct adjoint operator (circular convolution)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # # * version 1.4: circular convolution using a linear convolution,
    # # * post-padding
    # # ! in this case, additional circular shift is required
    # K = N
    # # with fft
    # hpad = pad_array(h, K, padmode="after")  # after, using fft convention for center
    # yc, H = fft2_conv(x, hpad, K)
    # # pad signal, then linear convolution w/o additional border effect
    # shift = [[0, M[n]-1] for n in range(len(M))]
    # xp = np.pad(x, shift, mode="wrap")
    # shape = [N[n] + 2*(M[n] - 1) for n in range(len(M))]
    # hpad2 = pad_array(h, shape, padmode="after")
    # yl, _ = fft2_conv(xp, hpad2, shape)
    # # additional shift required for the borders to have elements in the same
    # # place as the reference Fourier-based definition
    # # ! crop first, then circular shift
    # yc2 = yl[M[0]-1:-(M[0]-1), M[1]-1:-(M[1]-1)]
    # yc2 = np.roll(yc2, [M[n]-1 for n in range(len(M))], axis=(0, 1))
    # print("yc2 == yc? {0}".format(np.allclose(yc2, yc)))

    # # * adjoint circular shift
    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # Ax_ = np.roll(x, [M[n]-1 for n in range(len(M))], axis=(0, 1))

    # y_ = rng.standard_normal(N)
    # Aadj_y_ = np.roll(y_, [-M[n]+1 for n in range(len(M))], axis=(0, 1))
    # hp1 = np.sum(Ax_ * y_)
    # hp2 = np.sum(x_ * Aadj_y_)
    # print(
    #     "Correct adjoint operator (circular convolution)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # ! not functional for now
    # * adjoint fft2_conv (for linear convolution)
    # K = [N[n] + M[n] - 1 for n in range(len(N))]

    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # hpad = pad_array(h, K, padmode="after")

    # Hx_, H = fft2_conv(x_, h, shape=K)

    # y_ = rng.standard_normal(K)
    # Hadj_y_, _ = fft2_conv(y_, np.conj(np.flip(hpad)), shape=K)  # ! this is wrong at the moment!!
    # # Hadj_y_ = sg.convolve2d(y_, np.conj(np.flip(h)), boundary="fill", mode="full")
    # Hadj_y_ = Hadj_y_[:-(M[0]-1), :-(M[1]-1)]
    # hp1 = np.sum(Hx_ * y_)
    # hp2 = np.sum(x_ * Hadj_y_)
    # print(
    #     "Correct adjoint operator (circular convolution)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # # * version 1.5: circular convolution using a linear convolution,
    # # * phase shift in Fourier to avoid circshift in signal space
    # # ! not functional in here
    # K = N
    # # with fft
    # hpad = pad_array(h, K, padmode="after")  # after, using fft convention for center
    # yc, H = fft2_conv(x, hpad, K)
    # # pad signal, then linear convolution w/o additional border effect
    # shift = [[0, M[n]-1] for n in range(len(M))]
    # xp = np.pad(x, shift, mode="wrap")

    # x_test = x[0, :]

    # # shape = [N[n] + 2*(M[n] - 1) for n in range(len(M))]
    # # fft_h = np.fft.fft2(h, shape)
    # # fft_h_shift = fft_h * np.exp(-2*1j*np.pi*(M[0]-1)*np.arange(fft_h.shape[0]) / shape[0])[:, None] * np.exp(-2*1j*np.pi*(M[1]-1)*np.arange(fft_h.shape[1]) / shape[1])[None, :]
    # # h_shift = np.fft.ifft2(fft_h_shift, shape)[M[0]:, M[1]:]

    # # ! applying phase shift in Fourier (not working for now)
    # # fft_x_shift = np.fft.rfft2(x, shape) * np.exp(-2*1j*np.pi*(M[0]-1)*np.arange(fft_h.shape[0]) / shape[0])[:, None] * np.exp(-2*1j*np.pi*(M[1]-1)*np.arange(fft_h.shape[1]) / shape[1])[None, :]

    # # x_shift = np.fft.irfft2(fft_x_shift, shape)[:N[0], :N[1]]
    # # xs = np.roll(x, [M[n]-1 for n in range(len(M))], axis=(0, 1))

    # # yl = np.fft.irfft2(fft_h * fft_x_shift, shape)
    # # yc2 = yl[M[0]-1:-(M[0]-1), M[1]-1:-(M[1]-1)]
    # # # yc2 = np.roll(yc2, [M[n]-1 for n in range(len(M))], axis=(0, 1))
    # # print("yc2 == yc? {0}".format(np.allclose(yc2, yc)))

    # # * version 2: linear convolution
    # # linear convolution
    # K = [N[n] + M[n] - 1 for n in range(len(N))]
    # H = np.fft.rfft2(h, K)
    # yl = np.fft.irfft2(H * np.fft.rfft2(x, K), K)  # zeros appear around

    # yl2 = fft_conv(x, H, K)
    # print("yl2 == yl ? {0}".format(np.allclose(yl2, yl)))

    # linear_conv = SerialConvolution(np.array(N, dtype="i"), h, np.array(K, dtype="i"))
    # yl3 = linear_conv.forward(x)
    # print("yl3 == yl ? {0}".format(np.allclose(yl3, yl)))

    # # plt.imshow(yl, cmap=plt.cm.gray)
    # # plt.show()

    # # check adjoint operator (linear convolution)
    # rng = np.random.default_rng(1234)
    # x_ = rng.standard_normal(N)
    # Hx_ = linear_conv.forward(x_)
    # y_ = rng.standard_normal(K)
    # Hadj_y_ = linear_conv.adjoint(y_)
    # hp1 = np.sum(Hx_ * y_)
    # hp2 = np.sum(x_ * Hadj_y_)

    # print(
    #     "Correct adjoint operator (linear convolution)? {}".format(np.isclose(hp1, hp2))
    # )

    # # adjoint operator (w/o fft)
    # Hadj_y = adjoint_conv(x_, h, N)
    # hp2 = np.sum(np.conj(x_) * Hadj_y_)

    # print(
    #     "Correct adjoint convolution operator (linear convolution, no fft)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    # # Notes:
    # # 5::4 -> slice(5, None, 4)
    # # 5:-1 -> slice(5, -1) -> np.s_[5:-1]
    # # s = alpha[5::4] == t = alpha[slice(5, None, 4)]

    # * convolution with symmetric boundary conditions
    # rng = np.random.default_rng(1234)
    # N_ = np.array(N, dtype="i")
    # M_ = np.array(M, dtype="i")
    # ext_size = M_ - 1
    # K_ = N_ + ext_size
    # Fh_d = np.fft.rfft2(h, s=N_ + 3 * M_ - 3)
    # Fh_a = np.fft.rfft2(h, s=K_ + M_ - 1)

    # x_ = rng.standard_normal(N)
    # y_ = rng.standard_normal(K_)

    # # direct operator (with sg.convolve2d)
    # Hx_ = sg.convolve2d(x_, h, boundary="symm", mode="full")

    # # direct operator (alternative using manual padding)
    # # xp = pad_array_nd(x_, ext_size, ext_size, mode="symmetric")
    # # Hx2 = sg.convolve(xp, h, mode="valid")

    # # direct operator (alternative based on fft)
    # xp = pad_array_nd(x_, ext_size, ext_size, mode="symmetric")
    # Hx2_ = fft_conv(xp, Fh_d, K_ + 2 * M_ - 2)
    # Hx2 = adjoint_padding(
    #     Hx2_, ext_size, ext_size, mode="constant"
    # )  # equivalent to valid
    # print(
    #     "Correct direct convolution (symmetric extension)? {}".format(
    #         np.allclose(Hx2, Hx_)
    #     )
    # )
    # # -> y = CHPx, C cropping (valid, remove M-1 on each side), P boundary extension, H convolution

    # # adjoint operator (using sg.convolve2d)
    # Hy_ = sg.convolve2d(y_, h, boundary="fill", mode="full")
    # Hadj_y_ = adjoint_padding(Hy_, ext_size, ext_size, mode="symmetric")

    # # adjoint operator (using fft)
    # # ! correct, but no conjugate of Fh_a here! why?
    # Hy0_ = fft_conv(y_, Fh_a, K_ + M_ - 1)
    # Hadj_y0 = adjoint_padding(Hy0_, ext_size, ext_size, mode="symmetric")

    # hp1 = np.sum(np.conj(Hx_) * y_)
    # hp2 = np.sum(np.conj(x_) * Hadj_y_)

    # print(
    #     "Correct adjoint convolution operator (symmetric extension)? {}".format(
    #         np.isclose(hp1, hp2)
    #     )
    # )

    pass

""" Set of helper functions to pad a numpy array to a predefined size under
several boundary conditions, with the implementation of the associated adjoint
operator.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np

from dsgs.operators.linear_operator import LinearOperator


def crop_array(y, output_size, padmode="after", center_convention=False):
    r"""Crop an array to a specified size.

    Crop the input array ``y`` to the desired size ``output_size``. Removes
    values on one end or on both sides of each dimension.

    Parameters
    ----------
    y : numpy.ndarray
        Input array.
    output_size : tuple[int]
        Size of the output array.
    padmode : str, optional
        Add zeros around or after the content of the array along each
        dimension. Defaults to "after". By default "after".
    center_convention : bool, optional
        Convention adopted for the center of the array (only for around). By
        default False (following the same convention as the `np.fft.fftshift`
        function).

    Returns
    -------
    numpy.ndarray
        Cropped array.

    Raises
    ------
    ValueError
        Ensures ``y.shape`` and ``output_size`` have the same length.
    ValueError
        Size after cropping should be smaller than the size of the input array.

    Example
    -------
    >>> array_shape = y.shape
    >>> output_size = [n//2 for n in array_shape]
    >>> x = crop_array(y, output_size)
    """
    array_shape = y.shape

    if not len(y.shape) == len(output_size):
        raise ValueError("`x.shape` and `array_shape` must have the same length")

    if any(np.array(array_shape) < np.array(output_size)):
        raise ValueError(
            "All the elements in `array_shape` should be greater \
            than `x.shape`."
        )

    crop_size = [array_shape[n] - output_size[n] for n in range(len(output_size))]
    if padmode == "after":  # add trailing zeros
        start_crop = [0 for n in range(len(output_size))]
        stop_crop = crop_size
    else:  # pad around
        if (
            center_convention
        ):  # center in index "np.floor((output_size+1)/2)-1" (imfilter convention)
            start_crop = [
                int(np.floor(crop_size[n] / 2)) for n in range(len(output_size))
            ]
            stop_crop = [
                int(np.ceil(crop_size[n] / 2)) for n in range(len(output_size))
            ]
        else:  # center in index "np.floor(output_size/2)" (np.fftshift convention)
            start_crop = [
                int(np.ceil(crop_size[n] / 2)) for n in range(len(output_size))
            ]
            stop_crop = [
                int(np.floor(crop_size[n] / 2)) for n in range(len(output_size))
            ]

    return y[
        tuple([np.s_[start_crop[n] : -stop_crop[n]] for n in range(len(output_size))])
    ]


def crop_array_nd(x, lsize, rsize):
    r"""Cropping an array.

    Crop the input array ``x`` using from the left (resp. right) using the
    entries in ``lsize`` (resp. ``rsize``).

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    lsize : tuple[int or None]
        Number of elements added to the left along each dimension. If no
        element added on a dimension, the corresponding value should be
        ``None``.
    rsize : tuple[int or None]
        Opposite of the number of elements added to the right along each
        dimension. If no element added on a dimension, the corresponding value
        should be ``None``.

    Returns
    -------
    y : numpy.ndarray
        Cropped array.

    Raises
    ------
    ValueError
        Ensures ``x.shape``, ``lsize`` and ``rsize`` contain the same length.

    Example
    -------
    >>> lsize = [n for n in len(x.shape)]
    >>> rsize = [n for n in len(x.shape)]
    >>> y = crop_array(x, lsize, rsize)
    """
    if not (len(lsize) == len(x.shape) and len(rsize) == len(x.shape)):
        raise ValueError("`x.shape`, `lsize` and `rsize` must have the same length.")
    return x[
        tuple([np.s_[lsize[n] : x.shape[n] - rsize[n]] for n in range(len(x.shape))])
    ]


def pad_array(
    x, output_size, padmode="after", center_convention=False, mode="constant"
):
    r"""Padding an array.

    Pad the input array ``x`` to the desired size ``output_size``. Inserts
    elements before or after the content of the array along each dimension.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    output_size : tuple[int]
        Size of array after padding
    padmode : str, optional
        Padding mode, around or after the content of the array along each
        dimension. By default "after".
    center_convention : bool, optional
        Convention adopted for the center of the array (only for the `around`
        padding mode). Defaults to False, following the same convention as the
        `np.fft.fftshift` function.
    mode : str, optional
        Type of padding ("constant", "reflect", "symmetric", ...). Same options
        as the `numpy.pad` function. By default "constant" (zero-padding).

    Returns
    -------
    y : numpy.ndarray
        Padded array.

    Raises
    ------
    ValueError
        Ensures ``x.shape`` and ``output_size`` have the same number of elements.
    ValueError
        Size after padding should be larger than the size of the input array.

    Example
    -------
    >>> N = x.shape
    >>> output_size = [2*n for n in N]
    >>> y = pad_array(x, output_size)

    Note
    ----
    See ``numpy.padding_func`` to pad a pre-allocated array in-place along a
    specified dimension.
    """
    array_shape = x.shape

    if not len(output_size) == len(x.shape):
        raise ValueError(
            "`x.shape` and `output_size` must have the same \
            length."
        )

    if any(np.array(output_size) < np.array(array_shape)):
        raise ValueError(
            "All the elements in `output_size` should be greater \
            than `x.shape`."
        )

    padding = [output_size[n] - array_shape[n] for n in range(len(array_shape))]

    if padmode == "after":  # add trailing zeros
        pad_width = [[0, padding[n]] for n in range(len(padding))]
    else:  # pad around
        if (
            center_convention
        ):  # center in index "np.floor((array_shape+1)/2)-1" (imfilter convention)
            pad_width = [
                [int(np.floor(padding[n] / 2)), int(np.ceil(padding[n] / 2))]
                for n in range(len(array_shape))
            ]
        else:  # center in index "np.floor(array_shape/2)" (fftshift convention)
            pad_width = [
                [int(np.ceil(padding[n] / 2)), int(np.floor(padding[n] / 2))]
                for n in range(len(array_shape))
            ]

    y = np.pad(x, pad_width, mode=mode)

    return y


def pad_array_nd(x, lsize, rsize, mode="constant"):
    r"""Padding an array using a specific boundary condition.

    Pad the input array ``x`` following the boundary condition ``mode``,
    pre-inserting a number of elements based on the entries of ``lsize``, and
    trailing elements based on the entries in ``rsize`` along each dimension.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    lsize : tuple[int] or numpy.ndarray of int
        Number of elements to be added to the left.
    rsize : tuple[int] or numpy.ndarray of int
        Number of elements to be added to the right.
    mode : str, optional
        Type of padding ("constant", "symmetric", ...). Same options as the
        `numpy.pad` function. By default "constant" (zero-padding).

    Returns
    -------
    y : numpy.ndarray
        Padded array.

    Raises
    ------
    ValueError
        Ensures ``x.shape``, ``lsize`` and ``rsize`` contain the same length.

    Example
    -------
    >>> array_shape = x.shape
    >>> lsize = [n for n in array_shape]
    >>> rsize = [n for n in array_shape]
    >>> y = pad_array(x, lsize, rsize)
    """
    array_shape = x.shape

    if not (len(lsize) == len(array_shape) and len(rsize) == len(array_shape)):
        raise ValueError("`x.shape`, `lsize` and `rsize` must have the same length.")

    pad_width = [[int(lsize[n]), int(rsize[n])] for n in range(len(array_shape))]
    y = np.pad(x, pad_width, mode=mode)

    return y


def adjoint_padding(y, lsize, rsize, mode="constant"):
    """Adjoint of the padding operator corresponding to a given extension mode.

    Adjoint of a boundary extension operator, following the boundary condition
    ``mode`` according to which ``lsize`` elements have been pre-inserted, and
    ``rsize``appended along each dimension of the array.

    Parameters
    ----------
    y : numpy.ndarray
        Input array.
    lsize : numpy.ndarray of int
        Number of elements added to the left along each dimension. If no
        element added on a dimension, the corresponding value should be
        ``0``.
    rsize : numpy.ndarray of int
        Opposite of the number of elements added to the right along each
        dimension. If no element added on a dimension, the corresponding value
        should be ``0``.
    mode : str, optional
        Type of padding ("constant", "symmetric", ...). Limited to the
        "constant" (zero-padding), "symmetric" (hal-point symmetric) and "wrap"
        (circular) extension modes of the `numpy.pad` function. By default
        "constant".

    Returns
    -------
    x : numpy.ndarray
        Adjoint of the padding operator evaluated in ``y``.

    Raises
    ------
    ValueError
        Ensures ``x.shape``, ``lsize`` and ``rsize`` contain the same length.
    ValueError
        Unknown extension ``mode``.
    """
    array_shape = y.shape
    ndims = len(array_shape)

    if not (len(lsize) == ndims and len(rsize) == ndims):
        raise ValueError("`x.shape`, `lsize` and `rsize` must have the same length.")

    if mode == "constant":
        x = crop_array_nd(y, lsize, rsize)

    elif mode == "symmetric":
        # ! requires lsize[d] <= y.shape[d]
        if np.any(
            np.maximum(lsize, rsize)
            > np.array(array_shape, dtype="i") - (lsize + rsize)
        ):
            raise ValueError(
                r"Extension mode {} requires: `np.maximum(lsize[d], rsize[d]) > input.shape[d]` for each axis `d`".format(
                    mode
                )
            )

        # sel_core = tuple([np.s_[lsize[d] : rsize[d]] for d in range(ndims)])
        sel_core = []
        x_ = np.copy(y)

        # fold extension successively along each dimension (flip + sum)
        for d in range(ndims):
            sel_core.append(np.s_[lsize[d] : y.shape[d] - rsize[d]])

            lselx = tuple(
                d * [np.s_[:]]
                + [np.s_[lsize[d] : 2 * lsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rselx = tuple(
                d * [np.s_[:]]
                + [np.s_[-2 * rsize[d] : -rsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            lsely = tuple(
                d * [np.s_[:]]
                + [np.s_[: lsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rsely = tuple(
                d * [np.s_[:]]
                + [np.s_[y.shape[d] - rsize[d] :]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            # left-hand side extension folded on first entries
            x_[lselx] += np.flip(x_[lsely], axis=d)

            # right-hand side extension aliased on first entries
            x_[rselx] += np.flip(x_[rsely], axis=d)
        x = x_[tuple(sel_core)]

    elif mode == "reflect":
        # ! requires lsize[d] + 1 <= y.shape[d], cannot be implemented otherwise
        if np.any(
            np.maximum(lsize, rsize) + 1
            > np.array(array_shape, dtype="i") - (lsize + rsize)
        ):
            raise ValueError(
                r"Extension mode {} requires: `np.maximum(lsize[d], rsize[d]) + 1 > output.shape[d]` for each axis `d`".format(
                    mode
                )
            )

        # sel_core = tuple([np.s_[lsize[d] : rsize[d]] for d in range(ndims)])
        sel_core = []
        x_ = np.copy(y)

        # fold extension successively along each dimension (flip + sum)
        for d in range(ndims):
            sel_core.append(np.s_[lsize[d] : y.shape[d] - rsize[d]])

            lselx = tuple(
                d * [np.s_[:]]
                + [np.s_[lsize[d] + 1 : 2 * lsize[d] + 1]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rselx = tuple(
                d * [np.s_[:]]
                + [np.s_[-2 * rsize[d] - 1 : -rsize[d] - 1]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            lsely = tuple(
                d * [np.s_[:]]
                + [np.s_[: lsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rsely = tuple(
                d * [np.s_[:]]
                + [np.s_[y.shape[d] - rsize[d] :]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            # left-hand side extension folded on first entries
            x_[lselx] += np.flip(x_[lsely], axis=d)

            # right-hand side extension aliased on first entries
            x_[rselx] += np.flip(x_[rsely], axis=d)
        x = x_[tuple(sel_core)]

    elif mode == "wrap":
        # sel_core = tuple([np.s_[lsize[d] : rsize[d]] for d in range(ndims)])
        sel_core = []
        x_ = np.copy(y)

        # fold extension successively along each dimension (sum first elements
        # with last ones)
        for d in range(ndims):
            sel_core.append(np.s_[lsize[d] : y.shape[d] - rsize[d]])

            lselx = tuple(
                d * [np.s_[:]]
                + [np.s_[lsize[d] : lsize[d] + rsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rselx = tuple(
                d * [np.s_[:]]
                + [np.s_[y.shape[d] - (lsize[d] + rsize[d]) : y.shape[d] - rsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            lsely = tuple(
                d * [np.s_[:]]
                + [np.s_[: lsize[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            rsely = tuple(
                d * [np.s_[:]]
                + [np.s_[y.shape[d] - rsize[d] : y.shape[d]]]
                + np.max((ndims - d - 1), 0) * [np.s_[:]]
            )

            # right-hand side extension aliased on first entries
            # ! only performed if extension from the left (before)
            x_[lselx] += x_[rsely]

            # left-hand side extension aliased on first entries
            # ! only performed if extension from the right (after)
            x_[rselx] += x_[lsely]

        x = x_[tuple(sel_core)]
    else:
        raise ValueError("Unknown extension `mode`: {}".format(mode))

    return x


class Padding(LinearOperator):
    """Padding operator.

    Attributes
    ----------
    lsize : numpy.ndarray of int
            Number of elements added to the left along each dimension. If no
            element added on a dimension, the corresponding value should be
            ``0``.
    rsize : numpy.ndarray of int
        Opposite of the number of elements added to the right along each
        dimension. If no element added on a dimension, the corresponding
        value should be ``0``.
    mode : str, optional
        Type of padding ("constant", "symmetric", ...). Limited to the
        "constant" (zero-padding), "symmetric" (hal-point symmetric) and "wrap"
        (circular) extension modes of the `numpy.pad` function. By default
        "constant".
    """

    # TODO: include tests at the level of the object (not the application of
    # TODO: - the functions / method)
    def __init__(self, lsize, rsize, mode="constant"):
        """Implementation of padding as a linear operator.

        Parameters
        ----------
        lsize : numpy.ndarray of int
            Number of elements added to the left along each dimension. If no
            element added on a dimension, the corresponding value should be
            ``0``.
        rsize : numpy.ndarray of int
            Opposite of the number of elements added to the right along each
            dimension. If no element added on a dimension, the corresponding
            value should be ``0``.
        mode : str, optional
            Type of padding ("constant", "symmetric", ...). Limited to the
            "constant" (zero-padding), "symmetric" (hal-point symmetric) and "wrap"
            (circular) extension modes of the `numpy.pad` function. By default
            "constant".

        Raises
        ------
        ValueError
            Ensures ``lsize`` and ``rsize`` contain the same
            length.
        """

        if not (len(lsize) == len(rsize)):
            raise ValueError("`lsize` and `rsize` must have the same length.")

        self.lsize = lsize
        self.rsize = rsize
        self.mode = mode

    def set_padding_size(self, lsize, rsize):
        """Reset the padding size.

        Parameters
        ----------
        lsize : numpy.ndarray of int
            Number of elements added to the left along each dimension. If no
            element added on a dimension, the corresponding value should be
            ``0``.
        rsize : numpy.ndarray of int
            Opposite of the number of elements added to the right along each
            dimension. If no element added on a dimension, the corresponding
            value should be ``0``.
        """
        self.lsize = lsize
        self.rsize = rsize

    def set_padding_mode(self, mode):
        """Rese tthe padding mode.

        Parameters
        ----------
        mode : str, optional
            Type of padding ("constant", "symmetric", ...). Limited to the
            "constant" (zero-padding), "symmetric" (hal-point symmetric) and
            "wrap" (circular) extension modes of the `numpy.pad` function. By
            default "constant".
        """
        self.mode = mode

    def forward(self, x):
        """Pad an array to the desired size.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Padded array.
        """
        return pad_array_nd(x, self.lsize, self.rsize, self.mode)

    def adjoint(self, y):
        """Apply ajoint padding operator.

        Parameters
        ----------
        y : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output of the adjoint padding operator.
        """
        return adjoint_padding(y, self.lsize, self.rsize, self.mode)


# if __name__ == "__main__":
#     # import matplotlib.pyplot as plt
#     # from imageio import imread

#     # Generate 2D Gaussian convolution kernel
#     vr = 1
#     M = 7
#     # if np.mod(M, 2) > 0:  # M odd
#     #     n = np.arange(-(M - 1) // 2, (M - 1) // 2 + 1)
#     # else:
#     #     n = np.arange(-M // 2, M // 2)
#     # h = np.exp(-(n ** 2 + n[:, np.newaxis] ** 2) / (2 * vr))

#     # plt.imshow(h, cmap=plt.cm.gray)
#     # plt.show()

#     # x = imread("img/cameraman.png")
#     # N = x.shape
#     # M = h.shape

#     # # version 1: circular convolution
#     # # circular convolution: pad around and fftshift kernel for nicer results
#     # input_size = N
#     # hpad = pad_array(h, input_size, padmode="after")  # around

#     # plt.imshow(hpad, cmap=plt.cm.gray)
#     # plt.show()

#     # testing symmetric padding (with adjoint) (constant is fine)
#     rng = np.random.default_rng(1234)
#     x = rng.standard_normal(size=(M,))
#     y = rng.standard_normal(size=(M + 4,))

#     Hx = pad_array(
#         x, [M + 4], padmode="around", mode="symmetric"
#     )  # around, 2 on both sides
#     Hadj_y = adjoint_padding((y, np.array([2], dtype="i"), np.array([2], dtype="i"), mode="symmetric")

#     hp1 = np.sum(Hx * y)
#     hp2 = np.sum(x * Hadj_y)

#     print("Correct adjoint operator? {}".format(np.isclose(hp1, hp2)))

#     # 2D: ok
#     # hpad = pad_array(
#     #     h, [M[0] + 3, M[1] + 1], padmode="around", mode="symmetric"
#     # )  # around
#     # plt.imshow(hpad, cmap=plt.cm.gray)
#     # plt.show()

#     pass

"""Implementation of the discrete istropic total variation, with a smoothed
variant. Functions benefit from numba jit compilation.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import numpy as np
from numba import jit

# ? note: apparently, @njit not compatible with np.r_ or np.c_
# ? np.diff not compatible with @jit (at least the axis keyword...)
# ! need to remove keywords for proper jitting
# ! does not support concatenation over a new dimension (cannot use np.newaxis)
# ! does not support type elision
# ! only jit costly parts (by decomposing function), keep flexibility of Python
# ! as much as possible
# import importlib
# importlib.reload(...)

# TODO: investigate jitted nD version for the TV (not only 2D)
# TODO: try to simplify the 2d implementation of the chunked version of the TV
# TODO (many conditions to be checked at the moment)

# * Useful numba links
# https://stackoverflow.com/questions/57662631/vectorizing-a-function-returning-tuple-using-numba-guvectorize
# https://stackoverflow.com/questions/30363253/multiple-output-and-numba-signatures
# https://numba.pydata.org/numba-doc/0.17.0/reference/types.html


@jit(nopython=True, cache=True)
def gradient_2d(x):
    r"""Compute 2d discrete gradient (with jit support).

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    *i.e.*, by computing horizontal and vertical differences (using jit compilation):

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : numpy.ndarray, 2d
        Input 2d array :math:`\mathbf{x}`.

    Returns
    -------
    uh : numpy.ndarray
        Horizontal differences.
    uv : numpy.ndarray
        Vertical differences.
    """
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected len(x.shape)==2"
    uh = np.zeros_like(x)
    uh[:, :-1] = x[:, 1:] - x[:, :-1]  # np.diff(x,1,1) horizontal differences
    uv = np.zeros_like(x)
    uv[:-1, :] = x[1:, :] - x[:-1, :]  # np.diff(x,1,0) vertical differences
    return uh, uv


@jit(nopython=True, cache=True)
def gradient_2d_adjoint(uh, uv):
    r"""Adjoint of the 2d discrete gradient operator (with jit support).

    Compute the adjoint of the 2d discrete gradient of a 2d input array
    :math:`\mathbf{x}` (using jit compilation),

    .. math::
       \nabla^*(\mathbf{y}) = - \text{div} (\mathbf{y})
       = \nabla_v^*\mathbf{y}_v + \mathbf{y}_h\nabla_h^*.

    Parameters
    ----------
    uh : numpy.ndarray, 2d
        Horizontal differences.
    uv : numpy.ndarray, 2d
        Vertical differences.

    Returns
    -------
    v : numpy.ndarray
        Adjoint of the 2d gradient operator, evaluated in
        :math:`(\mathbf{u}_h, \mathbf{u}_v)`.
    """
    assert (len(uh.shape) == 2) and (
        len(uv.shape) == 2
    ), "gradient_2d_adjoint: Invalid input, expected len(uh.shape)==len(uv.shape)==2"
    # horizontal, vertical
    v = np.zeros_like(uh)
    v[0, :] = -uv[0, :]
    v[1:-1, :] = uv[:-2, :] - uv[1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = uv[-2, :]
    v[:, 0] -= uh[:, 0]
    v[:, 1:-1] += uh[:, :-2] - uh[:, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += uh[:, -2]
    return v


# ! try to simplify the structure of this function make sure uh and uv are
# ! directly concatenated
# @jit(nopython=True, cache=True)
@jit(
    [
        "UniTuple(float64[:,:], 2)(float64[:,:], b1[:])",
        "UniTuple(complex128[:,:], 2)(complex128[:,:], b1[:])",
    ],
    nopython=True,
    cache=True,
)
def chunk_gradient_2d(x, islast):
    r"""Chunk of the 2d discrete gradient (with jit support).

    Compute a chunk of the 2d discrete gradient operator (using jit
    compilation).

    Parameters
    ----------
    x : numpy.ndarray[float64 or complex128], 2d
        Input array.
    islast : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the last one along each
        dimension of the Cartesian process grid.

    Returns
    -------
    uh : numpy.ndarray[float64 or complex128], 2d
        Horizontal differences.
    uv : numpy.ndarray[float64 or complex128], 2d
        Vertical differences.
    """
    assert (
        len(x.shape) == 2 and islast.size == 2
    ), "gradient_2d: Invalid input, expected len(x.shape)==len(offset.shape)==2"
    # horizontal differences
    if islast[1]:  # true if facet is the last along dimension 1
        if islast[0]:
            uh = np.zeros(x.shape, dtype=x.dtype)
            uh[:, :-1] = x[:, 1:] - x[:, :-1]
        else:
            uh = np.zeros((x.shape[0] - 1, x.shape[1]), dtype=x.dtype)
            uh[:, :-1] = x[:-1, 1:] - x[:-1, :-1]
    else:
        if islast[0]:
            uh = x[:, 1:] - x[:, :-1]
        else:
            uh = x[:-1, 1:] - x[:-1, :-1]

    # vertical differences
    if islast[0]:  # true if facet is the last along dimension 0
        if islast[1]:
            uv = np.zeros(x.shape, dtype=x.dtype)
            uv[:-1, :] = x[1:, :] - x[:-1, :]
        else:
            uv = np.zeros((x.shape[0], x.shape[1] - 1), dtype=x.dtype)
            uv[:-1, :] = x[1:, :-1] - x[:-1, :-1]
    else:
        if islast[1]:
            uv = x[1:, :] - x[:-1, :]
        else:
            uv = x[1:, :-1] - x[:-1, :-1]

    return uh, uv


# ! try to simplify the structure of this function make sure uh and uv are
# ! directly concatenated
@jit(
    [
        "void(f8[:,:], f8[:,:], f8[:,:], b1[:], b1[:])",
        "void(c16[:,:], c16[:,:], c16[:,:], b1[:], b1[:])",
    ],
    nopython=True,
    cache=True,
)
def chunk_gradient_2d_adjoint(uh, uv, v, isfirst, islast):
    r"""Chunk of the adjoint 2d discrete gradient (with jit support).

    Compute a chunk of the adjoint 2d discrete gradient.

    Parameters
    ----------
    uh : numpy.ndarray[float64 or complex128], 2d
        Horizontal differences.
    uv : numpy.ndarray[float64 or complex128], 2d
        Vertical differences.
    v : numpy.ndarray[float64 or complex128], 2d
        Output array (updated in-place).
    isfirst : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the first one along each
        dimension of the Cartesian process grid.
    islast : numpy.ndarray, bool, 1d
        Vector indicating whether the chunk is the last one along each
        dimension of the Cartesian process grid.

    ..note::
        The array ``v`` is updated in-place.
    """
    # TODO: implicit conditions on the size of the facet: needs to be chekced
    # TODO  before-hand
    # TODO: find a simpler way to encode this

    assert (len(uh.shape) == 2) and (
        len(uv.shape) == 2
    ), "gradient_2d_adjoint: Invalid input, expected \
        len(uh.shape)==len(uv.shape)==len(offset.shape)==2"
    # v = np.zeros_like(uh, shape=Nk)

    # vertical
    # overlap from the left
    if isfirst[0]:
        if isfirst[1]:
            v[0, :] -= uv[0, :]
            if islast[0]:
                v[1:-1, :] += uv[:-2, :] - uv[1:-1, :]
                v[-1, :] += uv[-2, :]
            else:
                v[1:, :] += uv[:-1, :] - uv[1:, :]
        else:
            v[0, :] -= uv[0, 1:]
            if islast[0]:
                v[1:-1, :] += uv[:-2, 1:] - uv[1:-1, 1:]
                v[-1, :] += uv[-2, 1:]
            else:
                v[1:, :] += uv[:-1, 1:] - uv[1:, 1:]
    else:
        if isfirst[1]:
            if islast[0]:
                v[:-1, :] += uv[:-2, :] - uv[1:-1, :]
                v[-1, :] += uv[-2, :]
            else:
                v += uv[:-1, :] - uv[1:, :]
        else:
            if islast[0]:
                v[:-1, :] += uv[:-2, 1:] - uv[1:-1, 1:]
                v[-1, :] += uv[-2, 1:]
            else:
                v += uv[:-1, 1:] - uv[1:, 1:]

    # horizontal
    if isfirst[1]:
        if isfirst[0]:
            v[:, 0] -= uh[:, 0]
            if islast[1]:
                v[:, 1:-1] += uh[:, :-2] - uh[:, 1:-1]
                v[:, -1] += uh[:, -2]
            else:
                v[:, 1:] += uh[:, :-1] - uh[:, 1:]
        else:
            v[:, 0] -= uh[1:, 0]
            if islast[1]:
                v[:, 1:-1] += uh[1:, :-2] - uh[1:, 1:-1]
                v[:, -1] += uh[1:, -2]
            else:
                v[:, 1:] += uh[1:, :-1] - uh[1:, 1:]
    else:
        if isfirst[0]:
            if islast[1]:
                v[:, :-1] += uh[:, :-2] - uh[:, 1:-1]
                v[:, -1] += uh[:, -2]
            else:
                v += uh[:, :-1] - uh[:, 1:]
        else:
            if islast[1]:
                v[:, :-1] += uh[1:, :-2] - uh[1:, 1:-1]
                v[:, -1] += uh[1:, -2]
            else:
                v += uh[1:, :-1] - uh[1:, 1:]
    return


@jit(nopython=True, cache=True)
def tv(x):
    r"""Discrete anisotropic total variation (TV) (with jit support).

    Compute the discrete anisotropic total variation of a 2d array (using
    jit compilation):

    .. math::
       \text{TV}(\mathbf{x}) = \Vert \nabla (\mathbf{x}) \Vert_{2, 1},

    where :math:`\nabla` is the 2d discrete gradient operator.

    Parameters
    ----------
    x : numpy.ndarray, 2d
        Input array.

    Returns
    -------
    float
        Total variation evaluated in ``x``.
    """
    u = gradient_2d(x)
    return np.sum(np.sqrt(np.abs(u[0]) ** 2 + np.abs(u[1]) ** 2))


@jit(nopython=True, cache=True)
def smooth_tv(x, eps):
    r"""Smooth approximation to the 2d discrete total variation (TV)
    (with jit support).

    Compute a smooth approximation to the discrete anisotropic total variation
    of a 2d array (with jit support):

    .. math::
        \text{TV}_{\varepsilon}(\mathbf{x}) = \sum_{n=1}^N \sum_{m=1}^M \sqrt{
        [\nabla(\mathbf{x})]_{1, m, n}^2 + [\nabla(\mathbf{x})]_{2, m, n}^2
        + \varepsilon}, \; \varepsilon > 0,

    where :math:`\nabla` is the 2d discrete gradient operator.

    Parameters
    ----------
    x : numpy.ndarray, 2d
        Input array.
    eps : float, > 0
        Smoothing parameter.

    Returns
    -------
    float
        Smooth TV evaluated in ``x``, :math:`\text{TV}_{\varepsilon}(\mathbf{x})`.
    """
    u = gradient_2d(x)
    return np.sum(np.sqrt(np.abs(u[0]) ** 2 + np.abs(u[1]) ** 2 + eps))


@jit(nopython=True, cache=True)
def gradient_smooth_tv(x, eps):
    r"""Jitted gradient of a smoothed 2d anisotropic total variation (with jit
    support).

    Compute the gradient of a smooth approximation to the 2d discrete
    anisotropic total variation, evaluated in the input array `x` (with jit
    support).

    Parameters
    ----------
    x : numpy.ndarray, 2d
        Input array.
    eps : float, > 0
        Smoothing parameter.

    Returns
    -------
    numpy.ndarray, 2d
        Gradient of :math:`\text{TV}_\varepsilon`, evaluated in ``x``.
    """
    u = gradient_2d(x)
    w = np.sqrt(np.abs(u[0]) ** 2 + np.abs(u[1]) ** 2 + eps)
    return gradient_2d_adjoint(u[0] / w, u[1] / w)


# if __name__ == "__main__":
#     import timeit
#     from inspect import cleandoc  # handle identation in multi-line strings

#     rng = np.random.default_rng(1234)
#     x = rng.standard_normal((10, 5))
#     eps = np.finfo(float).eps

#     stmt_s = "tv.gradient_smooth_tv(x, eps)"
#     setup_s = cleandoc(
#         """
#     import tv
#     import from __main__ import x, eps
#     """
#     )
#     t_tv_np = timeit.timeit(
#         stmt_s, number=100, globals=globals()
#     )  # save in pandas (prettier display)
#     print("TV (numpy version): ", t_tv_np)

#     _ = gradient_smooth_tv(x, eps)  # trigger numba compilation
#     stmt_s = "gradient_smooth_tv(x, eps)"
#     setup_s = "from __main__ import gradient_smooth_tv, x, eps"
#     t_tv_numba = timeit.timeit(stmt_s, setup=setup_s, number=100)
#     print("TV (numba version): ", t_tv_numba)

#     # ! multiple facets (serial test)
#     # uh0, uv0 = gradient_2d(x)

#     # yh0 = (1 + 1j) * rng.standard_normal(uh0.shape)
#     # yv0 = (1 + 1j) * rng.standard_normal(uv0.shape)
#     # z0 = gradient_2d_adjoint(yh0, yv0)

#     # ndims = 2
#     # grid_size = np.array([3, 2], dtype="i")
#     # nchunks = np.prod(grid_size)
#     # N = np.array(x.shape, dtype="i")

#     # overlap = (grid_size > 1).astype(int)
#     # isfirst = np.empty((nchunks, ndims), dtype=bool)
#     # islast = np.empty((nchunks, ndims), dtype=bool)

#     # range0 = []
#     # range_direct = []
#     # range_adjoint = []
#     # uh = np.zeros(x.shape)
#     # uv = np.zeros(x.shape)
#     # z = np.zeros(x.shape, dtype=complex)

#     # for k in range(nchunks):

#     #     ranknd = np.array(np.unravel_index(k, grid_size), dtype="i")
#     #     islast[k] = ranknd == grid_size - 1
#     #     isfirst[k] = ranknd == 0
#     #     range0.append(ucomm.local_split_range_nd(grid_size, N, ranknd))
#     #     Nk = range0[k][:,1] - range0[k][:,0] + 1

#     #     # * direct operator
#     #     # version with backward overlap
#     #     # range_direct.append(ucomm.local_split_range_nd(grid_size, N, ranknd, overlap=overlap, backward=True))
#     #     # facet = x[range_direct[k][0,0]:range_direct[k][0,1]+1, \
#     #     # range_direct[k][1,0]:range_direct[k][1,1]+1]
#     #     # uh_k, uv_k = chunk_gradient_2d(facet, islast[k])

#     #     # start = range_direct[k][:,0]
#     #     # stop = start + np.array(uv_k.shape, dtype="i")
#     #     # uv[start[0]:stop[0], start[1]:stop[1]] = uv_k
#     #     # stop = start + np.array(uh_k.shape, dtype="i")
#     #     # uh[start[0]:stop[0], start[1]:stop[1]] = uh_k

#     #     # version with forward overlap
#     #     range_direct.append(ucomm.local_split_range_nd(grid_size, N, ranknd, overlap=overlap, backward=False))
#     #     facet = x[range_direct[k][0,0]:range_direct[k][0,1]+1, \
#     #     range_direct[k][1,0]:range_direct[k][1,1]+1]
#     #     uh_k, uv_k = chunk_gradient_2d(facet, islast[k])

#     #     start = range0[k][:,0]
#     #     stop = start + np.array(uv_k.shape, dtype="i")
#     #     uv[start[0]:stop[0], start[1]:stop[1]] = uv_k
#     #     stop = start + np.array(uh_k.shape, dtype="i")
#     #     uh[start[0]:stop[0], start[1]:stop[1]] = uh_k

#     #     # * adjoint (backward overlap only, forward more difficult to encode)
#     #     range_adjoint.append(ucomm.local_split_range_nd(grid_size, N, ranknd, overlap=overlap, backward=True))
#     #     facet_h = yh0[range_adjoint[k][0,0]:range_adjoint[k][0,1]+1, \
#     #     range_adjoint[k][1,0]:range_adjoint[k][1,1]+1]
#     #     facet_v = yv0[range_adjoint[k][0,0]:range_adjoint[k][0,1]+1, \
#     #     range_adjoint[k][1,0]:range_adjoint[k][1,1]+1]
#     #     x_k = np.zeros(Nk, dtype=complex)
#     #     chunk_gradient_2d_adjoint(facet_h, facet_v, x_k, isfirst[k], islast[k])

#     #     start = range0[k][:,0]
#     #     stop = start + Nk
#     #     z[start[0]:stop[0], start[1]:stop[1]] = x_k

#     # print(np.allclose(uh, uh0))
#     # print(np.allclose(uv, uv0))
#     # print(np.allclose(z, z0))

#     pass

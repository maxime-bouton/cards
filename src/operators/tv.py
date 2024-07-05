""" Implementation of the discrete istropic total variation, with a smoothed
variant. Functions do not benefit from numba jit compilation. Includes a
version supporting input tensors.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

from collections import deque

import numpy as np


def gradient_2d(x):
    r"""Compute 2d discrete gradient.

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    **i.e.**, by computing horizontal and vertical differences:

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : numpy.ndarray
        Input 2d array :math:`\mathbf{x}`.

    Returns
    -------
    numpy.ndarray, of shape ``(2, *x.shape)``
        Vertical and horizontal differences, concatenated along the axis 0.
    """
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected a 2d numpy array"
    # horizontal differences
    uh = np.c_[np.diff(x, axis=1), np.zeros([x.shape[0], 1])]
    # vertical differences
    uv = np.r_[np.diff(x, axis=0), np.zeros([1, x.shape[1]])]
    # ! concatenate along the 1st dimension (slowest access)
    return np.r_["0,2", uv[np.newaxis, ...], uh[np.newaxis, ...]]


def gradient_2d_adjoint(y):
    r"""Adjoint of the 2d discrete gradient operator.

    Compute the adjoint of the 2d discrete gradient of a 2d input array
    :math:`\mathbf{x}`,

    .. math::
       \nabla^*(\mathbf{y}) = - \text{div} (\mathbf{y})
       = \nabla_v^*\mathbf{y}_v + \mathbf{y}_h\nabla_h^*.

    Parameters
    ----------
    y : numpy.ndarray, 3d
        Input array.

    Returns
    -------
    numpy.ndarray, of shape ``(y.shape[1], y.shape[2])``
        Adjoint of the 2d gradient operator, evaluated in :math:`\mathbf{y}`.
    """
    # v: horizontal, vertical
    # np.r_['0,2',-y[0,:,0],-np.diff(y[:-1,:,0],1,0),y[-2,:,0]] + \
    # np.c_['1,2',-y[:,0,1],-np.diff(y[:,:-1,1],1,1),y[:,-2,1]]
    return (
        np.r_["0,2", -y[0, 0, :], -np.diff(y[0, :-1, :], 1, 0), y[0, -2, :]]
        + np.c_["1,2", -y[1, :, 0], -np.diff(y[1, :, :-1], 1, 1), y[1, :, -2]]
    )


def tv(x):
    r"""Discrete anisotropic total variation (TV).

    Compute the discrete anisotropic total variation of a 2d array

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
        total variation evaluated in ``x``.
    """
    u = gradient_2d(x)
    return np.sum(np.sqrt(np.sum(np.abs(u) ** 2, axis=0)))


# TODO: see of the n-D version of the TV can be possibly simplified (i.e., to
# TODO: enable jit support)
def gradient_nd(x):
    r"""Nd discrete gradient operator.

    Compute the discrete gradient of an input tensor :math:`\mathbf{x}`,
    **i.e.**, by computing differences along each dimension of the tensor.

    Parameters
    ----------
    x : numpy.ndarray
        Input tensor.

    Returns
    -------
    tuple[numpy.ndarray]
        Nd discrete gradient :math:`\nabla \mathbf{x}`.

    Note
    ----
    This function is likely to be slow.
    """
    s = x.shape
    sz = np.array(s)  # number of zeros to be added
    u = np.zeros((len(s), *s), dtype=x.dtype)
    for k in range(len(s)):
        sz[k] = 1
        u[k] = np.concatenate((np.diff(x, axis=k), np.zeros(sz)), axis=k)
        sz[k] = s[k]
    return u


def gradient_nd_adjoint(u):
    r"""Adjoint of the nd discrete gradient operator.

    Compute the adjoint of the nd discrete gradient of an input tensor.

    Parameters
    ----------
    u : tuple[numpy.ndarray]
        Input elements.

    Returns
    -------
    numpy.ndarray
        adjoint of the nd gradient operator, evaluated in ``u``.

    Note
    ----
    This function is likely to be slow.
    """
    ndims = int(len(u.shape) - 1)
    # auxiliary indices to handle slices and np.newaxis needed for the concatenation
    # ! use deque to allow circshift of the content of the indexing tuples
    id1 = deque((0,) + (ndims - 1) * (slice(None),))  # [0, :, :, ..., :]
    id2 = deque((slice(0, -1),) + (ndims - 1) * (slice(None),))  # [:-1, :, ...]
    id3 = deque((-2,) + (ndims - 1) * (slice(None),))  # [-2, :, :, ..., :]
    idna = deque((np.newaxis,) + (ndims - 1) * (slice(None),))  # [np.newaxis, :, ...]
    # evaluate adjoint operator
    x = np.concatenate(
        (
            -u[0, 0, ...][np.newaxis, ...],
            -np.diff(u[0, :-1, ...], 1, axis=0),
            u[0, -2, ...][np.newaxis, ...],
        ),
        axis=0,
    )
    for k in range(1, ndims):
        # shift indexing deque towards the right
        id1.rotate(1)
        id2.rotate(1)
        id3.rotate(1)
        idna.rotate(1)
        # add new contribution
        x += np.concatenate(
            (
                -u[(k,) + tuple(id1)][tuple(idna)],
                -np.diff(u[(k,) + tuple(id2)], 1, axis=k),
                u[(k,) + tuple(id3)][tuple(idna)],
            ),
            axis=k,
        )
    return x


def tv_nd(x):
    r"""Generalisation of the discrete total variation (TV) to tensors.

    Compute the discrete anisotropic TV for any input tensor.

    Parameters
    ----------
    x : numpy.ndarray
        Input tensor.

    Returns
    -------
    float
        Total variation evaluated in ``x``.

    Note
    ----
    This function is likely to be slow.
    """
    u = gradient_nd(x)
    return np.sum(np.sqrt(np.sum(np.abs(u) ** 2, axis=0)))


def smooth_tv(x, eps):
    r"""Smooth approximation to the 2d discrete total variation (TV).

    Compute a smooth approximation to the discrete anisotropic total variation
    of a 2d array:

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
        smooth TV evaluated in ``x``, :math:`\text{TV}_{\varepsilon}(\mathbf{x})`.
    """
    u = gradient_2d(x)
    return np.sum(np.sqrt(np.sum(np.abs(u) ** 2, axis=0) + eps))


def gradient_smooth_tv(x, eps):
    r"""Gradient of a smoothed 2d anisotropic total variation.

    Compute the gradient of a smooth approximation to the 2d discrete
    anisotropic total variation, evaluated in the input array ``x``.

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
    v = gradient_2d_adjoint(u / np.sqrt(np.sum(np.abs(u) ** 2, axis=0) + eps))
    return v


# if __name__ == "__main__":
#     rng = np.random.default_rng(1234)
#     x = rng.standard_normal((5, 5))
#     eps = np.finfo(float).eps

#     u2 = gradient_2d(x)
#     y2 = gradient_2d_adjoint(u2)

#     u = gradient_nd(x)
#     y = gradient_nd_adjoint(u)

#     err_ = np.linalg.norm(y - y2)
#     print("Error: {0:1.5e}".format(err_))

#     tv_x = tv_nd(x)
#     tv_x_2d = tv(x)
#     err = np.linalg.norm(tv_x - tv_x_2d)
#     print("Error: {0:1.5e}".format(err))

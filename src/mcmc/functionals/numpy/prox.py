"""Math library: implementation of a few selected functions and their proximal
operator.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import copy as cp

import numpy as np
from numba import float64, jit, vectorize

from mcmc.operators.tv import gradient_2d, gradient_2d_adjoint, tv


@jit(nopython=True, cache=True)
def kullback_leibler(x, y):
    r"""Evaluate the Kullback-Leibler (KL) divergence.

    Compute :math:`d_{\text{KL}}(y \mathrel{\Vert} x)`, the Kullback-Leibler
    (KL) divergence between the arrays ``y`` and ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    y : numpy.ndarray
        Input array (first term in the KL divergence).

    Returns
    -------
    float
        Value of the KL divergence :math:`d_{\text{KL}}(y \mathrel{\Vert} x)`.

    Example
    -------
    >>> y = np.full((2, 2), 8)
    >>> x = np.full((2, 2), 5)
    >>> kl_yx = kullback_leibler(x, y)

    Note
    ----
    - By convention :cite:p:`Figueiredo2010`, :math:`0 \log(0) = 0`.
    - An assertion should be added to check ``x`` and ``y`` have the same size.
    """
    # ! issue here: need all entries > 0 to avoid Nan values
    return np.sum(
        x
        - y
        * (
            1
            - np.log(
                np.maximum(y, np.finfo(y.dtype).eps)
                / np.maximum(x, np.finfo(x.dtype).eps)
            )
        )
    )  # add auxiliary constants to ensure KL is >= 0
    # return np.sum(x - y * np.log(np.maximum(x, np.finfo(x.dtype).eps)))
    # if np.any(x <= 0):
    #     kl_yx = np.inf
    # else:
    # mask = y > 0
    # kl_yx = np.sum(x) - np.sum(y[mask] * np.log(x[mask]))
    # return kl_yx


@jit(nopython=True, cache=True)
def prox_kullback_leibler(x, y, lam=1.0):
    r"""Proximal operator of the Kullback-Leibler divergence.

    Evaluate the proximal operator of the Kulllback-Leibler divergence
    :math:`d_{\text{KL}} (y \mathrel{\Vert} \cdot)` in :math:`x`, i.e.
    :math:`\text{prox}_{\lambda d_{\text{KL}} (y \mathrel{\Vert} \cdot)} (x)`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    y : numpy.ndarray
        Input array (first term in the KL divergence).
    lam : float, optional
        Multiplicative constant, by default 1.

    Returns
    -------
    numpy.ndarray
        Evaluation of the proximal operator
        :math:`\text{prox}_{\lambda d_{\text{KL}} (y \mathrel{\Vert} \cdot)} (x)`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.

    Example
    -------
    >>> y = np.full((2, 2), 8)
    >>> x = np.full((2, 2), 5)
    >>> z = prox_kullback_leibler(x, y, lam=1)
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    x1 = x - lam

    return (x1 + np.sqrt(x1**2 + 4 * lam * y)) / 2


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def hard_thresholding(x, thres):  # pragma: no cover
    """Hard-thresholding operator.

    Apply a hard-thresholding operator to the input value ``x``, with
    threshold value ``thres``.

    Parameters
    ----------
    x : float64
        Input scalar.
    thres : float64
        Threshold value.

    Returns
    -------
    float64
        Hard-thresholded input.

    Example
    -------
    >>> x = -1.
    >>> hard_thresholding(x, 0.5)
    """
    if x >= thres:
        return x
    else:
        return 0.0


@jit(nopython=True, cache=True)
def prox_nonegativity(x):
    r"""Projection onto the nonnegative orthant.

    Evaluate the proximal operator of the indicator function
    :math:`\iota_{ \cdot \geq 0}` on the array ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Note
    ----
    The input array ``x`` is modified in-place.

    Example
    -------
    >>> x = np.full((2, 2), -1)
    >>> prox_nonegativity(x)
    """
    for i in range(len(x)):
        x[i] = hard_thresholding(x[i], 0.0)


@jit(nopython=True, cache=True)
def proj_box(x, low, high):
    r"""Projection onto a box set.

    For :math:`x \in \mathbb{R}^N`, compute the projection onto the box
    :math:`C = \prod_{n=1}^N [a_N, b_N]`

    .. math::

        \text{prox}_{\iota_C}(x) = \Big[ \max\big\{ a_n, \min\{ x_n, b_n \}
        \big\} \Big]_{1 \leq n \leq N}.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    low : np.ndarray
        Lower bounds defining the box constraint.
    high : np.ndarray
        Uppers bounds defining the box constraint.

    Returns
    -------
    np.ndarray
        Projected array.

    Raises
    ------
    ValueError
        Checks whether :math:`a_n \leq b_n` for any `n`.

    Example
    -------
    >>> x = np.arrray([0, 2., 10, 3])
    >>> xp = (x, np.ones(x.shape), 2*np.ones(x.shape))
    """
    if (high < low).any():
        raise ValueError(
            "Each entry in `low` should be lower than or equal to the corresponding entry in `high`"
        )
    return np.maximum(low, np.minimum(x, high))
    # ! faster version in pure numpy, not compatible with numba though
    # return np.clip(x, low, high)


# @jit(nopython=True, cache=True)
def l21_norm(x, axis=0):
    r"""Compute the :math:`\ell_{2,1}` norm of an array.

    Compute the :math:`\ell_{2,1}` norm of the input array ``x``, where the
    underlying :math:`\ell_2` norm acts along the specified ``axis``.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the :math:`\ell_2` norm is taken. By default 0.

    Returns
    -------
    float
        :math:`\ell_{2,1}` norm of ``x``.

    Example
    -------
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((2, 2))
    >>> l21_x = l21_norm(x, axis=0)
    """
    return np.sum(np.sqrt(np.sum(x**2, axis=axis)))


@jit(nopython=True, cache=True)
def prox_l21norm_conj(x, lam=1.0, axis=0):
    r"""Proximal operator of the conjugate of the function :math:`\lambda \|\cdot\|_{2,1}`.

    Evaluate the proximal operator of the conjugate of
    :math:`\lambda \|\cdot\|_{2,1}` in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{2,1}^*} (\mathbf{x})`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.
    axis : int, optional
        Axis along which the :math:`\ell_2` norm is taken, by default 0.

    Returns
    -------
    numpy.ndarray
        Evaluation of the proximal operator :math:`\text{prox}_{\lambda \Vert
        \cdot \Vert_{2,1}^*}(\mathbf{x})`.
    """
    return x / np.maximum(np.sqrt(np.sum(x**2, axis=axis)) / lam, 1.0)


# ! to be debugged with jit: creates segfault when using a large number of MPI
# ! cores
# @jit(nopython=True, cache=True)
def prox_l21norm(x, lam=1.0, axis=0):
    r"""Proximal operator of :math:`\ell_{2,1}` norm.

    Evaluate the proximal operator of the :math:`\ell_{2, 1}` norm in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{2,1}} (\mathbf{x})`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.
    axis : int, optional
        Axis along which the :math:`\ell_2` norm is taken, by default 0.

    Returns
    -------
    numpy.ndarray
        Evaluation of the proximal operator :math:`\text{prox}_{\lambda \Vert
        \cdot \Vert_{2,1}}(\mathbf{x})`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.

    Example
    -------
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((2, 2))
    >>> y = prox_l21norm(x, lam=1., axis=0)
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    # return x * (1 - 1 / np.maximum(np.sqrt(np.sum(x**2, axis=axis)), 1.0))
    # return x - lam * prox_l21norm_conj(x / lam, lam=1 / lam, axis=0)
    return x * (1 - 1 / np.maximum(np.sqrt(np.sum(x**2, axis=axis)) / lam, 1.0))


def prox_tv_primal_dual(
    y, tau, lam=1.0, tol=1e-5, max_iter=1e6, verbose=False, rho=1.99
):  # pragma: no cover
    r"""Proximal operator of the discrete TV norm.

    Compute the proximal operator of :math:`\lambda \text{TV}(\mathbf{x})`
    (:math:`\lambda > 0`) with the Condat-Vu primal-dual algorithm
    :cite:p:`Condat2013,Vu2013`.

    Parameters
    ----------
    y : numpy.ndarray
        Input array.
    tau : float
        Proximal parameter :math:`\tau > 0`. Influences the convergence speed.
    lam : float, optional
        TV regularization parameter :math:`\lambda > 0`, by default 1.
    tol : float, optional
        Convergence tolerance (relative variation of the objective function),
        by default 1e-5.
    max_iter : int, optional
        Maximum number of iteration, by default 1e6.
    verbose : bool, optional
        Display convergence monitoring, by default False.
    rho : float, optional
        Relaxation parameter, in :math:`[1, 2[`. By default 1.99.

    Returns
    -------
    x : numpy.ndarray
        Estimated proximal operator.
    crit : numpy.ndarray (vector)
        Evolution of the objective function along the iterations.
    u : numpy.ndarray, of shape ``(2, *x.shape)``
        Dual variable at convergence.

    Note
    ----
    This function corresponds to the original MATLAB implementation associated
    with :cite:p:`Condat2014spl`, available from the
    `author's webpage <https://lcondat.github.io/software.html>`__.
    """
    sigma = 1 / tau / 8  # proximal parameter

    # prox_tau_f = lambda z: (z + tau * y) / (1 + tau)
    # prox_sigma_g_conj = lambda u: u / np.maximum(
    #     np.sqrt(np.sum(u ** 2, axis=0)) / lam, 1
    # )
    # prox_sigma_g_conj = lambda u: u - prox_l21norm(u, lam=lam, axis=0)

    # initialize primal and dual variable
    x2 = cp.deepcopy(y)
    u2 = prox_l21norm_conj(gradient_2d(x2), lam=lam, axis=0)

    # auxiliary variables and convergence monitoring
    cy = np.sum(y**2) / 2
    primalcostlowerbound = 0.0
    stopping_crit = tol + 1
    crit = np.zeros(max_iter)
    count = 0

    while stopping_crit > tol and count < max_iter:
        # x = prox_tau_f(x2 - tau * gradient_2d_adjoint(u2))
        # u = prox_sigma_g_conj(u2 + sigma * gradient_2d(2 * x - x2))
        x = x2 - tau * gradient_2d_adjoint(u2)
        x += tau * y
        x /= 1 + tau
        u = prox_l21norm_conj(u2 + sigma * gradient_2d(2 * x - x2), lam=lam, axis=0)
        x2 += rho * (x - x2)
        u2 += rho * (u - u2)

        # criterion / objective function
        crit[count] = np.sum((x - y) ** 2) / 2 + lam * tv(x)
        if count > 0:
            stopping_crit = np.abs(crit[count] - crit[count - 1]) / np.abs(
                crit[count - 1]
            )

        if verbose and (np.mod(count, 25) == 0):
            dualcost = cy - np.sum((y - gradient_2d_adjoint(u)) ** 2) / 2
            # best value of dualcost computed so far:
            primalcostlowerbound = np.maximum(primalcostlowerbound, dualcost)
            # The gap between primalcost and primalcostlowerbound is even
            # better than between primalcost and dualcost to monitor
            # convergence.
            print(
                r"No iter: {:d}, primal-cost: {:.3e}, primal-bound: {:.3e}, \
                    gap: {:.3e}".format(
                    count,
                    crit[count],
                    primalcostlowerbound,
                    crit[count] - primalcostlowerbound,
                )
            )
        count += 1

    return x, crit, u


def prox_tv_chambolle(
    y, tau=0.249, lam=1.0, tol=1e-3, max_iter=10, verbose=False
):  # pragma: no cover:
    r"""Proximal point operator for the discrete TV norm.

    Uses Chambolle's projection algorithm described in :cite:p:`Chambolle2004`.

    Parameters
    ----------
    y : numpy.ndarray
        Input array.
    tau : float, optional
        Algorithm parameter, by default 0.249.
    lam : float, optional
        Regularization parameter, by default 1.0.
    tol : float, optional
        Tolerance for the stopping criterion, by default 1e-3.
    max_iter : int, optional
        Maximum number of iterations, by default 10.
    verbose : bool, optional
        Display convergence monitoring, by default False.

    Returns
    -------
    x : numpy.ndarray
        Proximal operator of the TV.
    p : numpy.ndarray of size (2, *y.shape)
        Output dual variable.

    Note
    ----
    Adapted from original Matlab code provided by Jose Bioucas-Dias, June 2009,
    (email: bioucas@lx.it.pt)
    """
    # auxiliary variable and convergence monitoring
    err = tol + 1
    count = 0
    p = np.zeros((2, *y.shape))

    while err > tol and count < max_iter:
        # * compute divergence (div = - grad_adjoint)
        div_p = -gradient_2d_adjoint(p)
        u = div_p - y / lam
        # * compute gradient
        up = gradient_2d(u)

        tmp = np.sqrt(np.sum(up**2, axis=0))
        err = np.sqrt(np.sum((-up + tmp * p) ** 2))
        # p = (p + tau * up) / (1 + tau * tmp)
        p += tau * up
        p /= 1 + tau * tmp

        if verbose and (np.mod(count, 25) == 0):
            print(
                r"No iter: {:d}, err TV: {:.3e}".format(
                    count,
                    err,
                )
            )
        count += 1

    x = y + lam * gradient_2d_adjoint(p)

    return x, p


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from numpy.random import PCG64, Generator
#     from PIL import Image

#     # ! /!\ mpimg.imread normalizes the max of the image to 1!
#     # x0 = mpimg.imread('img/parrotgray.png').astype(float)

#     img = Image.open("img/parrotgray.png", "r")
#     x0 = np.asarray(img)
#     x0 = x0 / np.max(x0)

#     niter = 400
#     lam = 0.1
#     tau = 0.01
#     tol = 1e-8

#     # x = np.full((2, 2), -1)
#     # prox_nonegativity(x)
#     # z1 = prox_l21norm(x0, lam=1, axis=0)
#     # z2 = prox_kullback_leibler(x0, x0, lam=1)
#     # z3 = l21_norm(x0)

#     plt.figure()
#     plt.imshow(x0, interpolation="None", cmap=plt.cm.gray)
#     plt.title("Ground truth")
#     plt.colorbar()
#     plt.axis("off")
#     plt.show()

#     rng = Generator(PCG64(0))
#     # isnr = 30
#     # sig2 = np.sum(x0 ** 2) * 10 ** (-isnr/10) / x0.size
#     # y = x0 + np.sqrt(sig2) * rng.standard_normal(size=x0.shape)
#     y = x0 + 0.1 * rng.standard_normal(size=x0.shape)

#     plt.figure()
#     plt.imshow(y, interpolation="None", cmap=plt.cm.gray)
#     plt.title("Noisy image")
#     plt.colorbar()
#     plt.axis("off")
#     plt.show()

#     x, crit, u = prox_tv_primal_dual(y, tau, lam, tol, niter, verbose=True, rho=1.99)

#     plt.figure()
#     plt.plot(crit)
#     plt.xlabel("Objective function")
#     plt.ylabel("Iteration number")
#     plt.show()

#     plt.figure()
#     plt.imshow(x, interpolation="None", cmap=plt.cm.gray)
#     plt.colorbar()
#     plt.axis("off")
#     plt.title("Reconstructed image (Condat2013)")
#     plt.show()

#     x2, u2 = prox_tv_chambolle(
#         y, tau=0.249, lam=lam, tol=tol, max_iter=niter, verbose=True
#     )

#     plt.figure()
#     plt.imshow(x2, interpolation="None", cmap=plt.cm.gray)
#     plt.colorbar()
#     plt.axis("off")
#     plt.title("Reconstructed image (Chambolle2004)")
#     plt.show()

#     pass

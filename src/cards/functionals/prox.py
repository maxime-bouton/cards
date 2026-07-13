r"""Implementation of useful mathematical functions and operators such as simple proximal operators."""

import cards.backend as xp


def prox_nonegativity(x):
    r"""Projection onto the nonnegative orthant.

    Evaluate the proximal operator of the indicator function
    :math:`\iota_{ \cdot \geq 0}` on the array ``x``.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.

    Note
    ----
    The input array ``x`` is modified in-place.

    Example
    -------
    >>> import numpy as np
    >>> x = np.full((2, 2), -1)
    >>> prox_nonegativity(x)
    """
    return xp.maximum(x, 0)


def l21_norm(x, axis=0):
    r"""Compute the :math:`\ell_{2,1}` norm of an array.

    Compute the :math:`\ell_{2,1}` norm of the input array ``x``, where the
    underlying :math:`\ell_2` norm acts along the specified ``axis``.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.
    axis : int, optional
        Axis along which the :math:`\ell_2` norm is taken. By default 0.

    Returns
    -------
    float
        :math:`\ell_{2,1}` norm of ``x``.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((2, 2))
    >>> l21_x = l21_norm(x, axis=0)
    """
    return xp.sum(xp.sqrt(xp.sum(x**2, axis=axis)))


def prox_l21norm(x, lam=1.0, axis=0):
    r"""Proximal operator of :math:`\ell_{2,1}` norm.

    Evaluate the proximal operator of the :math:`\ell_{2, 1}` norm in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{2,1}} (\mathbf{x})`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.
    axis : int, optional
        Axis along which the :math:`\ell_2` norm is taken, by default 0.

    Returns
    -------
    cards.backend.xp.ndarray
        Evaluation of the proximal operator :math:`\text{prox}_{\lambda \Vert
        \cdot \Vert_{2,1}}(\mathbf{x})`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((2, 2))
    >>> y = prox_l21norm(x, lam=1., axis=0)
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return x * (1 - 1 / xp.maximum(xp.sqrt(xp.sum(x**2, axis=axis)) / lam, 1.0))


def KL(x: xp.ndarray, y: xp.ndarray) -> float:
    r"""Evaluate the Kullback-Leibler (KL) divergence.

    Compute :math:`d_{\text{KL}}(y \mathrel{\Vert} x)`, the Kullback-Leibler
    (KL) divergence between the arrays ``y`` and ``x``.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.
    y : cards.backend.xp.ndarray
        Input array (first term in the KL divergence).

    Returns
    -------
    float
        Value of the KL divergence :math:`d_{\text{KL}}(y \mathrel{\Vert} x)`.

    Example
    -------
    >>> import numpy as np
    >>> y = np.full((2, 2), 8)
    >>> x = np.full((2, 2), 5)
    >>> kl_yx = kullback_leibler(x, y)

    Note
    ----
    - By convention :cite:p:`Figueiredo2010`, :math:`0 \log(0) = 0`.
    - An assertion should be added to check ``x`` and ``y`` have the same size.
    """
    eps = xp.finfo(x.dtype).eps
    return xp.sum(x * xp.log(xp.maximum(x, eps) / xp.maximum(y, eps))) + xp.sum(y - x)


def prox_KL(x: xp.ndarray, y: xp.ndarray, lam: float = 1.0) -> xp.ndarray:
    r"""Proximal operator of the Kullback-Leibler divergence.

    Evaluate the proximal operator of the Kulllback-Leibler divergence
    :math:`d_{\text{KL}} (y \mathrel{\Vert} \cdot)` in :math:`x`, i.e.
    :math:`\text{prox}_{\lambda d_{\text{KL}} (y \mathrel{\Vert} \cdot)} (x)`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.
    y : cards.backend.xp.ndarray
        Input array (first term in the KL divergence).
    lam : float, optional
        Multiplicative constant, by default 1.

    Returns
    -------
    cards.backend.xp.ndarray
        Evaluation of the proximal operator
        :math:`\text{prox}_{\lambda d_{\text{KL}} (y \mathrel{\Vert} \cdot)} (x)`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.

    Example
    -------
    >>> import numpy as np
    >>> y = np.full((2, 2), 8)
    >>> x = np.full((2, 2), 5)
    >>> z = prox_kullback_leibler(x, y, lam=1)
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    x1 = x - lam
    return (x1 + xp.sqrt(x1**2 + 4 * lam * y)) / 2


def prox_l1norm(x, lam):
    r"""Proximal operator of :math:`\ell_{1}` norm, a.k.a. soft-thresholding.

    Evaluate the proximal operator of the :math:`\ell_{1}` norm in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{1}} (\mathbf{x})`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : cards.backend.xp.ndarray
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.

    Returns
    -------
    cards.backend.xp.ndarray
        Evaluation of the proximal operator :math:`\text{prox}_{\lambda \Vert
        \cdot \Vert_{1}}(\mathbf{x})`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    return xp.sign(x) * xp.maximum(xp.abs(x) - lam, 0)

""" Implementation of a few useful proximity operators (see for instance
`the Prox Repository<http://proximity-operator.net/>`_).
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# Code from this file is directly adapted from the Python code
# available https://gitlab.cristal.univ-lille.fr/pthouven/dsgs (GPL 3.0
# license), associated with the folllowing paper
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

import torch


@torch.no_grad
def prox_nonnegativity(x):
    r"""Projection onto the nonnegative orthant.

    Evaluate the proximal operator of the indicator function
    :math:`\iota_{\mathbb{R}^N_+}` on the array ``x``

    ..math::
        \text{\prox}_{\iota_{\mathbb{R}^N_+}} (x) = \big( \max\{ x_n, 0 \} \big)_{0 \leq n \leq N-1}.

    Parameters
    ----------
    x : torch.Tensor
        Input array.

    Example
    -------
    >>> x = np.full((2, 2), -1)
    >>> prox_nonnegativity(x)
    """
    # in-place instruction
    # x.clamp_(min=torch.tensor([0.0]))
    return torch.clamp(x, min=torch.tensor([0.0]))


@torch.no_grad
def prox_l1norm(x, threshold=1.0):
    r"""Proximity operator of the :math:`\ell_1`-norm.

    Evaluate the proximity operator of the :math:`\ell_1`-norm

    .. math::
        \text{prox}_{\lambda \|\cdot\|_1}(x) = \max\{ |x| - \lambda, 0 \} \text{sign}(x),

    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    threshold : float, optional
        Regularization parameter :math:`\lambda > 0`, by default 1.

    Returns
    -------
    torch.Tensor
        Evaluation of the proximity operator
        :math:`\text{prox}_{\lambda \|\cdot\|_1}(x)`.

    Example
    -------
    >>> device = torcch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> rng = torch.Generator(device=device)
    >>> rng.manual_seed(1234)
    >>> x = torch.standard_normal((2, 2))
    >>> proxl1_x = prox_l1norm(x, threshold=1.0)
    """
    if threshold <= 0:
        raise ValueError("`threshold` should be positive.")
    return (
        torch.maximum(torch.abs(x) - threshold, torch.zeros(1)) * torch.sign(x)
    ).detach()


@torch.no_grad
def prox_l21norm(x: torch.Tensor, lam=1.0, dim=0):
    r"""Proximal operator of :math:`\lambda \|\cdot\|_{2,1}`.

    Evaluate the proximal operator of the :math:`\ell_{2, 1}` norm in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{2,1}} (\mathbf{x})`
    , with :math:`\lambda > 0`.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.
    dim : int, optional
        Axis along which the :math:`\ell_2` norm is computed, by default 0.

    Returns
    -------
    torch.Tensor
        Evaluation of the proximal operator :math:`\text{prox}_{\lambda \Vert
        \cdot \Vert_{2,1}}(\mathbf{x})`.

    Raises
    ------
    ValueError
        Checks whether :math:`\lambda > 0`.

    Example
    -------
    >>> rng = torch.Generator(device="cpu")
    >>> rng = torch.manual_seed(1234)
    >>> x = torch.normal(0., 1., (2, 2), generator=rng)
    >>> y = prox_l21norm(x, lam=1., dim=0)
    """
    if lam <= 0:
        raise ValueError("`lam` should be positive.")
    # return x * (1 - 1 / np.maximum(np.sqrt(np.sum(x**2, axis=axis)), 1.0))
    # return x - lam * prox_l21norm_conj(x / lam, lam=1 / lam, axis=0)
    return x * (
        1
        - 1
        / torch.maximum(torch.sqrt(torch.sum(x**2, dim=dim)) / lam, torch.ones((1,)))
    )


@torch.no_grad
def prox_l21norm_conj(x: torch.Tensor, lam=1.0, dim=0):
    r"""Proximal operator of the conjugate of
    :math:`\lambda \| \cdot \|_{2, 1}` norm.

    Evaluate the proximal operator of the conjugate of the :math:`\ell_{2, 1}`
    norm in `x`, i.e.
    :math:`\text{prox}_{\lambda \mathrel{\Vert} \cdot \Vert_{2,1}^*} (\mathbf{x})`,
    with :math:`\lambda > 0`.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    lam : float, optional
        Multiplicative constant, by default 1.
    dim : int, optional
        Axis along which the :math:`\ell_2` norm is computed, by default 0

    Returns
    -------
    torch.Tensor
        Evaluation of the proximal operator :math:`\text{prox}_{(\lambda \Vert
        \cdot \Vert_{2,1})^*}(\mathbf{x})`.
    """
    return x / torch.maximum(
        torch.sqrt(torch.sum(x**2, dim=dim)) / lam, torch.ones((1,))
    )


@torch.no_grad
def proj_l2ball(x, radius: float = 1.0, center: torch.Tensor | None = None):
    r"""Compute projection on the :math:`\ell_2` ball.

    Evaluate the proximity operator of :math:`iota_{\mathcal{B}_2(y,
    \varepsilon)}` in :math:`x`, with
    :math:`\mathcal{B}_2(y, \varepsilon)` the :math:`\ell_2` ball with center
    :math:`y` and radius :math:`\varepsilon`.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    radius : torch.Tensor[float], optional
        Radius of the :math:`\ell_2` ball, by default 1.
    center : torch.Tensor | None, optional
        Center of the :math:`\ell_2` ball, by default None (center in 0).

    Returns
    -------
    proj_x: torch.Tensor
        Euclidean projection of ``x`` onto
        :math:`\mathcal{B}_2(y,
    \varepsilon)`.

    Raises
    ------
    ValueError
        Ball `radius` should be positive.
    """
    if radius <= 0:
        raise ValueError("Ball radius should be positive.")
    if center is None:
        proj_x = x * (
            radius / torch.maximum(torch.norm(x, p="fro"), torch.tensor(radius))
        )
    else:
        proj_x = center + (x - center) * (
            radius / torch.maximum(torch.norm(x - center, p="fro"), radius)
        )
    return proj_x


if __name__ == "__main__":
    # TODO: turn into proper unit-tests
    a = torch.tensor([1.0, -1.0], requires_grad=True)
    b = prox_nonnegativity(a)
    c = proj_l2ball(a)

    print("Correctness proj. non-negativity: {}".format(torch.all(b >= 0)))
    print("Correctness proj. l2-ball: {}".format(torch.norm(c) <= 1.0))
    print("Grad. tracking for b: {}".format(b.requires_grad))
    print("Grad. tracking for a: {}".format(a.requires_grad))

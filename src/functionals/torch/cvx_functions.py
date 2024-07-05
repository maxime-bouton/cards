"""Implementation of some convex functions.
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

from dsgs.operators.torch.discrete_gradient import gradient_2d, gradient_2d_adjoint


@torch.no_grad
def l21_norm(x, dim=0):
    r"""Compute the :math:`\ell_{2,1}` norm of an array.

    Compute the :math:`\ell_{2,1}` norm of the input array ``x``, where the
    underlying :math:`\ell_2` norm acts along the specified ``axis``.

    Parameters
    ----------
    x : torch.Tensor
        Input array.
    dim : int, optional
        Axis along which the :math:`\ell_2` norm is taken. By default 0.

    Returns
    -------
    float
        :math:`\ell_{2,1}` norm of ``x``.

    Example
    -------
    >>> device = torcch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> rng = torch.Generator(device=device)
    >>> rng.manual_seed(1234)
    >>> x = torch.standard_normal((2, 2))
    >>> l21_x = l21_norm(x, dim=0)
    """
    return torch.sum(torch.sqrt(torch.sum(x**2, dim=dim)))


@torch.no_grad
def l1_norm(x, dim=0):
    r"""Compute the :math:`\ell_1` norm of an array.

    Parameters
    ----------
    x : torch.Tensor
        Input array.

    Returns
    -------
    float
        :math:`\ell_{1}` norm of ``x``.

    Example
    -------
    >>> device = torcch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> rng = torch.Generator(device=device)
    >>> rng.manual_seed(1234)
    >>> x = torch.standard_normal((2, 2))
    >>> l1_x = l1_norm(x)
    """
    return torch.sum(torch.abs(x))


@torch.no_grad
def tv(x):
    r"""Discrete isotropic total variation (TV).

    Compute the discrete isotropic total variation of a 2d array

    .. math::
       \text{TV}(\mathbf{x}) = \Vert \nabla (\mathbf{x}) \Vert_{2, 1},

    where :math:`\nabla` is the 2d discrete gradient operator.

    Parameters
    ----------
    x : torch.Tensor, 2d
        Input array.

    Returns
    -------
    float
        total variation evaluated in ``x``.
    """
    u = gradient_2d(x)
    return torch.sum(torch.sqrt(torch.sum(torch.abs(u) ** 2, dim=0)))


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rng = torch.Generator(device)
    rng.manual_seed(1234)

    x = torch.normal(0.0, 1.0, (5, 5), generator=rng, device=device)
    y = torch.normal(0.0, 1.0, (2, 5, 5), generator=rng, device=device)

    Ax = gradient_2d(x)
    Aadj_y = gradient_2d_adjoint(y)

    sp1 = torch.sum(Ax * y)
    sp2 = torch.sum(x * Aadj_y)
    print("Correct adjoint implementation? {0}".format(torch.isclose(sp1, sp2)))

    tv_x = tv(x)
    print("TV(x) = {0:1.3e}".format(tv_x.item()))

    pass

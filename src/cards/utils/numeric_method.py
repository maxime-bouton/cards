r"""Short implementation of the power method to evaluate the Lipshotz contsant of operators."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

from typing import Callable, Sequence

import torch

# FIXME: commented cupy variant for now to avoid explicit cupy import


def power_method(
    H: Callable,
    H_adjoint: Callable,
    shape: Sequence[int],
    tol: float = 1e-4,
    max_iter: int = 300,
    rng: torch.Generator | None = None,  # cp.random.BitGenerator
) -> float:
    """
    Computes the largest singular value of an operator using the power method.

    Automatically determines whether H is PyTorch-based or CuPy-based.

    Parameters
    ----------
    H : callable
        Forward operator.
    H_adjoint : callable
        Adjoint (transpose) operator.
    shape : Sequence[int]
        Shape of the input tensor.
    tol : float, optional
        Convergence tolerance (default is 1e-4).
    max_iter : int, optional
        Maximum number of iterations (default is 300).
    rng : torch.Generator | cp.random.BitGenerator, optional
        Random generator for input initialisation.

    Returns
    -------
    float
        Estimated largest singular value.
    """
    # if isinstance(H, torch.nn.Module) or getattr(H, "__module__", "").startswith(
    #     "torch"
    # ):
    xp = torch
    x = torch.rand(*shape, device=next(H.parameters()).device, generator=rng)  # type: ignore
    # else:
    #     xp = cp
    #     if rng is None:
    #         x = cp.random.rand(*shape)
    #     else:
    #         x = rng.rand(shape)  # type: ignore

    x /= xp.linalg.norm(x)
    val = 1.0

    for _ in range(max_iter):
        old_val = val
        x = H_adjoint(H(x))
        val = xp.linalg.norm(x).item()
        if abs(val - old_val) / old_val < tol:
            break
        x /= val

    return val

"""
Created on Dec 1, 2017

@author: mjiang
ming.jiang@epfl.ch

Adapted by: pthouvenin
"""

import numpy as np

# from typing import Callable
# Callable[[np.ndarray], np.ndarray]


def power_method(A, At, im_size, tol: float, max_iter: int, rng: np.random.Generator):
    r"""Compute the spectral radius (squared :math:`\ell_2`:-norm) of a linear
    operator (norm of the compuond operator :math:`A^TA`).

    Parameters
    ----------
    A : function
        Linear operator
    At : function
        Adjoint of the linear operator.
    im_size : numpy.ndarray of int
        Size of the variable on which ``A`` operates.
    tol : float
        Convergence tolerance (relative variation between two consecutive
        iterates).
    max_iter : int
        Maximum number of iterations.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        Squared maximum eigenvalue of the input linear operator.
    """
    x = rng.normal(size=im_size)
    x /= np.sqrt(np.sum(np.abs(x) ** 2))
    init_val = 1.0

    for n in range(max_iter):
        y = A(x)
        x = At(y)
        val = np.sqrt(np.sum(np.abs(x) ** 2))
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            # breakpoint()
            break
        init_val = val
        x /= val

    return val


# if __name__ == "__main__":
#     from dsgs.operators.operator_norm import power_method
#     from dsgs.operators.padding import adjoint_padding, pad_array

#     rng = np.random.default_rng(1234)
#     im_size = (21, 23)

#     patch_size = np.array([6, 6], dtype="i")
#     output_size = [im_size[d] + 2 * (patch_size[d] - 1) for d in range(len(im_size))]
#     lsize = (patch_size - 1).astype(int)
#     stride = np.ones(len(im_size), dtype="i")
#     dilation = np.ones(len(im_size), dtype="i")

#     tol = 1e-4
#     max_iter = 100

#     # * spectral norm of adjoint x patch_extractor = np.prod(patch_size)
#     # A = lambda x: patch_extractor(x, patch_size, stride, dilation)
#     # At = lambda y: adjoint_patch_extractor(y, output_size, patch_size, stride, dilation)

#     # * spectral norm of adjoint x reflect padding: 4 (ok!)
#     A = lambda x: pad_array(x, output_size, padmode="around", mode="reflect")
#     At = lambda y: adjoint_padding(y, lsize, lsize, mode="reflect")

#     # * spectral norm of patch_extractor + reflect padding (120.46 ?)
#     # A = lambda x: patch_extractor(pad_array(x, output_size, padmode="around", mode="reflect"), patch_size, stride, dilation)
#     # At = lambda y: adjoint_padding(adjoint_patch_extractor(y, output_size, patch_size, stride, dilation), lsize, lsize, mode="reflect")

#     sq_norm_A = power_method(A, At, im_size, tol, max_iter, rng)

#     print("Norm of A^T A? {}".format(sq_norm_A))
#     print("Safe upper bound: {}".format(sq_norm_A + tol))
#     pass

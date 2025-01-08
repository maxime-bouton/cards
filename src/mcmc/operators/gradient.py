import numpy as np
from numba import jit

from mcmc.operators.linear_operator import LinearOperator


@jit(nopython=True, cache=True)
def gradient_2d(x: np.ndarray) -> (np.ndarray, np.ndarray):
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
def gradient_2d_adjoint(x: np.ndarray) -> np.ndarray:
    r"""gradient_2d_adjoint Compute the adjoint of the 2d gradient operator.

    Parameters
    ----------
    x : np.ndarray
        Input 2d array

    Returns
    -------
    np.ndarray
        Gradient adjoint.
    """
    v = np.zeros_like(x[0, :, :])
    v[0, :] = -x[1, 0, :]
    v[1:-1, :] = x[1, :-2, :] - x[1, 1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[-1, :] = x[1, -2, :]
    v[:, 0] -= x[0, :, 0]
    v[:, 1:-1] += x[0, :, :-2] - x[0, :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[:, -1] += x[0, :, -2]
    return v


class Gradient2d(LinearOperator):
    def __init__(self, image_size):
        r"""Gradient2d constructor.

        Parameters
        ----------
        image_size : numpy.ndarray of int, of size ``d``
            Full image size.
        """
        super(Gradient2d, self).__init__(
            image_size, np.array([2, *image_size], dtype=int)
        )
        pass

    def forward(self, input_image: np.ndarray):
        r"""forward Compute 2d discrete gradient.

        Parameters
        ----------
        input_image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Discrete gradient.
        """
        return gradient_2d(input_image)

    def adjoint(self, input_data: np.ndarray):
        """adjoint Compute the adjoint of the 2d gradient operator.

        Parameters
        ----------
        input_data : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Adjoint of the 2d discrete gradient.
        """
        return gradient_2d_adjoint(input_data)

"""Implementation of the gradient as a linear operator.
The computations can be done either on CPU or GPU depending on the settings.
"""

from mcmc.operators.linear_operator import LinearOperator
from mcmc.backend import xp


def gradient_2d(x: xp.ndarray):
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected len(x.shape)==2"
    u = xp.zeros((2, *x.shape))
    u[0, :, :-1] = xp.diff(x, 1, 1)  # horizontal differences
    u[1, :-1, :] = xp.diff(x, 1, 0)  # vertical differences
    return u


def gradient_2d_adjoint(x: xp.ndarray) -> xp.ndarray:
    v = xp.zeros_like(x[0, :, :])
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
            image_size, xp.array([2, *image_size], dtype=int)
        )
        pass

    def forward(self, input_image: xp.ndarray):
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

    def adjoint(self, input_data: xp.ndarray):
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

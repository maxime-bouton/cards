"""Implementation of the gradient as a linear operator.
The computations can be done either on CPU or GPU depending on the settings.
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

# TODO: documentation

import cards.backend as xp
from cards.operators.linear_operator import LinearOperator


def gradient_2d(x: xp.ndarray):
    assert len(x.shape) >= 2, "gradient_2d: Invalid input, expected len(x.shape)>=2"
    u = xp.zeros((2, *x.shape), dtype=x.dtype)
    u[0, ..., :, :-1] = xp.diff(x, 1, -1)  # vertical differences
    u[1, ..., :-1, :] = xp.diff(x, 1, -2)  # horizontal differences
    return u


def gradient_2d_adjoint(u: xp.ndarray) -> xp.ndarray:
    v = xp.zeros_like(u[0])
    v[..., 0, :] = -u[1, ..., 0, :]
    v[..., 1:-1, :] = u[1, ..., :-2, :] - u[1, ..., 1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[..., -1, :] = u[1, ..., -2, :]
    v[..., :, 0] -= u[0, ..., :, 0]
    v[..., :, 1:-1] += u[0, ..., :, :-2] - u[0, ..., :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[..., :, -1] += u[0, ..., :, -2]
    return v


class Gradient2d(LinearOperator):
    r"""Serial implementation of the 2d discrete gradient operator.

    Note
    ----
    The horizontal and vertical differences involved in the forward operator act along the last two dimension of the input tensor. Horizontal and vertical gradients are concatenated along axis 0 in the output tensor.
    """

    def __init__(self, image_shape):
        super().__init__(image_shape, [2, *image_shape])

    def forward(self, image: xp.ndarray):
        return gradient_2d(image)

    def adjoint(self, data: xp.ndarray):
        return gradient_2d_adjoint(data)

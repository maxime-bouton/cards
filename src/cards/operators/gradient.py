"""Serial implementation of the 2D discrete gradient operator."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import cards.backend as xp
from cards.operators.linear_operator import LinearOperator


def gradient_2d(x: xp.ndarray) -> xp.ndarray:
    r"""Compute 2d discrete gradient.

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    *i.e.*, by computing horizontal and vertical differences (using jit compilation):

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : xp.ndarray
        Input array.

    Returns
    -------
    u : xp.ndarray
        Vertical and horizontal differences.

    Note
    ----
    The horizontal and vertical differences act along the last two dimension of the input tensor.
    """
    assert len(x.shape) >= 2, "gradient_2d: Invalid input, expected len(x.shape)>=2"
    u = xp.zeros((2, *x.shape), dtype=x.dtype)
    u[0, ..., :, :-1] = xp.diff(x, 1, -1)  # vertical differences
    u[1, ..., :-1, :] = xp.diff(x, 1, -2)  # horizontal differences
    return u


def gradient_2d_adjoint(u: xp.ndarray) -> xp.ndarray:
    r"""Adjoint of the 2d discrete gradient operator.

    Compute the adjoint of the 2d discrete gradient of a 2d input array
    :math:`\mathbf{x}`,

    .. math::
       \nabla^*(\mathbf{y}) = - \text{div} (\mathbf{y})
       = \nabla_v^*\mathbf{y}_v + \mathbf{y}_h\nabla_h^*.

    Parameters
    ----------
    u : numpy.ndarray
        Horizontal and vertical differences

    Returns
    -------
    v : numpy.ndarray
        Adjoint of the 2d gradient operator, evaluated in
        :math:`(\mathbf{u}_h, \mathbf{u}_v)`.

    Note
    ----
    Horizontal and vertical gradients are assumed to be concatenated along the
    axis 0 in the output tensor.
    """
    v = xp.zeros_like(u[0])
    v[..., 0, :] = -u[1, ..., 0, :]
    v[..., 1:-1, :] = u[1, ..., :-2, :] - u[1, ..., 1:-1, :]  # -np.diff(uv[:-1,:],1,0)
    v[..., -1, :] = u[1, ..., -2, :]
    v[..., :, 0] -= u[0, ..., :, 0]
    v[..., :, 1:-1] += u[0, ..., :, :-2] - u[0, ..., :, 1:-1]  # -np.diff(uv[:,:-1],1,1)
    v[..., :, -1] += u[0, ..., :, -2]
    return v


class Gradient2d(LinearOperator):
    r"""Serial implementation of the 2d discrete gradient operator."""

    def __init__(self, image_shape):
        super().__init__(image_shape, [2, *image_shape])

    def forward(self, image: xp.ndarray) -> xp.ndarray:
        return gradient_2d(image)

    def adjoint(self, data: xp.ndarray) -> xp.ndarray:
        return gradient_2d_adjoint(data)

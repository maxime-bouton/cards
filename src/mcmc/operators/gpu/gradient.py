import cupy as cp

from mcmc.operators.linear_operator import LinearOperator


def gradient_2d(x: cp.ndarray) -> cp.ndarray:
    r"""Compute 2d discrete gradient.

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    *i.e.*, by computing horizontal and vertical differences (using jit compilation):

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : cupy.ndarray, 2d
        Input 2d array :math:`\mathbf{x}`.

    Returns
    -------
    u : cupy.ndarray
        Discrete gradient.
    """
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected len(x.shape)==2"
    u = cp.zeros((2, *x.shape))
    u[0, :, :-1] = cp.diff(x, 1, 1)  # horizontal differences
    u[1, :-1, :] = cp.diff(x, 1, 0)  # vertical differences
    return u


def gradient_2d_adjoint(x: cp.ndarray) -> cp.ndarray:
    """gradient_2d_adjoint Compute the adjoint of the 2d gradient operator.

    Parameters
    ----------
    x : cp.ndarray
        Input 2d array

    Returns
    -------
    cp.ndarray
        Gradient adjoint.
    """
    v = cp.zeros_like(x[0, :, :])
    v[0, :] = -x[1, 0, :]
    v[1:-1, :] = x[1, :-2, :] - x[1, 1:-1, :]  # -cp.diff(uv[:-1,:],1,0)
    v[-1, :] = x[1, -2, :]
    v[:, 0] -= x[0, :, 0]
    v[:, 1:-1] += x[0, :, :-2] - x[0, :, 1:-1]  # -cp.diff(uv[:,:-1],1,1)
    v[:, -1] += x[0, :, -2]
    return v


class GpuGradient2d(LinearOperator):
    def __init__(self, image_size):
        r"""GpuGradient2d constructor.

        Parameters
        ----------
        image_size : cp.ndarray of int, of size ``d``
            Full image size.
        """
        super(GpuGradient2d, self).__init__(image_size, (2, *image_size))
        pass

    def forward(self, input_image):
        return gradient_2d(input_image)

    def adjoint(self, input_data):
        return gradient_2d_adjoint(input_data)

""" Implementation of the 2D discrete gradient and its adjoint, with a linear
operator involved when decomposing the operator in the Fourier domain ADMM
splitting (see tutorial 1).
"""
# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import torch
from mlsp.model.linear_operator import LinearOperator

# TODO
# - let students complete `gradient_2d` and its adjoint in torch (tutorial)
# - hints: torch.cat, torch.diff, [None, ...] to add one dimension
# - give unit-test for testing correctness


def gradient_2d(x: torch.Tensor) -> torch.Tensor:
    r"""Compute 2d discrete gradient.

    Compute the 2d discrete gradient of a 2d input array :math:`\mathbf{x}`,
    **i.e.**, by computing horizontal and vertical differences:

    .. math::
       \nabla(\mathbf{x}) = (\nabla_v\mathbf{x}, \mathbf{x}\nabla_h).

    Parameters
    ----------
    x : torch.Tensor
        Input 2d array :math:`\mathbf{x}`.

    Returns
    -------
    torch.Tensor, of shape ``(2, *x.shape)``
        Vertical and horizontal differences, concatenated along the axis 0.
    """
    assert len(x.shape) == 2, "gradient_2d: Invalid input, expected a 2d tensor"
    # vertical differences
    uv = torch.cat((torch.diff(x, dim=0), torch.zeros((1, x.shape[1]))), dim=0)
    # horizontal differences
    uh = torch.cat((torch.diff(x, dim=1), torch.zeros((x.shape[0], 1))), dim=1)
    # ! concatenate along the 1st dimension (slowest access)
    return torch.stack((uv, uh), dim=0)
    # opD = @(x) cat(3,[diff(x,1,1);zeros(1,W)],[diff(x,1,2) zeros(H,1)])


def gradient_2d_adjoint(y: torch.Tensor) -> torch.Tensor:
    r"""Adjoint of the 2d discrete gradient operator.

    Compute the adjoint of the 2d discrete gradient of a 2d input array
    :math:`\mathbf{x}`,

    .. math::
       \nabla^*(\mathbf{y}) = - \text{div} (\mathbf{y})
       = \nabla_v^*\mathbf{y}_v + \mathbf{y}_h\nabla_h^*.

    Parameters
    ----------
    y : torch.Tensor, 3d
        Input array.

    Returns
    -------
    torch.Tensor, of shape ``(y.shape[1], y.shape[2])``
        Adjoint of the 2d gradient operator, evaluated in :math:`\mathbf{y}`.
    """
    return torch.cat(
        (
            -y[0, 0, :][None, ...],
            -torch.diff(y[0, :-1, :], n=1, dim=0),
            y[0, -2, :][None, ...],
        ),
        dim=0,
    ) + torch.cat(
        (
            -y[1, :, 0][..., None],
            -torch.diff(y[1, :, :-1], n=1, dim=1),
            y[1, :, -2][..., None],
        ),
        dim=1,
    )


# torch.cat(
#         (
#             -y[0, 0, :][None, ...],
#             -torch.diff(y[0, :, :], n=1, dim=0),
#         ),
#         dim=0,
#     ) + torch.cat(
#         (
#             -y[1, :, 0][..., None],
#             -torch.diff(y[1, :, :], n=1, dim=1),
#         ),
#         dim=1,
#     )

# opDadj = @(u) -[u(1,:,1);diff(u(:,:,1),1,1)]-[u(:,1,2) diff(u(:,:,2),1,2)];


class DiscreteGradient(LinearOperator):
    """2D discrete gradient linear operator.

    Attributes
    ----------
    image_size : torch.Size
        Size of the input images to which the operator can be applied.
    data_size : torch.Size
        Output size of the 2D discrete gradient operator.
    """

    def __init__(
        self,
        image_size: torch.Size,
    ):
        super(DiscreteGradient, self).__init__(image_size, (2, *image_size))

    forward = staticmethod(gradient_2d)
    adjoint = staticmethod(gradient_2d_adjoint)


class ShiftNcrop(LinearOperator):
    r"""Circular horizontal and vertical shift composed with a crop operator.

    Horizontal and vertical circular shifts composed with a crop operator. This
    operator is used to decompose the implementation of the discrete gradient
    operator in the ADMM algorithm.

    Attributes
    ----------
    image_size: torch.Size
        Size of the input images on which the operator acts.
    """

    def __init__(
        self,
        image_size: torch.Size,
    ):
        """ShiftNcrop constructor.

        Parameters
        ----------
        image_size : torch.Size
            Size of the input images on which the operator acts.

        Raises
        ------
        ValueError
            Only images (2D) are supported.
        """
        if not (len(image_size) == 2):
            raise ValueError("Only images (2D) are supported.")
        super(ShiftNcrop, self).__init__(image_size, image_size)

    def forward(self, input_image: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Circular shifts and cropping.

        Parameters
        ----------
        input_image : list[torch.Tensor] or torch.Tensor of size
        ``(2, *self.image_size)``
            List of images on which the operator is applied.

        Returns
        -------
        torch.Tensor
            Ouput tensor, of size ``(2, *self.image_size)``.
        """
        # horizontal and vertical shifts
        v_shifted_x = torch.roll(input_image[0], -1, dims=0)
        h_shifted_x = torch.roll(input_image[1], -1, dims=1)

        # handle boundaries (autoadjoint operator)
        uh = torch.cat(
            (h_shifted_x[:, :-1], torch.zeros((self.image_size[0], 1))), dim=1
        )
        uv = torch.cat(
            (v_shifted_x[:-1, :], torch.zeros((1, self.image_size[1]))), dim=0
        )

        # return concatenated results (vertical, horizontal)
        return torch.stack((uv, uh), dim=0)

    def adjoint(self, input_data: torch.Tensor) -> torch.Tensor:
        """Adjoint of the circular shifts and cropping.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data, of size ``(2, *self.image_size)``.

        Return
        ------
        torch.Tensor
            Output of the adjoint operator, of size ``(2, *self.image_size)``.
        """
        # handle boundaries (autoadjoint operator)
        uv = torch.cat(
            (input_data[0, :-1, :], torch.zeros((1, self.image_size[1]))), dim=0
        )
        uh = torch.cat(
            (input_data[1, :, :-1], torch.zeros((self.image_size[0], 1))), dim=1
        )

        # adjoint horizontal and vertical shifts
        v_shifted = torch.roll(uv, 1, dims=0)
        h_shifted = torch.roll(uh, 1, dims=1)

        # return concatenated results
        return torch.stack((v_shifted, h_shifted), dim=0)


class ZeroNShift(LinearOperator):
    r"""Circular horizontal and vertical shift composed with a zeroing
    operator.

    Horizontal and vertical circular shifts composed with a crop operator. This
    operator is used to decompose the implementation of the discrete gradient
    operator in the ADMM algorithm.

    Attributes
    ----------
    image_size: torch.Size
        Size of the input images on which the operator acts.
    """

    def __init__(
        self,
        image_size: torch.Size,
    ):
        """ZeroNShift constructor.

        Parameters
        ----------
        image_size : torch.Size
            Size of the input images on which the operator acts.

        Raises
        ------
        ValueError
            Only images (2D) are supported.
        """
        if not (len(image_size) == 2):
            raise ValueError("Only images (2D) are supported.")
        super(ZeroNShift, self).__init__(image_size, image_size)

    def forward(self, input_image: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Circular shifts and cropping.

        Parameters
        ----------
        input_image : list[torch.Tensor] or torch.Tensor of size
        ``(2, *self.image_size)``
            List of images on which the operator is applied.

        Returns
        -------
        torch.Tensor
            Ouput tensor, of size ``(2, *self.image_size)``.
        """
        # handle boundaries (zeroing)
        xh = torch.cat(
            (torch.zeros((self.image_size[0], 1)), input_image[1, :, 1:]), dim=1
        )
        xv = torch.cat(
            (torch.zeros((1, self.image_size[1])), input_image[0, 1:, :]), dim=0
        )

        # horizontal and vertical shifts
        uh = torch.roll(xh, -1, dims=1)
        uv = torch.roll(xv, -1, dims=0)

        # return concatenated results (vertical, horizontal)
        return torch.stack((uv, uh), dim=0)

    def adjoint(self, input_data: torch.Tensor) -> torch.Tensor:
        """Adjoint of the circular shifts and cropping.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data, of size ``(2, *self.image_size)``.

        Return
        ------
        torch.Tensor
            Output of the adjoint operator, of size ``(2, *self.image_size)``.
        """
        # adjoint horizontal and vertical shifts
        uv = torch.roll(input_data[0], 1, dims=0)
        uh = torch.roll(input_data[1], 1, dims=1)

        # handle boundaries (autoadjoint operator)
        xv = torch.cat((torch.zeros((1, self.image_size[1])), uv[1:, :]), dim=0)
        xh = torch.cat((torch.zeros((self.image_size[0], 1)), uh[:, 1:]), dim=1)

        # return concatenated results
        return torch.stack((xv, xh), dim=0)


if __name__ == "__main__":
    from mlsp.model.cconv import CircularConvolutions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    image_size = torch.Size((5, 6))

    dg = DiscreteGradient(image_size)

    # shift/crop operator
    shiftncrop_op = ShiftNcrop(image_size)
    zeronshift_op = ZeroNShift(image_size)

    # convolution operators (vertical and horizontal)
    kernel = torch.tensor([1, -1], device=device)
    # ! full FFT needed for axis 0, given that rfft2 only saves size on last axis 1
    v_fft_kernel = torch.fft.fft(torch.tensor([1, -1]), n=image_size[0])[:, None]
    h_fft_kernel = torch.fft.rfft(torch.tensor([1, -1]), n=image_size[1])[None, :]
    fft_kernels = [v_fft_kernel, h_fft_kernel]  # vertical, horizontal
    cconv_op = CircularConvolutions(image_size, fft_kernels)

    x = torch.normal(0.0, 1.0, image_size, generator=rng, device=device)
    y = torch.normal(0.0, 1.0, (2, *image_size), generator=rng, device=device)

    # * consistency adjoint operator ShiftNcrop (ok)
    x_test = torch.stack((x, x), dim=0)
    op_x = shiftncrop_op.forward(x_test)
    op_adj_y = shiftncrop_op.adjoint(y)

    sp1 = torch.sum(op_x * y)
    sp2 = torch.sum(x_test * op_adj_y)
    print(
        "Correct adjoint implementation (shift and crop)? {0}".format(
            torch.isclose(sp1, sp2)
        )
    )

    # * consistency adjoint operator ZerotNshift (ok)
    op_x = zeronshift_op.forward(x_test)
    op_adj_y = zeronshift_op.adjoint(y)

    sp1 = torch.sum(op_x * y)
    sp2 = torch.sum(x_test * op_adj_y)
    print(
        "Correct adjoint implementation (zero and shift)? {0}".format(
            torch.isclose(sp1, sp2)
        )
    )

    # * consistency discrete gradient and composition ShiftNcrop with cconvs

    # direct operator (ok)
    ref_forward = gradient_2d(x)
    comp_forward = shiftncrop_op.forward(torch.stack(cconv_op.forward(x), dim=0))
    ref_class = dg.forward(x)

    print(
        "Correct compound implementation (forward)? {0}".format(
            torch.allclose(ref_forward, comp_forward)
            and torch.allclose(ref_forward, ref_class)
        )
    )

    # adjoint operator (ok)
    ref_adjoint = gradient_2d_adjoint(y)
    x_temp = shiftncrop_op.adjoint(y)
    comp_adjoint = cconv_op.adjoint([x_temp[0], x_temp[1]])

    print(
        "Correct compound implementation (adjoint)? {0}".format(
            torch.allclose(ref_adjoint, comp_adjoint)
        )
    )

    # * consistency discrete gradient and composition ShiftNcrop with cconvs

    # direct operator (ok)
    ref_forward = gradient_2d(x)
    comp_forward = zeronshift_op.forward(torch.stack(cconv_op.forward(x), dim=0))

    print(
        "Correct compound implementation (forward)? {0}".format(
            torch.allclose(ref_forward, comp_forward)
        )
    )

    # adjoint operator (ok)
    ref_adjoint = gradient_2d_adjoint(y)
    x_temp = zeronshift_op.adjoint(y)
    comp_adjoint = cconv_op.adjoint([x_temp[0], x_temp[1]])

    print(
        "Correct compound implementation (adjoint)? {0}".format(
            torch.allclose(ref_adjoint, comp_adjoint)
        )
    )

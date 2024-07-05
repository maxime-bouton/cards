import torch

from dsgs.operators.linear_operator import LinearOperator


class CircularConvolution(LinearOperator):
    r"""Model class implementing an image-size circular convolution operator
    for real-valued input images.

    Model class implementing an image-size circular convolution operator
    for real-valued input images. Computations are based on the FFT algorithm
    dedicated to real-valued arrays (rFFT, ``torch.fft.rfft2``).

    Attributes
    ----------
    image_size : torch.Size
        Size of the image to be convolved. The output of the convolution
        convolution is of the same size.
    fft_kernel : torch.Tensor
        rFFT of the 2D convolution kernel.
    """

    def __init__(
        self,
        image_size: torch.Size,
        fft_kernel: torch.Tensor,
    ):
        r"""CircularConvolution constructor.

        Parameters
        ----------
        image_size : torch.Size
            Size of the image to be convolved. The output of the convolution
            convolution is of the same size.
        fft_kernel : torch.Tensor
            rFFT of the 2D convolution kernel.

        Raises
        ------
        ValueError
            Only images (2D) are supported.
        ValueError
            Only 2D convolution kernels are supported.
        """
        if not (len(image_size) == 2):
            raise ValueError("Only images (2D) are supported.")
        if not (fft_kernel.dim() == 2):
            raise ValueError("Only 2D convolution kernels are supported.")
        self.fft_kernel = fft_kernel
        super(CircularConvolution, self).__init__(image_size, image_size)

    def forward(self, input_image: torch.Tensor):
        r"""Forward Fourier-based circular convolution.

        Parameters
        ----------
        input_image : torch.Tensor
            Input image of size ``self.image_size``.

        Returns
        -------
        torch.Tensor
            Circular convolution with the kernel stored in the object.
        """
        return torch.fft.irfft2(
            self.fft_kernel * torch.fft.rfft2(input_image, s=self.image_size),
            s=self.image_size,
        )

    def fft_forward(self, fft_input_image: torch.Tensor):
        r"""Forward Fourier-based circular convolution.

        Parameters
        ----------
        fft_input_image : torch.Tensor
            Fourier transform of the input image, obtained with
            ``torch.fft.rfft2``.

        Returns
        -------
        torch.Tensor
            Circular convolution with the kernel stored in the object.
        """
        return torch.fft.irfft2(self.fft_kernel * fft_input_image, s=self.image_size)

    def adjoint(self, input_data: torch.Tensor):
        r"""Adjoint Fourier-based circular convolution.

        Parameters
        ----------
        input_data : torch.Tensor
            Input image, of size ``self.image_size``.

        Returns
        -------
        torch.Tensor
            Adjoint circular convolution with the kernel stored in the object.
        """
        return torch.fft.irfft2(
            torch.conj(self.fft_kernel) * torch.fft.fft2(input_data, s=self.image_size),
            s=self.image_size,
        )

    def fft_adjoint(self, fft_input_data: torch.Tensor):
        r"""Adjoint Fourier-based circular convolution.

        Parameters
        ----------
        fft_input_data : torch.Tensor
            Fourier transform of the input image, obtained with
            ``torch.fft.rfft2``.

        Returns
        -------
        torch.Tensor
            Adjoint circular convolution with the kernel stored in the object.
        """
        return torch.fft.irfft2(
            torch.conj(self.fft_kernel) * fft_input_data, s=self.image_size
        )


class CircularConvolutions(LinearOperator):
    r"""Model class implementing an image-size circular convolution operator
    for real-valued input images.

    Model class implementing an image-size circular convolution operator
    for real-valued input images. Computations are based on the FFT algorithm
    dedicated to real-valued arrays (rFFT, ``torch.fft.rfft2``).

    Attributes
    ----------
    image_size : torch.Size
        Size of the image to be convolved. The output of the convolution
        convolution is of the same size.
    fft_kernel : list[torch.Tensor]
        rFFT of the 2D convolution kernel.
    """

    def __init__(
        self,
        image_size: torch.Size,
        fft_kernels: list[torch.Tensor],
    ):
        """CircularConvolutions constructor.

         Parameters
         ----------
         image_size : torch.Size
             Size of the image to be convolved. The output of the convolution
             convolution is of the same size.
         fft_kernel : list[torch.Tensor]
             rFFT of the 2D convolution kernel.

        Raises
         ------
         ValueError
             Only images (2D) are supported.
         ValueError
             Only 2D convolution kernels are supported.
        """
        if not (len(image_size) == 2):
            raise ValueError("Only images (2D) are supported.")
        if not torch.all(
            torch.tensor(
                [fft_kernels[ind].dim() == 2 for ind in range(len(fft_kernels))]
            )
        ):
            raise ValueError("Only 2D convolution kernels are supported.")

        self.fft_kernels = fft_kernels
        super(CircularConvolutions, self).__init__(image_size, image_size)

    def forward(self, input_image: torch.Tensor):
        r"""Forward Fourier-based circular convolutions.

        Parameters
        ----------
        input_image : torch.Tensor
            Input image of size ``self.image_size``.

        Returns
        -------
        list[torch.Tensor]
            Circular convolution with each kernel stored in the object.
        """
        output = []
        fft_input_image = torch.fft.rfft2(input_image, s=self.image_size)
        for ind in range(len(self.fft_kernels)):
            output.append(
                torch.fft.irfft2(
                    self.fft_kernels[ind] * fft_input_image, s=self.image_size
                )
            )
        return output

    def fft_forward(self, fft_input_image: torch.Tensor):
        r"""Forward Fourier-based circular convolutions.

        Parameters
        ----------
        fft_input_image : torch.Tensor
            Fourier transform of the input image, obtained with
            ``torch.fft.rfft2``.

        Returns
        -------
        list[torch.Tensor]
            Circular convolution with each kernel stored in the object.
        """
        output = []
        for ind in range(len(self.fft_kernels)):
            output.append(
                torch.fft.irfft2(
                    self.fft_kernels[ind] * fft_input_image, s=self.image_size
                )
            )
        return output

    def adjoint(self, input_data: list[torch.Tensor]):
        r"""Adjoint Fourier-based circular convolutions.

        Parameters
        ----------
        input_data : list[torch.Tensor]
            Input images, of size ``self.image_size``.

        Returns
        -------
        torch.Tensor
            Sum of the adjoint circular convolutions with the kernels stored in
            the object.
        """
        output = torch.zeros(self.image_size)
        for ind in range(len(self.fft_kernels)):
            fft_data = torch.fft.rfft2(input_data[ind], s=self.image_size)
            output += torch.fft.irfft2(
                torch.conj(self.fft_kernels[ind]) * fft_data, s=self.image_size
            )
        return output

    def fft_adjoint(self, fft_input_data: list[torch.Tensor]):
        r"""Adjoint Fourier-based circular convolutions.

        Parameters
        ----------
        fft_input_data : list[torch.Tensor]
            Input images, of size ``self.image_size``.

        Returns
        -------
        torch.Tensor
            Sum of the adjoint circular convolutions with the kernels stored in
            the object.
        """
        output = torch.zeros(self.image_size)
        for ind in range(len(self.fft_kernels)):
            output += torch.fft.irfft2(
                torch.conj(self.fft_kernels[ind]) * fft_input_data[ind],
                s=self.image_size,
            )
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    image_size = torch.Size((5, 10))

    # ! full FFT needed for axis 0, given that rfft2 only saves size on last axis 1
    v_fft_kernel = torch.fft.fft(torch.tensor([1, -1]), n=image_size[0])[:, None]
    h_fft_kernel = torch.fft.rfft(torch.tensor([1, -1]), n=image_size[1])[None, :]
    fft_kernels = [v_fft_kernel, h_fft_kernel]  # vertical, horizontal
    op = CircularConvolutions(image_size, fft_kernels)

    # check consistency adjoint operator
    x = torch.normal(0.0, 1.0, image_size, generator=rng, device=device)
    fft_x = torch.fft.rfft2(x, s=image_size)

    y = [
        torch.normal(0.0, 1.0, image_size, generator=rng, device=device),
        torch.normal(0.0, 1.0, image_size, generator=rng, device=device),
    ]
    fft_y = [torch.fft.rfft2(y[0], s=image_size), torch.fft.rfft2(y[1], s=image_size)]

    opfft_x = op.fft_forward(fft_x)
    op_x = op.forward(x)
    print(
        "Correct implementation (forward)? {0}".format(
            torch.allclose(opfft_x[0], op_x[0]) and torch.allclose(opfft_x[1], op_x[1])
        )
    )

    opfft_adj_y = op.fft_adjoint(fft_y)
    op_adj_y = op.adjoint(y)
    print(
        "Correct implementation (adjoint)? {0}".format(
            torch.allclose(opfft_adj_y, op_adj_y)
        )
    )

    sp1 = torch.sum(op_x[0] * y[0]) + torch.sum(op_x[1] * y[1])
    sp2 = torch.sum(x * op_adj_y)
    print(
        "Consistent implementation for the adjoint? {0}".format(torch.isclose(sp1, sp2))
    )

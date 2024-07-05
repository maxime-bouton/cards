from math import floor

import torch

from dsgs.operators.linear_operator import LinearOperator

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# Code from this file is directly adapted from the Python code
# available https://gitlab.cristal.univ-lille.fr/pthouven/dsgs (GPL 3.0
# license), associated with the folllowing paper
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.


def generate_random_mask(image_size, percent, rng):
    r"""Generate a random inpainting mask.

    Parameters
    ----------
    image_size : torch.Size
        Shape of the image to be masked.
    percent : float
        Fraction of observed pixels, ``1-p`` corresponding to the fraction of
        masked pixels in the image (``p`` needs to be nonnegative, smaller
        than or equal to 1).
    rng : torch._C.Generator
        Torch random number generator.

    Returns
    -------
    mask_id : torch.Tensor of int
        1D index of the observed pixels.
    mask : torch.Tensor of bool
        Boolean array such that ``mask.shape == image_size``.

    Raises
    ------
    ValueError
        Fraction of observed pixels ``percent`` should be such that
        :math:`0 \leq p \leq 1`.
    """
    if percent < 0 or percent > 1:
        raise ValueError(
            "Fraction of observed pixels percent should be such that: 0 <= percent <= 1."
        )
    # mask = torch.empty(image_size, dtype=bool)
    # mask.bernoulli_(p=percent, generator=rng)
    # mask_id = torch.nonzero(mask, as_tuple=True)

    # ! https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/7
    # counterpart of numpy.random.choice (uniform sampling without replacement)
    num_samples = floor(image_size[0] * image_size[1] * percent)
    probabilities = torch.ones((image_size[0] * image_size[1],))
    mask_id = probabilities.multinomial(
        num_samples=num_samples, replacement=False, generator=rng
    )

    mask = torch.full((image_size[0] * image_size[1],), False, dtype=bool)
    mask[mask_id] = True
    mask = torch.reshape(mask, image_size)

    return mask_id, mask


class Inpainting(LinearOperator):
    r"""Model class implementing an inpainting operator in a serial algorithm.

    Attributes
    ----------
    image_size : torch.Size, containing ``d`` elements.
        Full image size.
    mask_id : torch.Tensor of int
        Array of indices corresponding to the observed points.
    """

    def __init__(
        self,
        image_size,
        mask_id,
    ):
        r"""Inpainting constructor.

        Parameters
        ----------
        image_size : torch.Size, containing ``d`` elements.
            Full image size.
        mask_id : torch.Tensor of int
            Array of indices corresponding to the observed points.

        Raises
        ------
        ValueError
            Observed pixel indices in ``mask_id`` exceed number of image pixels.
        """
        if torch.max(mask_id) > torch.tensor(image_size[0] * image_size[1]):
            raise ValueError(
                "Observed pixel indices in ``mask_id`` exceed number of image pixels."
            )

        # data_size = torch.Size((torch.sum(mask),))
        # self.mask = mask
        # self.mask_id = torch.nonzero(mask, as_tuple=True)

        # ! flattened data dimension after applying the inpainting operator
        # ! (= cropping)
        data_size = mask_id.shape
        super(Inpainting, self).__init__(image_size, data_size)
        self.mask_id = mask_id

    def forward(self, input_image):
        r"""Implementation of the direct operator to update the input array
        ``input_image`` (from image to data space, cropping).

        Parameters
        ----------
        input_image : torch.Tensor of float
            Input array (image space).

        Returns
        -------
        torch.Tensor
            Result of the inpaiting operator (direct operator).
        """
        # return input_image[self.mask_id]
        return torch.take(input_image, self.mask_id)

    def adjoint(self, input_data):
        r"""Implementation of the adjoint operator to update the input array
        ``input_data`` (from data to image space, zero-padding).

        Parameters
        ----------
        input_data : torch.Tensor of float
            Input array (data space).

        Returns
        -------
        torch.Tensor
            Result of the adjoint of the inpainting operator.
        """
        # output_image = torch.zeros([*self.image_size], dtype=input_data.dtype)
        # output_image[self.mask_id] = input_data
        output_image = torch.zeros(
            (self.image_size[0] * self.image_size[1],), dtype=input_data.dtype
        )
        output_image[self.mask_id] = input_data
        return torch.reshape(output_image, self.image_size)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mlsp.utils.lab_utils import load_and_normalize_image

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    image_size = torch.Size((10, 10))
    percent = 0.6

    ids, mask = generate_random_mask(image_size, percent, rng)
    inpainting = Inpainting(image_size, ids)

    print("Number of masked entries: {}".format(ids.numel()))

    # # test implementation of the direct and adjoint operators
    # x = torch.normal(0.0, 1.0, image_size, generator=rng, device=device)
    # y = torch.normal(0.0, 1.0, inpainting.data_size, generator=rng, device=device)

    # Ax = inpainting.forward(x)
    # Aadj_y = inpainting.adjoint(y)

    # sp1 = torch.sum(Ax * y)
    # sp2 = torch.sum(x * Aadj_y)
    # print("Correct adjoint implementation? {0}".format(torch.isclose(sp1, sp2)))

    # test autodiff across inpainting object
    # https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
    # https://leimao.github.io/blog/PyTorch-Leaf-Tensor/
    x1 = torch.normal(
        0.0, 1.0, image_size, generator=rng, device=device, requires_grad=True
    )
    loss = torch.sum(inpainting.forward(x1) ** 2) / 2
    print(f"Gradient function for loss = {loss.grad_fn}")
    loss.backward()
    # loss.backward(torch.ones_like(loss), retain_graph=True)

    with torch.no_grad():
        reference_gradient = inpainting.adjoint(inpainting.forward(x1))

    print(
        "Correct autodiff. across inpainting operator? {0}".format(
            torch.allclose(x1.grad, reference_gradient)
        )
    )

    # * visualize inpainting operator on an image
    imagefilename = "/Users/pthouvenin/Documents/teaching/m2ds/mlsp/labs/img/house.h5"
    im = load_and_normalize_image(imagefilename, device)
    ids, mask = generate_random_mask(im.shape, percent, rng)
    inpainting = Inpainting(im.shape, ids)
    obs = inpainting.forward(im)
    masked_data = inpainting.adjoint(obs)

    plt.figure()
    plt.imshow(masked_data, interpolation="None", cmap=plt.cm.gray)
    plt.colorbar()
    plt.axis("off")
    plt.title("Reconstructed image")
    plt.show()

    pass

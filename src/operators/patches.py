import numpy as np

# from numba import jit

## ! Using tuple from numba jitted function
## https://numba-how-to.readthedocs.io/en/latest/tuples.html
# from numba.extending import overload
# from numba import types
# from numba.extending import intrinsic
# from numba.core.cgutils import unpack_tuple

# def tuple_zip(*args):
#     return tuple(zip(args))

# @overload(tuple_zip)
# def tuple_zip_ovrl(*args):
#     return tuple_zip_intr

# @intrinsic
# def tuple_zip_intr(tyctx, *tys):
#     if len(tys) > 1:
#         tys = types.StarArgTuple(tys)
#     elif len(tys) == 1:
#         raise ValueError("Only one argument received. Tuples to be zipped must be passed as individual arguments")
#     nitems = min((x.count for x in tys))
#     tuples = [types.Tuple(inner_ty) for inner_ty in zip(*tys)]
#     ret = types.Tuple(tuples)

#     from numba.core.cgutils import unpack_tuple
#     def codegen(cgctx, builder, sig, args):
#         assert len(args) == 1  # it is a vararg tuple
#         args_tup = unpack_tuple(builder, args[0])
#         values = []
#         for i in range(nitems):
#             inner_vals = [builder.extract_value(x, i) for x in args_tup]
#             inner_tup = cgctx.make_tuple(builder, tuples[i], inner_vals)
#             values.append(inner_tup)
#         return cgctx.make_tuple(builder, sig.return_type, values)
#     sig = ret(tys)
#     return sig, codegen


# @jit(nopython=True, cache=True)
def patch_extractor(
    x: np.ndarray,
    patch_size: np.ndarray[int],
    stride: np.ndarray[int],
    dilation: np.ndarray[int],
):
    """Extract all possible patches from an input array.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    patch_size : numpy.ndarray[int]
        Size of the patches to be extracted from the input array.
    stride : numpy.ndarray[int]
        For each axis, stride to extract contiguous patches from the input
        array.
    dilation : numpy.ndarray[int]
        For each axis, number of elements in the input array between two
        consecutive elements from the patch.

    Returns
    -------
    numpy.ndarray
        Collection of patches extracted. The spatial axes of the patches are
        preserved.

    Note
    ----
    Patch extraction is performed as long as the selection window does not go
    out of the scope of the input array (i.e., border elments may not be
    considered in the patch collection).
    """
    input_size = np.array(x.shape, dtype="i")
    ndims = len(x.shape)
    # patch_number = np.ceil((input_size - dilation * (patch_size - stride)) / stride).astype(int)
    # ! torch.nn.Unfold does not go out of bound, even if some elements remain to be taken (does so only if implicit 0-padding is used)
    patch_number = np.floor_divide(
        (input_size - dilation * (patch_size - stride)), stride
    )  # .astype(int)
    npatches = np.prod(patch_number)

    # ! need to put 0 if patch_size does not divide the image size?
    # ! need to accound for the remaining patch if needed
    patches = np.empty((npatches, *patch_size), dtype=x.dtype)

    for id_patch in range(npatches):
        # coordinates of the patch
        c = np.unravel_index(id_patch, patch_number, order="C")

        # ! revise use of max / maximum in padding.py (not sure this is what I
        # ! intended originally, if padding > 1
        sel = tuple(
            [
                np.s_[
                    c[d]
                    * stride[d] : np.minimum(
                        c[d] * stride[d] + dilation[d] * patch_size[d], input_size[d]
                    ) : dilation[d]
                ]
                for d in range(ndims)
            ]
        )
        # sel = tuple_zip([np.s_[c[d]*stride[d]:np.minimum(c[d]*stride[d] + dilation[d]*patch_size[d], input_size[d]):dilation[d]] for d in range(ndims)])

        patches[id_patch] = x[sel]

    return patches


def adjoint_patch_extractor(
    patches: np.ndarray,
    output_size: np.ndarray[int],
    patch_size: np.ndarray[int],
    stride: np.ndarray[int],
    dilation: np.ndarray[int],
):
    """Adjoint of the patch extraction operator.

    Parameters
    ----------
    patches : numpy.ndarray
        Collection of patches.
    output_size : numpy.ndarray[int]
        Size of the array to be formed.
    patch_size : numpy.ndarray[int]
        Size of the patches to be extracted from the input array.
    stride : numpy.ndarray[int]
        For each axis, stride to extract contiguous patches from the input
        array.
    dilation : numpy.ndarray[int]
        For each axis, number of elements in the input array between two
        consecutive elements from the patch.

    Returns
    -------
    numpy.ndarray
        Output array, with size specified in ``output_size``.
    """
    ndims = len(output_size)
    # patch_number = np.ceil((input_size - dilation * (patch_size - stride)) / stride).astype(int)
    # ! torch.nn.Unfold does not go out of bound, even if some elements remain to be taken (does so only if implicit 0-padding is used)
    patch_number = np.floor_divide(
        (output_size - dilation * (patch_size - stride)), stride
    )
    # npatches = np.prod(patch_number)
    npatches = patches.shape[0]

    # ! need to put 0 if patch_size does not divide the image size?
    # ! need to accound for the remaining patch if needed
    output = np.zeros(output_size, dtype=patches.dtype)

    for id_patch in range(npatches):
        # coordinates of the patch
        c = np.unravel_index(id_patch, patch_number, order="C")

        # ! revise use of max / maximum in padding.py (not sure this is what I
        # ! intended originally, if padding > 1
        sel = tuple(
            [
                np.s_[
                    c[d]
                    * stride[d] : np.minimum(
                        c[d] * stride[d] + dilation[d] * patch_size[d], output_size[d]
                    ) : dilation[d]
                ]
                for d in range(ndims)
            ]
        )
        output[sel] += patches[id_patch]
    return output


# if __name__ == "__main__":
#     import torch
#     import torch.nn as nn

#     from dsgs.utils.torch import np_to_tensor

#     device = torch.device("cpu")
#     Tensor = torch.FloatTensor
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     torch.set_num_threads(1)
#     rng = np.random.default_rng(1234)

#     # ndims = 2
#     # x = np.array(
#     #     [
#     #         [1, 2, 3, 4],
#     #         [5, 6, 7, 8],
#     #         [9, 10, 11, 12],
#     #     ]
#     # )

#     ndims = 2  # number of dimensions
#     psize = 6  # patch size
#     # ! check size after padding in Maxime's algorithm
#     # after adding extension in Maxime's case
#     input_size = (60, 60)  # (50, 50) + 5 on each side -> (60, 60)
#     # (5, 4)  # input size
#     stride = np.ones(ndims, dtype="i")
#     dilation = np.ones(ndims, dtype="i")  # np.array([2, 3], dtype="i")
#     patch_size = np.full((ndims,), psize, dtype="i")

#     x = rng.standard_normal(input_size)
#     patches = patch_extractor(x, patch_size, stride, dilation)
#     np_patches = np.reshape(
#         patches, (patches.shape[0], patches.shape[1] * patches.shape[2])
#     )

#     # print("p[0] = {}".format(patches[0]))

#     # ! comparison with torch.nn.Unfold (ok for configurations tested)
#     torch_x = np_to_tensor(x)
#     im2pat = nn.Unfold(kernel_size=patch_size, dilation=dilation, stride=stride)
#     torch_patches = im2pat(torch_x.unsqueeze(0)).squeeze(0).transpose(1, 0)

#     print(torch_patches.shape)
#     print(
#         "Are the results consistent? {}".format(np.allclose(torch_patches, np_patches))
#     )

#     # ! testing adjoint implementation
#     y = rng.standard_normal(patches.shape)
#     E_adj_y = adjoint_patch_extractor(
#         y, np.array([*x.shape], dtype="i"), patch_size, stride, dilation
#     )

#     sp1 = np.sum(patches * y)
#     sp2 = np.sum(x * E_adj_y)

#     print("Is adjoint correct? {}".format(np.isclose(sp1, sp2)))

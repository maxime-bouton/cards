r"""Save state of random number generators in numpy
(``np.random.Generator`` class) and torch (``torch._C.Generator`` class) to a
``.h5`` file using ``h5py``.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

import sys

import h5py
import numpy as np
import torch


def int_to_array(n: int) -> np.ndarray[np.uint8]:
    r"""Convert a built-on Python ``int`` to an array which can be saved in
    hdf5.

    Parameters
    ----------
    n : int
        Input `int`` value.

    Returns
    -------
    n_array
        Output array conversion.

    Note
    ----
    Built-in Python ``int``s, as those used to describe the state of a numpy
    random number generator, can be very large. They need to be converted to
    hexadecimal format, and from that into an array of ``np.uint8``, in order
    to be saved into an .h5 file.
    """
    # Reference:
    # https://docs.python.org/3/library/stdtypes.html#int.to_bytes
    # ! need 32 bytes in length: otherwise, the inverse operation
    # ! int.from_bytes(state_array,sys.byteorder) does not coincide with the
    # ! original int value
    # ! entries in the resulting array are of type np.uint8
    int_array = np.array(bytearray(n.to_bytes(32, sys.byteorder)))
    return int_array


def array_to_int(n_array: np.ndarray[np.uint8]) -> int:
    r"""Convert a numpy array of ``np.uint8`` back to a built-in Python ``int``.

    Inverse operation of :func:`mcmc.DataManager.warmstart_rng.int_to_array`.

    Parameters
    ----------
    n_array : np.ndarray[np.uint8]
        Input array.

    Returns
    -------
    int
        Output `int`` value.
    """
    return int.from_bytes(n_array, sys.byteorder)


def extract_rng_state(
    rng: np.random.Generator,
) -> (np.ndarray[np.uint8], np.ndarray[np.uint8]):
    r"""Extract the state of a random number generator in the form of two
    ``numpy.ndarray`` objects.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    state_array : np.ndarray[np.uint8]
        State parameter of the input random generator.
    inc_array : np.ndarray[np.uint8]
        Increment parameter of the input random generator.

    Note
    ----
    - Requires ``numpy>=2.0``.
    - The ``state`` and ``inc`` fields of a ``numpy.random.Generator`` object
    are very large integers, and thus need to be converted to the
    hexadecimal format (and later to an array of ``int``) to be saved into
    a ``.h5`` file.
    """
    # * state and inc are very large integers, and thus need to be
    # converted to hex format (later to an array of ints) to be saved in an
    # .h5 file
    state_array = int_to_array(rng.bit_generator.__getstate__()[0]["state"]["state"])
    inc_array = int_to_array(rng.bit_generator.__getstate__()[0]["state"]["inc"])
    return state_array, inc_array


def restore_rng_state(
    rng: np.random.Generator,
    loaded_state_array: np.ndarray[np.uint8],
    loaded_inc_array: np.ndarray[np.uint8],
) -> None:
    r"""Set the state of a random number generator using the 32 bytes
    increment and state arrays stored in ``loaded_inc_array`` and
    ``loaded_state_array``, respectively.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator object.
    loaded_state_array : np.ndarray of numpy.uint8, of size 32.
        State variable to restore the state of the generator.
    loaded_inc_array : np.ndarray of numpy.uint8, of size 32.
        Increment variable to restore the state of the generator.

    Note
    ----
    - Requires ``numpy>=2.0``.
    - Input generator updated in-place.
    """
    loaded_state = array_to_int(loaded_state_array)
    loaded_inc = array_to_int(loaded_inc_array)
    current_state = rng.bit_generator.__getstate__()[0]
    current_state["state"]["state"] = loaded_state
    current_state["state"]["inc"] = loaded_inc
    rng.bit_generator.__setstate__(current_state)
    return


def save_rng_np(rng: np.random.Generator, h5file: h5py.File) -> None:
    r"""Save current state of a numpy random number generator in a .h5 file.

    Parameters
    ----------
    rng : np.random.Generator
        Numpy random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator.
    """
    state_array, inc_array = extract_rng_state(rng)
    dset_state = h5file.create_dataset(
        "rng_state_array", state_array.shape, dtype=state_array.dtype
    )
    dset_state[:] = state_array
    dset_inc = h5file.create_dataset(
        "rng_inc_array", inc_array.shape, dtype=inc_array.dtype
    )
    dset_inc[:] = inc_array
    return


def load_rng_np(rng: np.random.Generator, h5file: h5py.File) -> None:
    r"""Load the state of a numpy random number generator from a .h5 file.

    Parameters
    ----------
    rng : np.random.Generator
        Numpy random number generator.
    h5file : h5py.File
        Handle to a `.h5` file from which the state of the generator will be loeaded.

    Note
    ----
    Requires ``numpy>=2.0``.
    """
    restore_rng_state(rng, h5file["rng_state_array"][:], h5file["rng_inc_array"][:])
    return


def save_rng_offset_torch(rng: torch._C.Generator, seed: int, h5file: h5py.File):
    r"""Save current state of a pytorch random number generator in a .h5 file
    using the offset from the initial seed state.

    Parameters
    ----------
    rng : torch._C.Generator
        Pytorch random number generator on the GPU.
    seed : int
        Seed used to initialize the generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator.

    Note
    ----
    Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
    """
    # ! 1. built-in int type from Python can be very large, and cannot be saved
    # as is to .h5. They need to be converted to hex format (later to an array of ints) to be saved in an .h5 file
    # ! 2. the offset is relative to the initial seed, and thus needs to be saved
    seed_array = int_to_array(seed)
    dset_seed = h5file.create_dataset("seed", seed_array.shape, dtype=seed_array.dtype)
    dset_seed[:] = seed_array

    offset_array = int_to_array(rng.get_offset())
    dset_offset = h5file.create_dataset(
        "offset", offset_array.shape, dtype=offset_array.dtype
    )
    dset_offset[:] = offset_array
    return


def load_rng_offset_torch(rng: torch._C.Generator, h5file: h5py.File):
    r"""Load the state of a pytorch random number generator from a .h5 file using the offset from an initial seed state.

    Parameters
    ----------
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator.

    Note
    ----
    Requires ``pythorch>=2.5``. Only supported for generators on the GPU.
    """
    # ! an offset is relative to some initial seed, which also needs to be
    # ! loaded and set
    seed = array_to_int(h5file["seed"][:])
    rng.manual_seed(seed)
    offset = array_to_int(h5file["offset"][:])
    rng.set_offset(offset)
    return


def save_rng_torch(rng: torch._C.Generator, h5file: h5py.File) -> None:
    r"""Save current state of a torch random number generator in a .h5 file.

    Parameters
    ----------
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator.
    """
    current_state = rng.get_state()
    dset = h5file.create_dataset(
        "torch_rng_state",
        current_state.shape,
        dtype=str(current_state.dtype).split(".")[-1],
    )  # (5056,), torch.uint8, int, work as well
    dset[:] = current_state
    return


def load_rng_torch(rng: torch._C.Generator, h5file: h5py.File) -> None:
    r"""Load the state of a torch random number generator from a .h5 file.

    Parameters
    ----------
    rng : torch._C.Generator
        Pytorch random number generator.
    h5file : h5py.File
        Handle to a `.h5` file to save the state of the generator.
    """
    # ! the loaded state should a priori be moved to the GPU device
    # ! rng.set_state yields currently yields an error (state should be a torch.ByteTensor). Loaded state is kept on the cpu for now.
    loaded_state = torch.tensor(
        h5file["torch_rng_state"][:], dtype=torch.uint8, device="cpu"
    )
    rng.set_state(loaded_state)
    return

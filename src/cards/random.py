"""Random number generation utilities.

For pure serialization purposes, the type of Random Number Generators (RNGs) is
determined entirely by the execution context device.
- CPU: RNGs rely on the default PCG64 bit generator, using :mod:`numpy`.
- GPU: RNGs rely on a Philox-based, counter-style generator, using :mod:`torch`.

Note
----
The functions ``serialize_rng``/``restore_rng`` are written against exactly these two
concrete cases, not generic numpy/torch generators.
"""

import numpy as np
import torch

from cards.core.execution_context import ExecutionContext

# byte order used when packing/unpacking integer RNG state is pinned explicitly (rather
# than using sys.byteorder) for portability across machines with different native endianness
_BYTEORDER = "little"


def _get_seed_sequence(seed: int, rank: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed, spawn_key=(rank,))


def _gpu_rng(local_seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(local_seed)


def _cpu_rng(seed_like: int | np.random.SeedSequence) -> np.random.Generator:
    return np.random.default_rng(seed_like)


def _derive_int_for_pytorch(seed: int, rank: int) -> int:
    # HACK: no solution as mathematically safe as numpy's `rng.spawn()` has been found
    # the global seed is encoded by the first 32 bits
    # the rank is encoded on the last 32 bits
    # the local seed is given by the resulting 64 bits integer
    seed_32 = seed & 0xFFFFFFFF
    rank_32 = rank & 0xFFFFFFFF

    return (seed_32 << 32) | rank_32


def _int_to_bytes(n: int, length: int = 16) -> np.ndarray:
    return np.frombuffer(n.to_bytes(length, _BYTEORDER), dtype=np.uint8)


def _bytes_to_int(arr: np.ndarray) -> int:
    return int.from_bytes(arr.tobytes(), _BYTEORDER)


def create_rng(
    seed: int,
    ctx: ExecutionContext,
) -> np.random.Generator | torch.Generator:
    """Create a rank-aware random number generator based on the execution context.

    Parameters
    ----------
    seed : int
        The base global seed used to initialize the random number generation.
    ctx : ExecutionContext
        The execution context defining the device type (CPU or GPU) and the MPI rank.

    Returns
    -------
    numpy.random.Generator | torch.Generator
        A PyTorch CUDA generator if the context specifies a GPU device, otherwise a
        NumPy default generator. The generator is initialized in a rank-aware manner to
        guarantee (CPU case), or ensure with high probability (GPU case), independent
        random streams across MPI processes.
    """
    rank = ctx.rank
    ss = _get_seed_sequence(seed, rank)

    if ctx.is_gpu:
        torch_seed = _derive_int_for_pytorch(seed, rank)
        return _gpu_rng(torch_seed)

    return _cpu_rng(ss)


def serialize_rng(rng: np.random.Generator | torch.Generator) -> dict[str, np.ndarray]:
    """Serialize a random number generator's internal state into byte arrays.

    Converts internal state integers (which can exceed standard 64-bit integer bounds)
    into 32-byte unsigned integer arrays. This ensures safe disk I/O operations,
    avoiding overflow issues when saving to HDF5.

    Parameters
    ----------
    rng : np.random.Generator | torch.Generator
        The local random number generator instance to serialize.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the serialized internal state.
        For NumPy generators, it contains ``'numpy_state'``, ``'numpy_inc'``,
        ``'numpy_has_uint32'`` and ``'numpy_uinteger'``.
        For PyTorch generators, it contains ``'torch_seed'`` and ``'torch_offset'``.
        The values are 1D ``np.uint8`` arrays in all cases.

    Raises
    ------
    TypeError
        If the provided ``rng`` object is not a supported generator type.
    """
    if isinstance(rng, np.random.Generator):
        full_state = rng.bit_generator.state
        state_dict = full_state["state"]
        return {
            "numpy_state": _int_to_bytes(state_dict["state"], 16),
            "numpy_inc": _int_to_bytes(state_dict["inc"], 16),
            # PCG64 packs two uint32 draws per 64-bit core output. These two
            # fields cache the spare 32-bit half between draws; dropping them
            # makes the restored stream diverge from the original whenever
            # serialization happens mid-cache (has_uint32 == 1).
            "numpy_has_uint32": _int_to_bytes(full_state["has_uint32"], 1),
            "numpy_uinteger": _int_to_bytes(full_state["uinteger"], 4),
        }

    elif isinstance(rng, torch.Generator):
        # Philox is counter-based: (seed, offset) is the complete state, so
        # no extra caching concerns like the numpy case above.
        return {
            "torch_seed": _int_to_bytes(rng.initial_seed(), 8),
            "torch_offset": _int_to_bytes(rng.get_offset(), 8),
        }

    raise TypeError(f"Unsupported generator type: {type(rng)}")


def restore_rng(schema: dict[str, np.ndarray]) -> np.random.Generator | torch.Generator:
    """Restore a random number generator from a serialized state schema.

    Dynamically reconstructs either a PyTorch CUDA generator or a NumPy CPU generator
    depending on the keys present in the provided schema.

    Parameters
    ----------
    schema : dict[str, np.ndarray]
        A dictionary containing the serialized state arrays, typically generated
        by :func:`serialize_rng`.

    Returns
    -------
    np.random.Generator | torch.Generator
        The fully restored random number generator, advanced to the exact internal
        state captured in the schema.

    Raises
    ------
    ValueError
        If the schema keys do not match any known generator recipes.
    """
    if "torch_seed" in schema and "torch_offset" in schema:
        loaded_seed = _bytes_to_int(schema["torch_seed"])
        loaded_offset = _bytes_to_int(schema["torch_offset"])

        rng = _gpu_rng(loaded_seed)
        rng.set_offset(loaded_offset)

        return rng

    elif "numpy_state" in schema and "numpy_inc" in schema:
        loaded_state = _bytes_to_int(schema["numpy_state"])
        loaded_inc = _bytes_to_int(schema["numpy_inc"])
        loaded_has_uint32 = _bytes_to_int(schema["numpy_has_uint32"])
        loaded_uinteger = _bytes_to_int(schema["numpy_uinteger"])

        rng = np.random.default_rng()
        new_state = {
            "bit_generator": "PCG64",
            "state": {
                "state": loaded_state,
                "inc": loaded_inc,
            },
            "has_uint32": loaded_has_uint32,
            "uinteger": loaded_uinteger,
        }
        rng.bit_generator.state = new_state

        return rng

    raise ValueError(f"Unrecognized RNG schema keys: {list(schema.keys())}")

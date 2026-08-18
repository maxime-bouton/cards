r"""MCMC execution backbones and sampling orchestration.

This package provides the core sampling loops to run composable MCMC algorithms, manage
GPU/CPU synchronization, and handle batched checkpoint I/O.

All samplers in this package adhere to the same architecture: chains are evaluated in
chunks of :attr:`~cards.samplers.base_sampler.BaseSampler.ckpt_size` samples, called
checkpoints. At the end of each checkpoint, batched estimates are computed and saved to
disk, along with the current chain state.

Classes
-------
:class:`~cards.samplers.base_sampler.BaseSampler`
    Abstract class defining the MCMC iteration loop, timing, and I/O hooks.
:class:`~cards.samplers.serial_cpu_sampler.SerialCpuSampler`
    Serial CPU implementation utilizing :mod:`numpy` as the backend.
:class:`~cards.samplers.serial_gpu_sampler.SerialGpuSampler`
    Serial GPU implementation utilizing :mod:`cupy` and :mod:`torch` as the backend.
:class:`~cards.samplers.distributed_cpu_sampler.DistributedCpuSampler`
    MPI-based distributed CPU implementation utilizing :mod:`numpy` as the backend.
:class:`~cards.samplers.distributed_gpu_sampler.DistributedGpuSampler`
    MPI-based distributed GPU implementation utilizing :mod:`cupy` and :mod:`torch` as the backend.

Examples
--------
>>> #TODO: add example usage of the samplers here
"""

from .distributed_cpu_sampler import DistributedCpuSampler
from .distributed_gpu_sampler import DistributedGpuSampler
from .sampler import SamplerParameters
from .serial_cpu_sampler import SerialCpuSampler
from .serial_gpu_sampler import SerialGpuSampler

__all__ = [
    "DistributedCpuSampler",
    "DistributedGpuSampler",
    "SamplerParameters",
    "SerialCpuSampler",
    "SerialGpuSampler",
]

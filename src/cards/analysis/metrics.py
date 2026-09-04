r"""Useful metrics to assess reconstruction quality."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import numpy as np
from skimage.metrics import structural_similarity

import cards.backend as xp
from cards.core.execution_context import ExecutionContext


def snr(x: xp.ndarray, y: xp.ndarray, ctx: ExecutionContext | None = None) -> float:
    r"""Compute the reconstruction Signal-to-Noise Ratio (SNR) with respect to a reference array ``x``.

    Parameters
    ----------
    x : xp.ndarray
        Reference array. If distributed, this is the local chunk.
    y : xp.ndarray
        Estimated array. If distributed, this is the local chunk.
    ctx : ExecutionContext, optional
        Execution context object containing MPI properties.

    Returns
    -------
    float
        Globally exact reconstruction SNR.

    Raises
    ------
    ValueError
        Input arrays must have the same shape.
    """
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")

    local_signal_power = float(xp.sum(x**2))
    local_noise_power = float(xp.sum((x - y) ** 2))

    if ctx is not None and ctx.is_mpi:
        from mpi4py import MPI

        global_signal_power = ctx.comm.allreduce(local_signal_power, op=MPI.SUM)
        global_noise_power = ctx.comm.allreduce(local_noise_power, op=MPI.SUM)
    else:
        global_signal_power = local_signal_power
        global_noise_power = local_noise_power

    if global_noise_power == 0:
        return float("inf")

    return float(10 * np.log10(global_signal_power / global_noise_power))


def psnr(x: xp.ndarray, y: xp.ndarray, ctx: ExecutionContext | None = None) -> float:
    r"""Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    x : xp.ndarray
        Reference array. If distributed, this is the local chunk.
    y : xp.ndarray
        Estimated array. If distributed, this is the local chunk.
    ctx : ExecutionContext, optional
        Execution context object containing MPI properties.

    Returns
    -------
    float
        Globally exact Peak Signal-to-Noise Ratio.
    """
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    local_sse = float(xp.sum((x - y) ** 2))
    local_size = int(x.size)

    if ctx is not None and ctx.is_mpi:
        from mpi4py import MPI

        global_sse = ctx.comm.allreduce(local_sse, op=MPI.SUM)
        global_size = ctx.comm.allreduce(local_size, op=MPI.SUM)
    else:
        global_sse = local_sse
        global_size = local_size

    if global_size == 0:
        raise ValueError("Global signal size is 0.")

    global_mse = global_sse / global_size

    if global_mse == 0:
        return float("inf")

    data_range = 1.0
    return float(10 * np.log10((data_range**2) / global_mse))


def ssim(x: xp.ndarray, y: xp.ndarray, ctx: ExecutionContext | None = None) -> float:
    r"""Compute the naive distributed Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    x : xp.ndarray
        Reference image. If distributed, this is the local chunk.
    y : xp.ndarray
        Estimated image. If distributed, this is the local chunk.
    ctx : ExecutionContext, optional
        Execution context object containing MPI properties.

    Returns
    -------
    float
        Approximated global SSIM (neglects MPI boundary overlap).
    """
    if x.shape != y.shape:
        raise ValueError("Input images must have the same shape.")

    local_max = float(x.max())
    local_min = float(x.min())
    local_size = int(x.size)

    if ctx is not None and ctx.is_mpi:
        from mpi4py import MPI

        global_max = ctx.comm.allreduce(local_max, op=MPI.MAX)
        global_min = ctx.comm.allreduce(local_min, op=MPI.MIN)
        global_size = ctx.comm.allreduce(local_size, op=MPI.SUM)
    else:
        global_max = local_max
        global_min = local_min
        global_size = local_size

    data_range = global_max - global_min
    if data_range == 0:
        data_range = 1.0

    x_np = x.get() if (ctx is not None) and ctx.is_gpu else np.asarray(x)
    y_np = y.get() if (ctx is not None) and ctx.is_gpu else np.asarray(y)

    local_ssim = structural_similarity(
        x_np,
        y_np,
        data_range=data_range,
        channel_axis=-3 if len(x_np.shape) > 2 else None,
    )

    local_ssim_sum = local_ssim * local_size

    if ctx is not None and ctx.is_mpi:
        global_ssim_sum = ctx.comm.allreduce(local_ssim_sum, op=MPI.SUM)
        return float(global_ssim_sum / global_size)

    return float(local_ssim)

import csv
import os
from datetime import datetime
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import torch
from mpi4py import MPI

from mcmc.backend import bm
from mcmc.denoisers.mpi_ddfb import MpiDDFB
from mcmc.denoisers.mpi_dncnn import MpiDnCNN
from mcmc.denoisers.mpi_drunet import MpiDRUNet
from mcmc.logger import build_logger
from mcmc.utils.utils_img import load_img, read_img_shape


def load_denoisers(comm, grid_size, image_size):
    return {
        "DDFB-4": MpiDDFB(comm, grid_size, image_size, n_layers=4, n_features=64),
        "DDFB-20": MpiDDFB(comm, grid_size, image_size, n_layers=20, n_features=64),
        "DnCNN": MpiDnCNN(comm, grid_size, image_size),
        "DRUNet": MpiDRUNet(comm, grid_size, image_size),
    }


def load_tensor_from_h5(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        return cp.asarray(f["x"][:], dtype=cp.float32)


def measure_runtime_distributed(denoiser, x, sigma, warmup=3, repeat=10):
    for _ in range(warmup):
        with torch.no_grad():
            _ = denoiser(x, sigma)
    torch.cuda.synchronize()

    runtimes = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            _ = denoiser(x, sigma)
        end.record()
        torch.cuda.synchronize()
        runtimes.append(start.elapsed_time(end))

    return np.array(runtimes)


if __name__ == "__main__":
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    bm.set_backend("cupy")
    gpu = bm.xp.cuda.Device(rank % bm.xp.cuda.runtime.getDeviceCount())
    gpu.use()

    torch.cuda.set_device(gpu.id)
    torch.set_default_device("cuda")
    torch.backends.cudnn.deterministic = True

    # Config
    H5_DIR = "data"
    SIGMA = 0.1
    PATCH_SIZES = [128, 256, 512, 1024, 2048, 4096]
    GRID_SHAPE = np.array([1, size, 1])  # Example: linear grid; adapt as needed

    # Initialize logger and CSV writer on rank 0
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"produced_data/benchmark_runtime/mpi_{size}/logs_{timestamp}")
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "runtime.csv"
        txt_log_path = log_dir / "runtime.txt"
        logger = build_logger(0, txt_log_path)
        csvfile = open(csv_path, mode="w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["patch_size", "denoiser_name", "runtime_ms"])

    for N in PATCH_SIZES:
        h5_path = os.path.join(H5_DIR, f"{N}.h5")
        if not os.path.isfile(h5_path):
            if rank == 0:
                logger.warning(f"Missing file: {h5_path}")
            continue

        img_size = np.array(read_img_shape(h5_path, key="x"))
        x = load_img(h5_path, key="x")

        if rank == 0:
            logger.info(f"Evaluating on image size: 3x{N}x{N}")

        # List of denoisers to evaluate
        denoiser_specs = [
            (
                "DDFB-4",
                lambda: MpiDDFB(comm, GRID_SHAPE, img_size, n_layers=4, n_features=64),
            ),
            (
                "DDFB-20",
                lambda: MpiDDFB(comm, GRID_SHAPE, img_size, n_layers=20, n_features=64),
            ),
            ("DnCNN", lambda: MpiDnCNN(comm, GRID_SHAPE, img_size)),
            ("DRUNet", lambda: MpiDRUNet(comm, GRID_SHAPE, img_size)),
        ]

        for name, net_fn in denoiser_specs:
            if name == "DRUNet" and N == 4096:
                continue
            try:
                # Load denoiser on the fly
                net = net_fn()
                s = net.global_to_tile_slice
                tile_x = x[s]

                # Measure runtime
                runtimes = measure_runtime_distributed(net, tile_x, sigma=SIGMA)
                mean_time = np.mean(runtimes)
                std_time = np.std(runtimes)

            except (RuntimeError, cp.cuda.memory.OutOfMemoryError) as e:
                if rank == 0:
                    logger.error(f"OOM encountered for {name} on patch {N}: {e}")
                mean_time = float("nan")
                std_time = float("nan")

            finally:
                # Delete denoiser and clear GPU memory to prevent cascading OOMs
                if "net" in locals():
                    del net
                torch.cuda.empty_cache()
                if hasattr(bm.xp.cuda, "Device"):
                    bm.xp.get_default_memory_pool().free_all_blocks()

            # Gather stats across MPI ranks
            all_means = comm.gather(mean_time, root=0)
            all_stds = comm.gather(std_time, root=0)

            if rank == 0:
                avg_time = np.nanmax(all_means)
                std_time = np.nanmax(all_stds)
                logger.info(
                    f"\t{name:<10} | runtime: {avg_time:.2f} ms (± {std_time:.2f} ms)"
                )
                writer.writerow([N, name, avg_time, std_time])

    if rank == 0:
        csvfile.close()

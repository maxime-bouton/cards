import csv
import os
from datetime import datetime
from pathlib import Path

import cupy as cp  # to handle potential CuPy OOM
import h5py
import numpy as np
import torch

from mcmc.denoisers.denoiser_loader import (
    load_pretrained_ddfb,
    load_pretrained_dncnn,
    load_pretrained_drunet,
)
from mcmc.logger import build_logger

torch.cuda.set_device(0)
torch.set_default_device("cuda")


def load_tensor_from_h5(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        data = torch.tensor(f["x"][:]).unsqueeze(0)
    return data.to(torch.device("cuda"))


def measure_runtime(model, input_tensor, sigma=None, warmup=3, repeat=10):
    # Warm-up runs
    for _ in range(warmup):
        with torch.no_grad():
            if sigma is not None:
                _ = model(input_tensor, sigma)
            else:
                _ = model(input_tensor)
    torch.cuda.synchronize()

    runtimes = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            if sigma is not None:
                _ = model(input_tensor, sigma)
            else:
                _ = model(input_tensor)
        end.record()
        torch.cuda.synchronize()

        runtimes.append(start.elapsed_time(end))

    return np.array(runtimes)


if __name__ == "__main__":
    SIZES = [128, 256, 512, 1024, 2048, 4096]
    H5_DIR = "data"
    SIGMA = 0.1

    # Define denoiser constructors (lazy loading)
    denoiser_specs = [
        ("DDFB-4", lambda: load_pretrained_ddfb(n_layers=4, n_features=64), True),
        ("DDFB-20", lambda: load_pretrained_ddfb(n_layers=20, n_features=64), True),
        ("DnCNN", lambda: load_pretrained_dncnn(), False),
        ("DRUNet", lambda: load_pretrained_drunet(), True),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"produced_data/benchmark_runtime/serial/logs_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "runtime.csv"
    txt_log_path = log_dir / "runtime.txt"
    logger = build_logger(0, txt_log_path)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_size", "denoiser_name", "runtime_ms", "std_ms"])

        for N in SIZES:
            h5_path = os.path.join(H5_DIR, f"{N}.h5")
            if not os.path.isfile(h5_path):
                logger.warning(f"Missing file: {h5_path}")
                continue

            x = load_tensor_from_h5(h5_path)
            logger.info(f"Evaluating on image size: 3x{N}x{N}")

            for name, net_fn, requires_sigma in denoiser_specs:
                try:
                    net = net_fn()
                    net.eval()
                    runtime = measure_runtime(
                        net, x, sigma=SIGMA if requires_sigma else None
                    )
                    mean = runtime.mean()
                    std = runtime.std()

                except (RuntimeError, cp.cuda.memory.OutOfMemoryError) as e:
                    logger.error(f"OOM encountered for {name} on patch {N}: {e}")
                    mean = float("nan")
                    std = float("nan")

                finally:
                    if "net" in locals():
                        del net
                    torch.cuda.empty_cache()
                    if hasattr(cp.cuda, "Device"):
                        cp.get_default_memory_pool().free_all_blocks()

                logger.info(f"\t{name:<10} | runtime: {mean:.2f} ms (± {std:.2f} ms)")
                writer.writerow([N, name, mean, std])

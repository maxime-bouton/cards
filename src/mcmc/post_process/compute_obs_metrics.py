from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mcmc.backend import bm, xp
from mcmc.operators.dft_convolution import DftConvolution
from mcmc.operators.masking import Masking
from mcmc.utils.metrics import snr, ssim
from mcmc.utils.utils_observations import fit_kernel_shape, fit_mask_shape


def save_tensor_as_image(tensor: np.ndarray, out_path: Path, range: int = 1):
    """Save a (C, H, W) numpy array as a PNG image."""
    if tensor.ndim != 3:
        raise ValueError("Expected tensor of shape (C, H, W)")

    C, H, W = tensor.shape
    if C not in [1, 3]:
        raise ValueError(f"Only 1 or 3 channel images supported, got C={C}")

    tensor = np.clip(tensor / range, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)

    if C == 1:
        img = Image.fromarray(tensor[0], mode="L")
    else:
        img = Image.fromarray(np.transpose(tensor, (1, 2, 0)), mode="RGB")

    img.save(out_path, format="JPEG")


def process_h5_file(file_path: Path):
    out_dir = file_path.parent
    app = out_dir.parent.name

    noise_type, operator = app.split("-")

    with h5py.File(file_path, "r") as f:
        x = xp.asarray(f["x"])
        y = xp.asarray(f["y"])
        image_size = np.array(x.shape)

        if operator == "deconvolution":
            kernel = xp.asarray(f["kernel"])
            kernel = fit_kernel_shape(kernel, image_size)
            kernel_size = np.asarray(kernel.shape, dtype=int)

            data_size = image_size.copy()
            data_size[-len(kernel_size) :] += kernel_size - 1
            H = DftConvolution(image_size, kernel, data_size)

            y_crop = y[
                ...,
                kernel_size[-2] // 2 : -(kernel_size[-2] // 2),
                kernel_size[-1] // 2 : -(kernel_size[-1] // 2),
            ]
        else:
            interpolation = xp.asarray(f["interpolation"])
            mask = xp.asarray(f["mask"])
            mask = fit_mask_shape(xp.asarray(mask), image_size)

            H = Masking(mask)

            y_crop = y

        if noise_type == "poisson":
            y /= f["dynamic_range"][()]

    Hx = xp.maximum(H.forward(x), 0)

    snr_wrt_Hx = snr(Hx, y)
    ssim_wrt_Hx = ssim(Hx, y)

    snr_wrt_x = snr(x, y_crop)
    ssim_wrt_x = ssim(x, y_crop)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write("Comparison `y` and `x`':\n")
        f.write(f"SNR: {snr_wrt_x:.2f} dB\n")
        f.write(f"SSIM: {ssim_wrt_x:.4f}\n")

        f.write("\nComparison `y` and `Hx`':\n")
        f.write(f"SNR: {snr_wrt_Hx:.2f} dB\n")
        f.write(f"SSIM: {ssim_wrt_Hx:.4f}\n")

        if operator == "inpainting":
            snr_interpolation = snr(x, interpolation)
            ssim_interpolation = ssim(x, interpolation)

            f.write("\nComparison `interpolation` and `x`':\n")
            f.write(f"SNR: {snr_interpolation:.2f} dB\n")
            f.write(f"SSIM: {ssim_interpolation:.4f}\n")


def walk_and_process(root_dir: Path):
    for h5_file in root_dir.rglob("data.h5"):
        process_h5_file(h5_file)


if __name__ == "__main__":
    import argparse

    bm.set_backend("numpy")

    parser = argparse.ArgumentParser(
        description="Convert 'x' and 'y' from data.h5 files to PNG images."
    )
    parser.add_argument(
        "root_dir", type=Path, help="Root directory to search for data.h5 files"
    )
    args = parser.parse_args()

    walk_and_process(args.root_dir)

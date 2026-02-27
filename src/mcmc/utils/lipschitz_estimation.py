import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as transforms
from deepinv.loss.regularisers import JacobianSpectralNorm
from PIL import Image

from mcmc.denoisers.denoiser_loader import (
    load_pretrained_ddfb,
    load_pretrained_dncnn,
    load_pretrained_drunet,
)
from mcmc.logger import build_logger


def load_denoisers():
    """Load all pretrained denoising models.

    Returns
    -------
    dict
        Dictionary containing model information with keys for each denoiser.
    """
    ddfb4 = load_pretrained_ddfb(n_layers=4, n_features=64)
    ddfb19 = load_pretrained_ddfb(n_layers=19, n_features=64)
    ddfb20 = load_pretrained_ddfb(n_layers=20, n_features=64)
    dncnn = load_pretrained_dncnn()
    drunet = load_pretrained_drunet()

    return {
        "DDFB-4": {"model": ddfb4, "requires_sigma": True},
        "DDFB-19": {"model": ddfb19, "requires_sigma": True},
        "DDFB-20": {"model": ddfb20, "requires_sigma": True},
        "DnCNN": {"model": dncnn, "requires_sigma": False},
        "DRUNet": {"model": drunet, "requires_sigma": True},
    }


def load_images_from_directory(image_dir: str) -> List[str]:
    """
    Load all JPEG image paths from a directory.

    Parameters
    ----------
    image_dir : str
        Path to directory containing JPEG images

    Returns
    -------
    List[str]
        List of paths to JPEG images
    """
    valid_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    image_paths = []

    for filename in os.listdir(image_dir):
        if any(filename.endswith(ext) for ext in valid_extensions):
            image_paths.append(os.path.join(image_dir, filename))

    return image_paths


def extract_random_patches(
    image_paths: List[str], num_patches: int, patch_size: int
) -> torch.Tensor:
    """
    Extract random patches from images.

    Parameters
    ----------
    image_paths : List[str]
        List of paths to image files
    num_patches : int
        Number of patches to extract (P)
    patch_size : int
        Size of square patches (N x N)

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_patches, 3, patch_size, patch_size)
    """
    patches = []
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    for _ in range(num_patches):
        img_path = random.choice(image_paths)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_tensor = transform(img)

                _, h, w = img_tensor.shape

                if h >= patch_size and w >= patch_size:
                    top = random.randint(0, h - patch_size)
                    left = random.randint(0, w - patch_size)
                    patch = img_tensor[
                        :, top : top + patch_size, left : left + patch_size
                    ]
                    patches.append(patch)
                else:
                    resize_transform = transforms.Resize((patch_size, patch_size))
                    patch = resize_transform(img_tensor)
                    patches.append(patch)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            fallback_patch = torch.randn(3, patch_size, patch_size)
            patches.append(fallback_patch)

    return torch.stack(patches)


if __name__ == "__main__":
    IMAGE_DIR = "/data3/mbouton/imagenet/test/"
    # IMAGE_DIR = "/lustre/fsmisc/dataset/ILSVRC/ILSVRC/Data/CLS-LOC/test/"
    sigmas = [0.01, 0.03, 0.05, 0.07, 0.1]
    P = 100
    N_initial = 8

    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    denoisers = load_denoisers()
    reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"produced_data/estimation_lip/logs_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "res.csv"
    txt_log_path = log_dir / "res.txt"

    logger = build_logger(0, txt_log_path)

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sigma", "patch_size", "denoiser_name", "spectral_norm"])

        logger.info(f"Number of patches: 2P (ground truth + noisy), with P={P}")

        for sigma in sigmas:
            logger.info(f"Testing with noise level: sigma={sigma:.2f}")
            for i in range(4):
                N = N_initial * (2**i)
                logger.info(f"\t>> Patch size: 3x{N}x{N}")
                image_paths = load_images_from_directory(IMAGE_DIR)

                if len(image_paths) == 0:
                    logger.warning("No images found! Using random noise as fallback.")
                    x = torch.randn((P, 3, N, N), requires_grad=True)
                else:
                    x = extract_random_patches(image_paths, P, N)
                    y = x + torch.randn_like(x) * sigma
                    x = y.to(torch.device("cuda")).requires_grad_(True)
                    # x = (
                    #     torch.concat((y, y), dim=0)
                    #     .to(torch.device("cuda"))
                    #     .requires_grad_(True)
                    # )

                for name, info in denoisers.items():
                    net = info["model"]
                    if name.startswith("DDFB"):
                        net.train(True)

                    if info["requires_sigma"]:
                        out = x - net(x, sigma)
                    else:
                        out = x - net(x)

                    regval = reg_l2(out, x)
                    logger.info(
                        f"\t\t{name:<10} | Jacobian spectral norm: {regval:.4f}"
                    )
                    writer.writerow([sigma, N, name, regval.item()])

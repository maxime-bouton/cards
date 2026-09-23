from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cards.backend as xp


def format_img(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    return img


def save_point_estimate(
    img: np.ndarray,
    path: Path,
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    img = format_img(img)
    img = np.clip(img, vmin, vmax)
    plt.imsave(path, img, vmin=vmin, vmax=vmax)


def save_map_with_colorbar(
    img: np.ndarray,
    path: Path,
    cmap: str = "inferno",
    dpi: int = 100,
) -> None:
    H, W = img.shape[-2:]

    gap_px = 15  # space between image and colorbar
    cb_px = 25  # width of the colorbar itself
    label_px = 60  # space reserved on the right for text labels
    W_new = W + gap_px + cb_px + label_px

    fig = plt.figure(figsize=(W_new / dpi, H / dpi), dpi=dpi)

    ax_img = fig.add_axes((0, 0, W / W_new, 1.0))
    im = ax_img.imshow(img, cmap=cmap)
    ax_img.axis("off")

    cb_left = (W + gap_px) / W_new
    cb_width = cb_px / W_new
    ax_cb = fig.add_axes((cb_left, 0.05, cb_width, 0.90))
    fig.colorbar(im, cax=ax_cb)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_uncertainty_maps(
    data: np.ndarray,
    path_prefix: Path,
    cmap: str = "inferno",
) -> None:
    channels = [data] if data.ndim == 2 else [data[c] for c in range(data.shape[0])]
    for c, channel_img in enumerate(channels):
        save_map_with_colorbar(
            channel_img,
            path_prefix.with_name(f"{path_prefix.name}_{c}.jpg"),
            cmap=cmap,
        )


def plot_potential(potential: xp.ndarray, path: Path) -> None:
    fig, ax = plt.subplots()
    p = potential.get() if xp.get_backend() == "cupy" else potential
    ax.plot(p)
    ax.set_title("Potential over sampling iteration steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Potential")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)

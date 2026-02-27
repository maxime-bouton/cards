import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker

COLORS = {
    "white": "#ffffff",
    "black": "#000000",
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "violet": "#8a2be2",
    "sky": "#008cff",
}


def save_image_with_zoom(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    size: int,
    path: str,
    zoom_fraction: float = 0.35,
    inset_border_thickness: int = 1,
    n_vblocks: int = 0,
    dash_pattern: tuple[int, int] = (5, 5),
    inset_color: str = "white",
    dash_color: str = "cyan",
):
    zoomed_img = img[y_start : y_start + size, x_start : x_start + size]

    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    rect = plt.Rectangle(
        (x_start, y_start),
        size,
        size,
        linewidth=(img.shape[0] / 100) * inset_border_thickness / 3,
        edgecolor=COLORS[inset_color],
        facecolor="none",
    )
    ax.add_patch(rect)

    if n_vblocks > 1:
        height = img.shape[0]
        for i in range(1, n_vblocks):
            y = i * height / n_vblocks
            ax.axhline(
                y=y,
                color=COLORS[dash_color],
                linestyle="--",
                linewidth=(img.shape[0] / 100) * inset_border_thickness,
                dashes=dash_pattern,
            )

    inset_size = zoom_fraction
    inset_ax = fig.add_axes([1 - inset_size, 0, inset_size, inset_size])
    inset_ax.imshow(zoomed_img, cmap="gray")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    inset_ax.spines["left"].set_linewidth((img.shape[1] / 100) * inset_border_thickness)
    inset_ax.spines["left"].set_edgecolor(COLORS[inset_color])
    inset_ax.spines["top"].set_linewidth((img.shape[0] / 100) * inset_border_thickness)
    inset_ax.spines["top"].set_edgecolor(COLORS[inset_color])
    inset_ax.spines["right"].set_visible(False)
    inset_ax.spines["bottom"].set_visible(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=100, format="jpg")
    plt.close(fig)


def save_image_with_color_bar(
    img, vmin, vmax, path, log=False, font_factor=0.03, font_color="red"
):
    img_h, img_w = img.shape
    fig_w = img_w / 100
    fig_h = img_h / 100
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    main_ax = plt.axes([0, 0, 1, 1])
    norm = "log" if log else None
    im = main_ax.imshow(img, cmap="gray_r", norm=norm, vmin=vmin, vmax=vmax)
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    base_font_size = min(img_w, img_h) * font_factor
    cbar_ax = plt.axes([0.85, 0.05, 0.05, 0.9])
    cbar = plt.colorbar(im, cax=cbar_ax, format="%.2e")
    cbar_ax.tick_params(labelsize=base_font_size, colors=COLORS[font_color])
    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    offset_text = cbar_ax.yaxis.get_offset_text()
    offset_text.set_color(COLORS[font_color])
    offset_text.set_fontsize(base_font_size)
    plt.savefig(path, bbox_inches=None, dpi=100, pad_inches=0)
    plt.close(fig)

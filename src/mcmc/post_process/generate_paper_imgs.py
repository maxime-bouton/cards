from pathlib import Path

import numpy as np

from mcmc.post_process.utils import save_image_with_color_bar, save_image_with_zoom
from mcmc.utils.utils_img import load_img


def walk_and_process(root_dir: Path):
    zoom_configs = {
        "2048": (400, 975, 100),
        "2896": (1800, 1376, 144),
        "4096": (1400, 950, 200),
    }

    for h5_file in root_dir.rglob("data.h5"):
        data_dir = h5_file.parent
        img_name = data_dir.name.split("-")[0]
        app = data_dir.parents[0].name
        poisson = app.split("-")[0] == "poisson"

        if poisson:
            rg = int("".join([c for c in data_dir.name.split("-")[3] if c.isdigit()]))
        else:
            rg = 1
        y = load_img(h5_file, key="y")
        x = load_img(h5_file, key="x")
        diff = np.asarray(y.shape) - np.asarray(x.shape)

        zoom_conf = zoom_configs[data_dir.name.split("-")[0]]

        print(f"Processing {h5_file}")

        inset_color = "yellow"
        dash_color = "yellow"

        # save_image_with_zoom(
        #     x.transpose(1, 2, 0),
        #     x_start=zoom_conf[0],
        #     y_start=zoom_conf[1],
        #     size=zoom_conf[2],
        #     path=data_dir / f"x_{img_name}.jpg",
        #     n_vblocks=4 if img_name != "2896" else 2,
        #     dash_color=dash_color,
        #     inset_color=inset_color,
        # )

        # save_image_with_zoom(
        #     y.transpose(1, 2, 0) / rg,
        #     x_start=zoom_conf[0] + diff[-1] // 2,
        #     y_start=zoom_conf[1] + diff[-2] // 2,
        #     size=zoom_conf[2],
        #     path=data_dir / f"y_{img_name}_{app}.jpg",
        #     inset_color=inset_color,
        # )

        try:
            interpolation = load_img(h5_file, key="interpolation")
            save_image_with_zoom(
                interpolation.transpose(1, 2, 0),
                x_start=zoom_conf[0],
                y_start=zoom_conf[1],
                size=zoom_conf[2],
                path=data_dir / f"interpolation_{img_name}.jpg",
                inset_color=inset_color,
            )
        except KeyError:
            pass

        for dirpath, dirnames, _ in data_dir.walk():
            for dirname in dirnames:
                if dirname.startswith("burnin"):
                    full_dir_path = Path(dirpath) / dirname
                    match full_dir_path.parent.name:
                        case "mpi-gpu_2":
                            B = 2
                        case "mpi-gpu_4":
                            B = 4
                        case _:
                            B = 1
                    prior = full_dir_path.relative_to(data_dir).parts[0]
                    print(f"Processing {full_dir_path}")
                    # x_mmse = load_img(full_dir_path / "x_mmse.h5", key="x")
                    x_mmse = load_img(full_dir_path / "estim.h5", key="x_mmse")
                    x_var = load_img(full_dir_path / "estim.h5", key="x_var")
                    suffix = f"_{img_name}_{app}_{prior}_{B}"
                    # save_image_with_zoom(
                    #     x_mmse.transpose(1, 2, 0),
                    #     x_start=zoom_conf[0],
                    #     y_start=zoom_conf[1],
                    #     size=zoom_conf[2],
                    #     path=full_dir_path / f"x_mmse{suffix}.jpg",
                    #     inset_color=inset_color,
                    # )
                    vmin = x_var[0].min()
                    vmax = x_var[0].max()
                    # save_image_with_color_bar(
                    #     x_var[0],
                    #     vmin=1e-3,
                    #     vmax=1.5e-2,
                    #     path=full_dir_path / f"x_var{suffix}.jpg",
                    #     font_color="red",
                    # )

                    error = (x_mmse - x).transpose(1, 2, 0)
                    print(np.abs(error).mean())
                    print(np.sum(x_mmse > 1.1) / error.size)

                    save_image_with_color_bar(
                        np.linalg.norm(error, axis=-1),
                        vmin=0,
                        vmax=1,
                        path=full_dir_path / f"error{suffix}.jpg",
                        font_color="red",
                    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MMSE from checkpoint aggregation and related metrics."
    )
    parser.add_argument("--root_dir", type=Path, help="Root directory")
    args = parser.parse_args()

    walk_and_process(args.root_dir)

import h5py
import numpy as np
from scipy import interpolate

from mcmc.backend import xp
from mcmc.operators.masking import Masking
from mcmc.utils.main_helper import run_main
from mcmc.utils.path_builder import gaussian_str, inpainting_str, obs_dir
from mcmc.utils.utils import extract_subset_from_dict
from mcmc.utils.utils_img import load_img, read_img_shape
from mcmc.utils.utils_observations import (
    apply_target_gaussian_noise,
    fit_mask_shape,
    generate_and_save_observations,
)


def gaussian_inpainting_params(params: dict) -> dict:
    if "denoiser_params" in params:
        return extract_subset_from_dict(params, ["reg_coef", "denoiser_params"])
    else:
        return extract_subset_from_dict(params, ["reg_coef", "split_coef"])


def application_params_dir(params: dict) -> str:
    if "denoiser_params" in params:
        return "."
    return f"split{params['split_coef']}".replace(".", "_")


def build_obs_and_model_paths(params: dict) -> tuple[str, str]:
    noise_str = gaussian_str(params)
    application_str = inpainting_str(params)
    obs_f = f"gaussian-inpainting/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f


def interpolate_masked_image_cubic(
    masked_image: xp.ndarray,
    mask: xp.ndarray,
) -> xp.ndarray:
    """
    Interpolate masked values in an image using cubic spline interpolation.
    Transfers data to CPU for interpolation.

    Parameters
    ----------
    masked_image : xp.ndarray
        Image with masked values, shape (C, H, W)
    mask : xp.ndarray
        Boolean mask where True/1 indicates visible pixels, shape (C, H, W)

    Returns
    -------
    xp.ndarray
        Interpolated image with the same shape as the input
    """
    C, H, W = masked_image.shape
    result = masked_image.copy()
    C_mask = mask.shape[-3]
    for c in range(C):
        channel_gpu = masked_image[c]
        mask_gpu = xp.asarray(mask[min(c, C_mask - 1)]).astype(bool)

        if xp.all(~mask_gpu) or xp.all(mask_gpu):
            continue

        if xp.__name__ == "cupy":
            channel_cpu = channel_gpu.get()
            mask_cpu = mask_gpu.get()

        known_coords = np.where(mask_cpu)
        known_values = channel_cpu[known_coords]

        y_grid, x_grid = np.mgrid[0:H, 0:W]

        filled_channel = interpolate.griddata(
            np.column_stack((known_coords[0], known_coords[1])),
            known_values,
            (y_grid, x_grid),
            method="cubic",
            fill_value=np.mean(known_values),
        )

        filled_channel_gpu = xp.asarray(filled_channel)
        result[c][~mask_gpu] = filled_channel_gpu[~mask_gpu]

    return result


def generate_interpolation(obs_path, mask):
    y = load_img(obs_path, key="y")
    interpolated_y = interpolate_masked_image_cubic(y, mask).clip(0, 1)

    with h5py.File(obs_path, "a") as file:
        file["interpolation"] = (
            interpolated_y
            if isinstance(interpolated_y, np.ndarray)
            else interpolated_y.get()
        )


def generate_inpainting_observations(
    original_img_path: str,
    mask_loss: float,
    isnr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    gt_size = np.asarray(read_img_shape(original_img_path))
    rng = np.random.default_rng(data_seed)
    mask = rng.random(gt_size[-2:]) < (1 - mask_loss)
    inpainting_params = {"mask": mask.copy()}

    mask_extended = fit_mask_shape(xp.asarray(mask), gt_size)
    inpainting_operator = Masking(mask_extended)

    generate_and_save_observations(
        original_img_path,
        obs_path,
        inpainting_operator,
        apply_target_gaussian_noise,
        data_seed,
        inpainting_params,
        maximum,
        isnr=isnr,
    )

    generate_interpolation(obs_path, mask_extended)


if __name__ == "__main__":
    run_main(
        gaussian_inpainting_params,
        lambda d: {},
        build_obs_and_model_paths,
        lambda p: generate_inpainting_observations(
            p["original_img_path"],
            p["mask_loss"],
            p["isnr"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_gaussian_inpainting",
    )

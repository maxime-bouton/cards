import numpy as np

from mcmc.operators.dft_convolution import DftConvolution
from mcmc.utils.main_helper import run_main
from mcmc.utils.path_builder import (
    deconvolution_str,
    obs_dir,
    poisson_str,
)
from mcmc.utils.utils import extract_subset_from_dict
from mcmc.utils.utils_img import read_dtype, read_img_shape
from mcmc.utils.utils_observations import (
    apply_poisson_noise,
    fit_kernel_shape,
    generate_and_save_observations,
    generate_gaussian_kernel,
    generate_motion_kernel,
    slice_linear_conv_to_original,
)


def poisson_deconvolution_params(params: dict) -> dict:
    d = extract_subset_from_dict(params, ["reg_coef", "split_coef1", "split_coef2"])
    if "denoiser_params" in params:
        d.update(extract_subset_from_dict(params, ["denoiser_params"]))
    return d


def define_slices_and_range(params: dict) -> dict:
    img_size = read_img_shape(params["original_img_path"])
    # TODO: find a more elegant way to deal with image of different dimensions
    kernel_size = np.asarray([1] * (len(img_size) - 2) + [params["kernel"]["size"]] * 2)
    return {
        "slices": slice_linear_conv_to_original(img_size, kernel_size),
        "obs_rg": params["dynamic_range"],
    }


def generate_poisson_deconvolution_observations(
    original_img_path: str,
    kernel_params: dict,
    dynamic_range: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
):
    gt_size = np.asarray(read_img_shape(original_img_path))
    dtype = read_dtype(original_img_path)

    rng = np.random.default_rng(data_seed)

    if kernel_params["type"] == "motion":
        kernel = generate_motion_kernel(
            kernel_params["size"],
            kernel_params["intensity"],
            dtype,
            rng,
        )
    else:
        kernel = generate_gaussian_kernel(
            kernel_params["size"],
            kernel_params["std"],
            dtype,
        )

    params_saved = {"kernel": kernel}

    reshaped_kernel = fit_kernel_shape(kernel, gt_size)

    obs_dims = gt_size.copy()
    # convolution affects only the last two dimensions (i.e., spatial dimensions)
    obs_dims[-2:] += np.asarray(kernel.shape, dtype=int) - 1

    convolution_handler = DftConvolution(gt_size, reshaped_kernel, obs_dims)

    generate_and_save_observations(
        original_img_path,
        obs_path,
        convolution_handler,
        apply_poisson_noise,
        data_seed,
        params_saved,
        maximum,
        dynamic_range=dynamic_range,
    )


def application_params_dir(params: dict) -> str:
    return f"split1_{params['split_coef1']}-split2_{params['split_coef2']}".replace(
        ".", "_"
    )


def build_obs_and_model_paths(params: dict) -> tuple[str, str]:
    noise_str = poisson_str(params)
    application_str = deconvolution_str(params)
    obs_f = f"poisson-deconvolution/{obs_dir(params, application_str, noise_str)}"
    model_params_f = application_params_dir(params)

    return obs_f, model_params_f


if __name__ == "__main__":
    run_main(
        poisson_deconvolution_params,
        define_slices_and_range,
        build_obs_and_model_paths,
        generate_observations_fn=lambda p: generate_poisson_deconvolution_observations(
            p["original_img_path"],
            p["kernel"],
            p["dynamic_range"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_poisson_deconvolution",
    )

import json

import h5py
import numpy as np

from mcmc.utils.utils import apply_gaussian_noise, generate_observations, load_img_size


def load_from_h5(filename, local_slice=slice(None)):
    """load the mask01, sig2 and data entries from the h5 file"""
    with h5py.File(filename, "r") as data_file:
        mask = data_file["mask01"][local_slice]
        sigma2 = data_file["sigma2"][()]
        observations = data_file["data"][local_slice]
    return mask, sigma2, observations


def add_inpainting_params(config_file_path: str, args: dict) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    pass


def generate_inpainting_observations(
    original_path: str,
    mask_loss: float,
    snr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
) -> None:
    #! backend/set_up must be done before the call
    from mcmc.operators.masking import Masking

    dims = load_img_size(original_path)
    rng = np.random.default_rng(data_seed)
    mask = rng.binomial(1, 1 - mask_loss, dims)

    inpainting_operator = Masking(mask)

    inpainting_params = {}
    inpainting_params["mask"] = mask
    inpainting_params["mask01"] = mask

    generate_observations(
        original_path,
        inpainting_operator,
        snr,
        apply_gaussian_noise,
        data_seed,
        obs_path,
        maximum=maximum,
        problem_parameters=inpainting_params,
    )

    pass

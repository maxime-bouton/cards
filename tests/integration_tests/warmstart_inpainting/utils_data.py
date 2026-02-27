from mcmc.utils.utils import load_img_size, generate_observations, apply_gaussian_noise
from mcmc.operators.masking import Masking as Inpainting

import numpy as np
import json
import h5py
from os.path import join


def add_inpainting_params(args: dict, config_file_path: str) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    return


def generate_inpainting_observations(
    original_path: str,
    mask_loss: float,
    snr: float,
    data_seed: int,
    obs_path: str,
    maximum: float = 1.0,
) -> None:
    dims = load_img_size(original_path)
    rng = np.random.default_rng(data_seed)
    mask = rng.binomial(1, 1 - mask_loss, dims)

    inpainting_operator = Inpainting(mask)

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
        problem_parameters=inpainting_params,
        maximum=maximum,
    )


def check_data(
    num_loaded_batch: int, nb_checkpoint: int, save_path: str, resumed_save_path: str
) -> bool:
    X = []
    resumed_X = []
    potential = []
    resumed_potential = []

    for i in range(num_loaded_batch + 1, nb_checkpoint + 1):
        with h5py.File(join(save_path, "sample" + str(i) + ".h5"), "r") as file:
            X = np.append(X, file["X"][:])
            potential = np.append(potential, file["potential"])
        with h5py.File(join(resumed_save_path, "sample" + str(i) + ".h5"), "r") as file:
            resumed_X = np.append(resumed_X, file["X"][:])
            resumed_potential = np.append(resumed_potential, file["potential"][:])

    check_variable = np.allclose(X, resumed_X)
    check_potential = np.allclose(potential, resumed_potential)

    # plt.figure()
    # plt.imshow(X-resumed_X, cmap="gray")
    # plt.show()

    # plt.plot(potential-resumed_potential)
    # plt.show()

    return check_potential and check_variable

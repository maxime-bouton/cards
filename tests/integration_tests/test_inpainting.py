import json
import logging
import os
from os.path import join

import h5py
import numpy as np
import pytest
from utils import (
    compute_distributed,
    compute_gpu,
    compute_serial,
    resume_distributed_sampler,
    resume_gpu_sampler,
    resume_serial_sampler,
)
from skimage.metrics import structural_similarity as ssim


def analyze_data(
    nb_batches: int,
    burnin: int,
    save_path: str,
    data_path: str,
    output_file_name: str,
) -> None:
    """analyze_data Read the sample produced by a "compute" function and make some metrics out of it.

    Parameters
    ----------
    nb_batches : int
        Number of batches of the sample.
    burnin : int
        Number of bacth to ignore to take into account the burn-in.
    save_path : str
        Path to the directory where the sample has been saved.
    data_path : str
        Path to the file containing the deteriorated signal.
    output_file_name : str
        Full path to the file on which we will write the metrics.
    """

    potential = []
    time = []
    for i in range(1, nb_batches + 1):
        file_name = join(save_path, "sample" + str(i) + ".h5")

        with h5py.File(file_name, "r") as file:
            potential = np.append(potential, file["potential"][:])
            time = np.append(time, file["computation_time"][:])

    atime = np.mean(time)
    total_time = np.sum(time)
    time_std = np.std(time)

    with h5py.File(data_path, "r") as file:
        original = file["x"][:]
        observations = file["data"][:]

    MMSE = np.zeros(original.shape)
    for i in range(burnin, nb_batches):
        file_name = join(save_path, "sample" + str(i) + ".h5")
        with h5py.File(file_name, "r") as file:
            MMSE += file["MMSE"]
    MMSE /= nb_batches - burnin

    snr_obs = 10 * np.log10(
        np.linalg.norm(original) ** 2 / (np.linalg.norm(original - observations) ** 2)
    )

    snr_recons = 10 * np.log10(
        np.linalg.norm(original) ** 2 / (np.linalg.norm(original - MMSE) ** 2)
    )

    ssim_obs = ssim(
        original, observations, data_range=np.amax(original) - np.amin(original)
    )
    ssim_recons = ssim(original, MMSE, data_range=np.amax(original) - np.amin(original))

    results = {
        "atime": atime,
        "total_time": total_time,
        "std_time": time_std,
        "snr_obs": snr_obs,
        "snr_recons": snr_recons,
        "ssim_obs": ssim_obs,
        "ssim_recons": ssim_recons,
    }
    results_json = json.dumps(results)

    results_file = join(save_path, output_file_name + ".json")
    with open(results_file, "w") as file:
        file.write(results_json)


def get_implementation(mode: str):
    if mode == "serial":
        return compute_serial
    if mode == "gpu":
        return compute_gpu
    if mode == "distributed":
        return compute_distributed
    raise ValueError(f"unsupported mode '{mode}' as argument to the function")


def get_resume_implementation(mode: str):
    if mode == "serial":
        return resume_serial_sampler
    if mode == "gpu":
        return resume_gpu_sampler
    if mode == "distributed":
        return resume_distributed_sampler
    raise ValueError(f"unsupported mode '{mode}' as argument to the function")


@pytest.mark.parametrize(
    "config_file, expected_result_file",
    [
        ("data/config.json", "data/reference_inpainting_serial.json"),
        # ("data/config.json", "data/results2.json"), # second test?
    ],
)
@pytest.mark.parametrize("mode", ["serial", "gpu", "distributed"])
def test_inpainting(mode, config_file, expected_result_file):
    """Tests that the inpainting computations for different scenarios returns the same results as the ones previously validated"""
    assert os.path.exists(config_file)
    assert os.path.exists(expected_result_file)

    inpainting = get_implementation(mode)

    with open(config_file, "r") as file:
        params = json.load(file)
    args = {
        "nb_batches": params["nbCheckpoint"],
        "batch_size": params["sampleSize"],
        "save_path": params["savePath"],
        "split_coef": params["alpha"],
        "reg_coef": params["regularizationCoefficient"],
        "seed": params["seed"],
        "data_path": params["dataPath"],
    }

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=params["logFilename"],
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inpainting(**args, logger=logger)

    results_file_name = join(params["savePath"], "results_inpainting_" + mode)
    analyze_data(
        params["nbCheckpoint"],
        params["burnin"],
        params["savePath"],
        params["dataPath"],
        results_file_name,
    )

    with open(results_file_name + ".json", "r") as file:
        results = json.load(file)

    with open(expected_result_file, "r") as file:
        expected = json.load(file)

    # FIXME: example to be revised / fine-tuned to pass with default tolerance
    assert np.isclose(results["snr_recons"], expected["snr_recons"], atol=2e-1)
    assert np.isclose(results["ssim_recons"], expected["ssim_recons"], atol=1e-3)


@pytest.mark.parametrize("mode", ["serial", "gpu", "distributed"])
def test_warmstart(mode):
    assert os.path.exists("data/config.json")

    inpainting = get_implementation(mode)
    resume = get_resume_implementation(mode)

    with open("data/config.json", "r") as file:
        params = json.load(file)
    args = {
        "nb_batches": params["nbCheckpoint"],
        "batch_size": params["sampleSize"],
        "save_path": params["savePath"],
        "split_coef": params["alpha"],
        "reg_coef": params["regularizationCoefficient"],
        "seed": params["seed"],
        "data_path": params["dataPath"],
    }

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=params["logFilename"],
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    inpainting(**args, logger=logger)

    resume(
        **args,
        logger=logger,
        restart_batch=params["numLoadedBatch"],
        resume_save_path=params["reloadSavePath"],
    )

    n = params["numLoadedBatch"]
    N = params["nbCheckpoint"]

    X = []
    resumed_X = []
    potential = []
    resumed_potential = []

    for i in range(n + 1, N + 1):
        with h5py.File(
            join(params["savePath"], "sample" + str(i) + ".h5"), "r"
        ) as file:
            X = np.append(X, file["X"][:])
            potential = np.append(potential, file["potential"])
        with h5py.File(
            join(params["reloadSavePath"], "sample" + str(i) + ".h5"), "r"
        ) as file:
            resumed_X = np.append(resumed_X, file["X"][:])
            resumed_potential = np.append(resumed_potential, file["potential"][:])

    assert np.allclose(X, resumed_X)
    assert np.allclose(potential, resumed_potential)

import argparse
import json
import logging
import pathlib
from os.path import exists
from socket import gethostname

import numpy as np
from inpainting_sampling_utils import (compute_distributed, compute_gpu,
                                       compute_serial)

from mcmc.operators.inpainting_v2 import SerialInpainting
from mcmc.utils.utils import (analyze_data, generate_observations,
                              load_args_analysis_from_json,
                              load_args_from_json, load_img_size)


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

    inpainting_operator = SerialInpainting(mask)

    inpainting_params = {}
    inpainting_params["mask"] = mask

    generate_observations(
        original_path,
        inpainting_operator,
        snr,
        data_seed,
        obs_path,
        problem_parameters=inpainting_params,
        maximum=maximum,
    )


def main(mode, args, args_analysis):
    if mode == "serial":
        pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=params["logFilename"],
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
        logger.info("Host: {}".format(gethostname()))
        compute_serial(logger, **args)
        analyze_data(**args_analysis, output_file_name="results_inpainting_serial")

    if mode == "gpu":
        pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=params["logFilename"],
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
        logger.info("Host: {}".format(gethostname()))
        compute_gpu(logger, **args)
        analyze_data(**args_analysis, output_file_name="results_inpainting_gpu")

    if mode == "distributed":
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger(__name__)
            logging.basicConfig(
                filename=params["logFilename"],
                level=logging.INFO,
                filemode="w",
                format="%(asctime)s %(levelname)s %(message)s",
            )
            logger.info("Host: {}".format(gethostname()))
        else:
            logger = None

        compute_distributed(logger, **args)

        if MPI.COMM_WORLD.Get_rank() == 0:
            analyze_data(**args_analysis, output_file_name="results_inpainting_mpi")


if __name__ == "__main__":
    mode = "serial"
    config_file_name = "config_cameraman.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Selest the implementation to use.",
        default="serial",
        type=str,
        choices={"serial", "gpu", "distributed"},
    )
    parser.add_argument(
        "--config",
        help="Configuration file containing the problem parameters. Expects a .json file.",
        type=str,
    )
    args = parser.parse_args()
    if args.mode:
        mode = str(args.mode)
    if args.config:
        config_file_name = str(args.config)

    config_file = open(config_file_name)
    params = json.load(config_file)

    main_args = load_args_from_json(config_file_name)
    add_inpainting_params(main_args, config_file_name)
    args_analysis = load_args_analysis_from_json(config_file_name)

    if not exists(main_args["data_path"]):
        generate_inpainting_observations(
            params["originPath"],
            params["mask_loss"],
            params["isnr"],
            params["data_seed"],
            params["dataPath"],
            maximum=params["maximum"],
        )

    main(mode, main_args, args_analysis)

# mpirun -n 9 python -m mpi4py main.py

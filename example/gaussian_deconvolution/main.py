import json
import logging
import pathlib
from os.path import exists

import numpy as np
import argparse

from utils_deconvolution import (
    compute_serial,
    compute_gpu,
    compute_distributed,
    compute_multi_gpu,
    add_deconvolution_param,
    slice_obs_to_original,
    generate_gaussian_deconvolution_observations,
)
from mcmc.utils.utils import (
    load_args_from_json,
    load_args_analysis_from_json,
    analyze_data,
    load_img_size,
)


def main(mode, args, args_analysis, slices, results_file_name):
    if mode == "serial":
        pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=params["logFilename"],
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
        compute_serial(logger=logger, **args)
        analyze_data(
            **args_analysis,
            output_file_name=results_file_name,
            slices=slices,
        )

    if mode == "gpu":
        pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=params["logFilename"],
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
        compute_gpu(logger=logger, **args)
        analyze_data(
            **args_analysis,
            output_file_name=results_file_name,
            slices=slices,
        )

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
        else:
            logger = None

        compute_distributed(logger, **args)

        if MPI.COMM_WORLD.Get_rank() == 0:
            analyze_data(
                **args_analysis,
                output_file_name=results_file_name,
                slices=slices,
                comm_size=MPI.COMM_WORLD.Get_size(),
            )

    if mode == "multi_gpu":
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
        else:
            logger = None

        compute_multi_gpu(logger, **args)

        if MPI.COMM_WORLD.Get_rank() == 0:
            analyze_data(
                **args_analysis,
                output_file_name=results_file_name,
                slices=slices,
                comm_size=MPI.COMM_WORLD.Get_size(),
                save_picture=True,
            )


if __name__ == "__main__":
    config_file_name = "config_house.json"
    mode = "serial"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Selest the implementation to use.",
        default="serial",
        type=str,
        choices={"serial", "gpu", "distributed", "multi_gpu"},
    )
    parser.add_argument(
        "--config",
        help="Config file containing the problem parameters. Exepect a .json file.",
        type=str,
    )
    args = parser.parse_args()
    if args.mode:
        mode = str(args.mode)
    if args.config:
        config_file_name = str(args.config)

    config_file = open(config_file_name)
    params = json.load(config_file)

    config_file = open(config_file_name)
    params = json.load(config_file)

    main_args = load_args_from_json(config_file_name)
    add_deconvolution_param(config_file_name, main_args)

    args_analysis = load_args_analysis_from_json(config_file_name)

    img_dims = load_img_size(params["originPath"])
    kernel_dims = np.asarray([params["kernel_size"], params["kernel_size"]])
    slices = slice_obs_to_original(img_dims, kernel_dims)

    if not exists(main_args["data_path"]):
        generate_gaussian_deconvolution_observations(
            params["originPath"],
            params["kernel_size"],
            params["kernel_std"],
            params["isnr"],
            params["data_seed"],
            params["dataPath"],
        )

    results_file_name = "results_gaussian_deconvolution_" + mode
    main(mode, main_args, args_analysis, slices, results_file_name)
    # main(mode="gpu", config_file_name="config_cameraman.json")
    # main(mode="distributed", config_file_name="config_cameraman.json")

# mpirun -n 9 python -m mpi4py main.py
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode multi_gpu

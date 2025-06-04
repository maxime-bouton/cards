import argparse
import json
import logging
import pathlib
from os.path import exists

import numpy as np
from utils_data import (
    add_deconvolution_param,
    generate_gaussian_deconvolution_observations,
    slice_obs_to_original,
)

import mcmc.backend as backend_module
from mcmc.utils.utils import (
    analyze_data,
    load_args_analysis_from_json,
    load_sampler_params_from_json,
    load_img_size,
)


def main(mode, args, args_analysis, slices, results_file_name):
    pathlib.Path(params["savePath"]).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=params["logFilename"],
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if mode == "serial-cpu":
        backend_module.set_backend("numpy")
        from utils_deconvolution import compute_serial

        compute_serial(logger=logger, **args)

        analyze_data(
            **args_analysis,
            output_file_name=results_file_name,
            slices=slices,
        )

    if mode == "serial-gpu":
        backend_module.set_backend("cupy")
        from utils_deconvolution import compute_gpu

        compute_gpu(logger=logger, **args)

        analyze_data(
            **args_analysis,
            output_file_name=results_file_name,
            slices=slices,
        )

    if (mode == "mpi-cpu") or (mode == "mpi-gpu"):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            logger = None

        if mode == "mpi-cpu":
            backend_module.set_backend("numpy")
            from utils_deconvolution import compute_distributed

            compute_distributed(logger, **args)

        if mode == "mpi-gpu":
            backend_module.set_backend("cupy")
            backend_module.enable_multi_gpu()
            from utils_deconvolution import compute_multi_gpu

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
    # mode = "serial-cpu"
    # config_file_name = "config_house.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Select the implementation to use.",
        default="serial-cpu",
        type=str,
        choices={"serial-cpu", "serial-gpu", "mpi-cpu", "mpi-gpu"},
    )
    parser.add_argument(
        "--config",
        help="Config file containing the problem parameters. Exepect a .json file.",
        default="config_house.json",
        type=str,
    )
    args = parser.parse_args()
    if args.mode:
        mode = str(args.mode)
    if args.config:
        config_file_name = str(args.config)

    config_file = open(config_file_name)
    params = json.load(config_file)

    main_args = {}
    main_args["data_path"] = params["dataPath"]
    main_args["sampler_params"] = load_sampler_params_from_json(config_file_name)
    add_deconvolution_param(config_file_name, main_args)

    args_analysis = load_args_analysis_from_json(config_file_name)

    img_dims = load_img_size(params["originPath"])
    kernel_dims = np.asarray([params["kernel_size"], params["kernel_size"]])
    slices = slice_obs_to_original(img_dims, kernel_dims)

    if not exists(main_args["data_path"]):
        # TODO: check the scripts work from scratch = delete previously generated synthetic datasets to test (start from a clean envrionment!)
        # main doesn't work as is for now
        # FIXME: added instruction to create the path if it does not exist
        # TODO: propagate change to all similar files, check in MPI settings
        from os.path import realpath, dirname
        from pathlib import Path

        if (mode == "distributed") or (mode == "multi_gpu"):
            # FIXME: see if import not redundant with the one in main(...)
            # TODO: progressbar display issue in MPI mode: to be investigated
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_rank() == 0:
                Path(dirname(realpath(params["dataPath"]))).mkdir(
                    parents=True, exist_ok=True
                )

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

# mpirun -n 9 python -m mpi4py main.py
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi-gpu

import argparse
import json
import logging
import pathlib
from os.path import exists

from utils_data import (
    add_inpainting_params,
    generate_inpainting_observations,
)

import mcmc.backend as backend_module
from mcmc.utils.utils import (
    analyze_data,
    load_args_analysis_from_json,
    load_sampler_params_from_json,
)


def main(mode, args, args_analysis, results_file_name):
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
        from utils_inpainting import compute_serial

        compute_serial(logger, **args)
        analyze_data(**args_analysis, output_file_name=results_file_name)

    if mode == "serial-gpu":
        backend_module.set_backend("cupy")
        from utils_inpainting import compute_gpu

        compute_gpu(logger, **args)
        analyze_data(**args_analysis, output_file_name=results_file_name)

    if (mode == "mpi-cpu") or (mode == "mpi-gpu"):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            logger = None

        if mode == "mpi-cpu":
            backend_module.set_backend("numpy")
            from utils_inpainting import compute_distributed

            compute_distributed(logger, **args)

        if mode == "mpi-gpu":
            backend_module.set_backend("cupy")
            backend_module.enable_multi_gpu()
            from utils_inpainting import compute_multi_gpu

            compute_multi_gpu(logger, **args)

        if MPI.COMM_WORLD.Get_rank() == 0:
            analyze_data(
                **args_analysis,
                output_file_name=results_file_name,
                comm_size=MPI.COMM_WORLD.Get_size(),
                save_picture=True,
            )


if __name__ == "__main__":
    mode = "serial-cpu"
    config_file_name = "config_cameraman.json"

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
    add_inpainting_params(config_file_name, main_args)
    args_analysis = load_args_analysis_from_json(config_file_name)

    if not exists(main_args["data_path"]):
        generate_inpainting_observations(
            params["originPath"],
            params["mask_loss"],
            params["isnr"],
            params["data_seed"],
            params["dataPath"],
        )

    results_file_name = "results_inpainting_" + mode
    main(mode, main_args, args_analysis, results_file_name)
    # main(mode="gpu", config_file_name="config_cameraman.json")
    # main(mode="distributed", config_file_name="config_cameraman.json")

# mpirun -n 9 python -m mpi4py main.py
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi-gpu

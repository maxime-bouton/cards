import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import torch

from mcmc.backend import bm
from mcmc.logger import build_logger
from mcmc.sampler.base_sampler import SamplerParameters
from mcmc.utils.path_builder import (
    generate_obs_dir_path,
    generate_save_dir_path,
    prior_dir,
    sampling_dir,
)
from mcmc.utils.utils import analyze_data


def create_sampler_params(params: dict) -> SamplerParameters:
    sampler_params = SamplerParameters(
        params["checkpoint_size"],
        params["n_checkpoint"],
        params["seed"],
        params["save_path"],
        params["save_all"],
        params["compute_ci"],
        params["reloaded_checkpoint"],
        params.get("reloaded_path", ""),
    )
    return sampler_params


def create_analysis_params(params: dict) -> dict:
    keys = [
        "checkpoint_size",
        "n_checkpoint",
        "burnin",
        "save_path",
        "obs_path",
    ]
    return {key: params[key] for key in keys if key in params}


def build_paths(params: dict, obs_f: str, model_params_f: str, mode_str: str) -> dict:
    obs_dir_path = generate_obs_dir_path(obs_f)
    prior_f = prior_dir(params)
    sampling_f = sampling_dir(params)

    save_dir = generate_save_dir_path(
        obs_dir_path,
        prior_f,
        model_params_f,
        sampling_f,
        mode_str,
    )

    save_dir_res = save_dir / f"resumed_from_{params['reloaded_checkpoint']}"
    log_dir = save_dir_res if params["reloaded_checkpoint"] else save_dir

    return {
        "obs_path": obs_dir_path / "data.h5",
        "save_dir": save_dir,
        "save_dir_resumed": save_dir_res,
        "log_dir": log_dir,
    }


def main(
    mode: str,
    rank: int,
    comm_size: int,
    params: dict,
    args_main: dict,
    args_analysis: dict,
    results_file_name: str,
    module_name: str,
):
    Path(params["save_path"]).mkdir(parents=True, exist_ok=True)
    logger = build_logger(rank, params["logfile_path"])

    module = importlib.import_module(module_name)
    if "denoiser_params" in params:
        compute_fn = getattr(module, "compute_pnp")
    else:
        compute_fn = getattr(module, "compute_tv")
    compute_fn(logger=logger, mode=mode, **args_main)

    if rank == 0:
        analyze_data(
            **args_analysis,
            output_file_name=results_file_name,
            comm_size=comm_size,
        )


def run_main(
    get_specific_problem_params_fn: Callable[[dict], dict],
    get_specific_analysis_params_fn: Callable[[dict], dict],
    build_obs_and_model_paths_fn: Callable[[dict], tuple[str, str]],
    generate_observations_fn: Callable,
    module_name: str,
):
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
        help="Config file containing the problem parameters. Expects a .json file.",
        default="config_180.json",
        type=str,
    )
    config_args = parser.parse_args()

    with open(config_args.config) as config_file:
        params = json.load(config_file)

    mpi = "mpi" in config_args.mode
    gpu = "gpu" in config_args.mode
    if mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()

        mode_str = f"{config_args.mode}_{comm_size}"
        log_file = f"rank_{rank}.log"
    else:
        mode_str = config_args.mode
        log_file = "sampling.log"

        rank = 0
        comm_size = 1

    if gpu:
        bm.set_backend("cupy")
        gpu = bm.xp.cuda.Device(rank % bm.xp.cuda.runtime.getDeviceCount())
        # gpu = bm.xp.cuda.Device(1)
        gpu.use()

        torch.cuda.set_device(gpu.id)
        torch.set_default_device("cuda")
        torch.backends.cudnn.deterministic = True
    else:
        bm.set_backend("numpy")
        torch.set_default_device("cpu")

    paths = build_paths(params, *build_obs_and_model_paths_fn(params), mode_str)
    params["obs_path"] = params["obs_path"] or paths["obs_path"]
    params["logfile_path"] = params["logfile_path"] or paths["log_dir"] / log_file
    params["save_path"] = params["save_path"] or paths["save_dir"]

    if params["reloaded_checkpoint"]:
        params["reloaded_path"] = (
            params["save_path"] / f"checkpoint_{params['reloaded_checkpoint']}.h5"
        )
        params["save_path"] = params["save_path_resumed"] or paths["save_dir_resumed"]

    if not (obs_path := Path(params["obs_path"])).exists():
        if mpi:
            # FIXME: progress bar not displayed when using mpi (alternative to tqdm ?)
            if rank == 0:
                obs_path.parent.mkdir(parents=True, exist_ok=True)
                # TODO: generate data in parallel as well when using mpi
                generate_observations_fn(params)
            comm.Barrier()
        else:
            obs_path.parent.mkdir(parents=True, exist_ok=True)
            generate_observations_fn(params)

    args_main = {
        "obs_path": params["obs_path"],
        "sampler_params": create_sampler_params(params),
    }
    args_main.update(get_specific_problem_params_fn(params))

    args_analysis = create_analysis_params(params)
    args_analysis.update(get_specific_analysis_params_fn(params))

    results_file_name = "results"
    main(
        config_args.mode,
        rank,
        comm_size,
        params,
        args_main,
        args_analysis,
        results_file_name,
        module_name,
    )

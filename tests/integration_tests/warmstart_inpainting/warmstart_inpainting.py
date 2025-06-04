import json
import logging
import os

from utils_data import (
    add_inpainting_params,
    check_data,
)

from mcmc.utils.utils import load_sampler_params_from_json
import mcmc.backend as backend_module

import pytest


@pytest.mark.env("serial-cpu")
@pytest.mark.env("serial-gpu")
@pytest.mark.env("mpi-cpu")
@pytest.mark.env("mpi-gpu")
def test_warmstart(cmdopt):
    config_file_path = "config.json"
    assert os.path.exists(config_file_path)
    config_file = open(config_file_path)

    params = json.load(config_file)

    args = {}
    args["data_path"] = params["dataPath"]
    args["sampler_params"] = load_sampler_params_from_json(config_file_path)
    add_inpainting_params(args, config_file_path)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=params["logFilename"],
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if cmdopt == "serial-cpu":
        if not os.path.exists(params["reloadSavePath"]):
            os.makedirs(params["reloadSavePath"])

        from utils_inpainting import compute_serial, resume_serial_sampler

        compute_serial(logger, **args)
        resume_serial_sampler(
            logger,
            **args,
            restart_batch=params["numLoadedBatch"],
            resume_save_path=params["reloadSavePath"],
        )
        assert check_data(
            num_loaded_batch=params["numLoadedBatch"],
            nb_checkpoint=params["nbCheckpoint"],
            save_path=params["savePath"],
            resumed_save_path=params["reloadSavePath"],
        )

    if cmdopt == "serial-gpu":
        if not os.path.exists(params["reloadSavePath"]):
            os.makedirs(params["reloadSavePath"])

        backend_module.set_backend("cupy")
        from utils_inpainting import compute_gpu, resume_gpu_sampler

        compute_gpu(logger, **args)
        resume_gpu_sampler(
            logger,
            **args,
            restart_batch=params["numLoadedBatch"],
            resume_save_path=params["reloadSavePath"],
        )

        assert check_data(
            num_loaded_batch=params["numLoadedBatch"],
            nb_checkpoint=params["nbCheckpoint"],
            save_path=params["savePath"],
            resumed_save_path=params["reloadSavePath"],
        )

    if (cmdopt == "mpi-cpu") or (cmdopt == "mpi-gpu"):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            logger = None
            if not os.path.exists(params["reloadSavePath"]):
                os.makedirs(params["reloadSavePath"])

        if cmdopt == "mpi-cpu":
            backend_module.set_backend("numpy")
            from utils_inpainting import compute_distributed, resume_distributed_sampler

            compute_distributed(logger, **args)
            resume_distributed_sampler(
                logger,
                **args,
                restart_batch=params["numLoadedBatch"],
                resume_save_path=params["reloadSavePath"],
            )

        if cmdopt == "mpi-gpu":
            backend_module.set_backend("cupy")
            backend_module.enable_multi_gpu()
            from utils_inpainting import compute_multi_gpu, resume_multi_gpu_sampler

            compute_multi_gpu(logger, **args)
            resume_multi_gpu_sampler(
                logger,
                **args,
                restart_batch=params["numLoadedBatch"],
                resume_save_path=params["reloadSavePath"],
            )

        if MPI.COMM_WORLD.Get_rank() == 0:
            assert check_data(
                num_loaded_batch=params["numLoadedBatch"],
                nb_checkpoint=params["nbCheckpoint"],
                save_path=params["savePath"],
                resumed_save_path=params["reloadSavePath"],
            )

import json
import logging
from os.path import exists
import pytest
from mpi4py import MPI
import mcmc.backend as backend_module

from utils_data import (
    add_inpainting_params,
    check_data,
)

from mcmc.utils.utils import (
    load_sampler_params_from_json,
)

pytestmark = pytest.mark.mpi


def test_warmstart_mpi_cpu():
    config_file_path = "config.json"
    assert exists(config_file_path)
    config_file = open(config_file_path)

    params = json.load(config_file)

    args = {}
    args["data_path"] = params["dataPath"]
    args["sampler_params"] = load_sampler_params_from_json(config_file_path)
    add_inpainting_params(args, config_file_path)

    comm = MPI.COMM_WORLD

    if comm.Get_rank() == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=params["logFilename"],
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logger = None

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
    assert check_data(
        num_loaded_batch=params["numLoadedBatch"],
        nb_checkpoint=params["nbCheckpoint"],
        save_path=params["savePath"],
        resumed_save_path=params["reloadSavePath"],
    )

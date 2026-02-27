import json
import logging
from os.path import exists
import mcmc.backend as backend_module


from utils_data import (
    add_inpainting_params,
    check_data,
)

from mcmc.utils.utils import (
    load_sampler_params_from_json,
)


def test_warmstart_gpu():
    config_file_path = "config.json"
    assert exists(config_file_path)
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

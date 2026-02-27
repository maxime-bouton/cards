import json
import logging
from os.path import exists
from os.path import join
import numpy as np

from utils_data import (
    add_inpainting_params,
)

from mcmc.utils.utils import (
    analyze_data,
    load_sampler_params_from_json,
)

import mcmc.backend as backend_module


def test_check_gpu():
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
    from utils_inpainting import compute_gpu

    compute_gpu(logger, **args)
    analyze_data(
        params["nbCheckpoint"],
        params["sampleSize"],
        params["burnin"],
        params["savePath"],
        params["dataPath"],
        "results_inpainting_serial-gpu",
        show_results=False,
    )

    ref_file = open("../data/reference_inpainting_serial-gpu.json")
    ref = json.load(ref_file)

    snr_ref = ref["snr_recons"]
    ssim_ref = ref["ssim_recons"]

    results_file = open(join(params["savePath"], "results_inpainting_serial-gpu.json"))
    results = json.load(results_file)

    snr = results["snr_recons"]
    ssim = results["ssim_recons"]

    assert np.isclose(snr, snr_ref, rtol=1e-2)  #! relatve tolerance too large
    assert np.isclose(ssim, ssim_ref, rtol=1e-2)

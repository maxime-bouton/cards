import json


def add_inpainting_params(args: dict, config_file_path: str) -> None:
    config_file = open(config_file_path)
    params = json.load(config_file)

    args["split_coef"] = params["alpha"]
    args["reg_coef"] = params["regularizationCoefficient"]
    return

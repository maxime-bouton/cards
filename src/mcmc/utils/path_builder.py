from pathlib import Path

PRODUCED_DATA_PATH = Path("../../produced_data")


def clean(func):
    """
    Decorator to clean the string returned by the function.
    It replaces dots with underscores to avoid issues with file paths.
    """

    def wrapper(*args, **kwargs):
        s = func(*args, **kwargs)
        return s.replace(".", "_")

    return wrapper


### application strings ###
@clean
def deconvolution_str(params: dict) -> str:
    params_k = params["kernel"]
    if params_k["type"] == "motion":
        return f"motion_size{params_k['size']}-intensity{params_k['intensity']}"
    else:
        return f"gaussian_size{params_k['size']}-std{params_k['std']}"


@clean
def inpainting_str(params: dict) -> str:
    return f"mask{params['mask_loss']}"


### noise strings ###
@clean
def gaussian_str(params: dict) -> str:
    return f"isnr{params['isnr']}"


def poisson_str(params: dict) -> str:
    return f"dynamic_range{params['dynamic_range']}"


### observation strings ###
def obs_dir(params: dict, application_str: str, noise_str: str) -> str:
    img_name = Path(params["original_img_path"]).stem
    return f"{img_name}-{application_str}-{noise_str}-data_seed{params['seed_data']}"


### prior strings ###
@clean
def prior_dir(params: dict) -> str:
    if "denoiser_params" in params:
        return f"{denoiser_dir(params)}/{pnp_ula_dir(params)}"
    else:
        return f"tv/reg{params['reg_coef']}"


def denoiser_dir(params: dict) -> str:
    denoiser = params["denoiser_params"]
    denoiser_type = denoiser["type"]
    denoiser_str = denoiser_type.lower()
    if denoiser_type == "ddfb":
        denoiser_str += (
            f"/n_layers{denoiser['n_layers']}-n_features{denoiser['n_features']}"
        )
    return denoiser_str


@clean
def pnp_ula_dir(params: dict) -> str:
    denoiser = params["denoiser_params"]
    pnp_ula_str = f"reg{params['reg_coef']}"
    if (lv := denoiser["denoising_level"]) is not None:
        pnp_ula_str += f"-beta{lv}"
    if (L := denoiser["L"]) is not None:
        pnp_ula_str += f"-L{L}"
    return pnp_ula_str


### sampling strings ###
def sampling_dir(params: dict) -> str:
    return f"checkpoint_size{params['checkpoint_size']}-seed{params['seed']}"


### combined path builder ###
def generate_obs_dir_path(obs_dir: str):
    return PRODUCED_DATA_PATH / obs_dir


def generate_save_dir_path(
    obs_dir_path: Path,
    prior_dir: str,
    model_params_dir: str,
    sampling_params_dir: str,
    mode_str: str,
) -> Path:
    return obs_dir_path / prior_dir / model_params_dir / sampling_params_dir / mode_str

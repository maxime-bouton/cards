import json
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class Application(Enum):
    DECONVOLUTION = auto()
    INPAINTING = auto()


class Prior(Enum):
    TV = auto()
    PnP = auto()


@dataclass
class Noise(ABC): ...


@dataclass
class Gaussian(Noise):
    sigma: float


@dataclass
class Poisson(Noise):
    M: int


@dataclass
class Denoiser(ABC):
    denoising_level: float | None


@dataclass
class DDFB(Denoiser):
    n_layers: int
    n_features: int


@dataclass
class DnCNN(Denoiser): ...


@dataclass
class DRUNet(Denoiser): ...


@dataclass
class Kernel:
    size: int | tuple[int, int]


@dataclass
class GaussianKernel(Kernel):
    std: float | tuple[float, float]


@dataclass
class MotionKernel(Kernel):
    intensity: float
    seed: int


if __name__ == "__main__":
    # sampling
    checkpoint_size = 1000
    n_checkpoint = 10
    burnin = 2
    id_reload = 0
    seed = 42
    compute_ci = True
    save_all = True

    # paths
    image_name = "2048.h5"
    save_path = ""
    reload_save_path = ""
    log_path = ""

    # application
    application = Application.INPAINTING

    # prior
    prior = Prior.PnP
    reg_coef = 1

    # gaussian with TV
    split_coef = 1e-3

    # poisson deconv
    split_coef1 = 1e-3
    split_coef2 = 1e-3

    # noise
    data_seed = 1234
    isnr = 25
    noise = Gaussian(0.01)
    # noise = Poisson(50)

    # deconvolution
    size = 27
    # kernel = GaussianKernel(size, std=1)
    kernel = MotionKernel(size, intensity=0.5, seed=data_seed)

    # inpainting
    mask_loss = 0.8

    # denoiser
    denoising_level = None
    denoiser: Denoiser = DDFB(denoising_level, n_layers=4, n_features=64)
    L: float | None = None

    ###########################################
    #### NO NEED TO MODIFY BELOW THIS LINE ####
    ###########################################

    str_app = f"{noise.__class__.__name__.lower()}-{application.name.lower()}"
    original_img_path = Path("../../data") / image_name

    json_dict = {
        "checkpoint_size": checkpoint_size,
        "n_checkpoint": n_checkpoint,
        "burnin": burnin,
        "reloaded_checkpoint": id_reload,
        "seed": seed,
        "compute_ci": compute_ci,
        "save_all": save_all,
        "original_img_path": str(original_img_path),
        "seed_data": data_seed,
        "obs_path": "",
        "save_path": "",
        "save_path_resumed": "",
        "logfile_path": "",
        "reg_coef": reg_coef,
    }

    if type(noise) is Gaussian:
        json_dict["isnr"] = isnr
        json_dict["split_coef"] = split_coef
    elif type(noise) is Poisson:
        json_dict["dynamic_range"] = noise.M
        json_dict["split_coef1"] = split_coef1
        json_dict["split_coef2"] = split_coef2

    if prior == Prior.PnP:
        json_dict["denoiser_params"] = {
            "type": denoiser.__class__.__name__.lower(),
            "n_layers": denoiser.n_layers,
            "n_features": denoiser.n_features,
            "denoising_level": denoiser.denoising_level,
            "L": L,
        }

    if application == Application.INPAINTING:
        json_dict["mask_loss"] = mask_loss
    elif application == Application.DECONVOLUTION:
        if isinstance(kernel, MotionKernel):
            json_dict["kernel"] = {
                "type": "motion",
                "size": kernel.size,
                "intensity": kernel.intensity,
                "seed": kernel.seed,
            }
        else:
            json_dict["kernel"] = {
                "type": "gaussian",
                "size": kernel.size,
                "std": kernel.std,
            }

    path_saving = (
        Path(__file__).parents[1]
        / "example"
        / str_app.replace("-", "_")
        / f"config_{original_img_path.stem}_{'pnp' if prior == Prior.PnP else 'tv'}.json"
    )

    with open(path_saving, "w") as f:
        json.dump(json_dict, f, indent=4)

r"""Utility functions to load the weights of a denoiser."""

from pathlib import Path

import torch

import cards.backend as xp
from cards.denoisers.ddfb.network_ddfb import DDFB
from cards.denoisers.dncnn.network_dncnn import DnCNN
from cards.denoisers.drunet.network_drunet import DRUNet


def get_torch_device() -> torch.device:
    """Read the current cards backend and return the corresponding PyTorch device."""
    if xp.get_backend() == "cupy":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def load_pretrained_ddfb(
    C=3,
    n_layers=4,
    n_features=64,
    weights_path=Path(__file__).parents[3] / "data/weights/ddfb",
) -> DDFB:
    r"""Instantiate a DDFB model with pre-trained weights.

    This function creates an instance of the DDFB model and loads pre-trained weights
    from the specified path. It also sets the model to evaluation mode and updates the
    lip size based on the provided image size.

    Parameters
    ----------
    C : int, optional
        Number of channels in the input image. Default is 3.
    n_layers : int, optional
        Number of layers in the unfolded network. Default is 4.
    n_features : int, optional
        Number of feature maps per layer. Default is 64.
    weights_path : str, optional
        Path to the pre-trained weights folder. Default is `"data/weights/ddfb"`.

    Returns
    -------
    DDFB
        An instance of the DDFB model with the specified weights.
    """
    device = get_torch_device()
    weights = weights_path / f"ddfb_nch{C}_nla{n_layers}_nfe{n_features}.pth"
    net = DDFB(C=C, n_layers=n_layers, n_features=n_features)
    net.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    net.to(device)
    net.eval()
    return net


def load_pretrained_drunet(
    nch=3,
    weights_path=Path(__file__).parents[3] / "data/weights/drunet",
) -> DRUNet:
    device = get_torch_device()
    weights = weights_path / f"drunet_nch{nch}.pth"
    net = DRUNet(in_nc=nch + 1, out_nc=nch, act_mode="R", bias=False)
    net.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    net.to(device)
    net.eval()
    return net


def load_pretrained_dncnn(
    nch=3,
    weights_path=Path(__file__).parents[3] / "data/weights/dncnn",
) -> DnCNN:
    device = get_torch_device()
    weights = weights_path / f"dncnn_nch{nch}.pth"
    net = DnCNN(in_nc=nch, out_nc=nch, act_mode="R")
    net.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    net.to(device)
    net.eval()
    return net


def build_denoiser(net):
    if isinstance(net, DRUNet):
        return lambda x, sigma=None, u=None: net(x, sigma)
    else:
        return lambda x, sigma=None, u=None: net(x, sigma, u)

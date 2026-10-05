r"""Statistical models for computational imaging inverse problems.

This package provides the core model abstractions and concrete implementations for a
variety of computational imaging inverse problems, including Gaussian and Poisson
deconvolution and inpainting.

Base Classes
------------
:class:`~cards.models.base_model.BaseModel`
    Abstract model class defining the interface for all statistical models to
    be used within a :class:`~cards.samplers.base_sampler.BaseSampler` for
    solving inverse problems.
:class:`~cards.models.base_model.BaseDistributedModel`
    Extends :class:`~cards.models.base_model.BaseModel` to support distributed
    computation using MPI.

Concrete Models
---------------
Gaussian Deconvolution (PnP)
    Gaussian deconvolution regularized via Plug-and-Play (PnP) denoising priors.

    * **Serial:** :class:`~cards.models.gaussian_deconvolution_pnp_model.GaussianDeconvolutionPnpModel`
    * **Distributed:** :class:`~cards.models.gaussian_deconvolution_pnp_model.DistributedGaussianDeconvolutionPnpModel`

Gaussian Deconvolution (TV)
    Gaussian deconvolution regularized via Total Variation (TV).

    * **Serial:** :class:`~cards.models.gaussian_deconvolution_tv_model.GaussianDeconvolutionTvModel`
    * **Distributed:** :class:`~cards.models.gaussian_deconvolution_tv_model.DistributedGaussianDeconvolutionTvModel`

Gaussian Inpainting (PnP)
    Gaussian inpainting regularized via Plug-and-Play (PnP) denoising priors.

    * **Serial:** :class:`~cards.models.gaussian_inpainting_pnp_model.GaussianInpaintingPnpModel`
    * **Distributed:** :class:`~cards.models.gaussian_inpainting_pnp_model.DistributedGaussianInpaintingPnpModel`

Gaussian Inpainting (TV)
    Gaussian inpainting regularized via Total Variation (TV).

    * **Serial:** :class:`~cards.models.gaussian_inpainting_tv_model.GaussianInpaintingTvModel`
    * **Distributed:** :class:`~cards.models.gaussian_inpainting_tv_model.DistributedGaussianInpaintingTvModel`

Poisson Deconvolution (PnP)
    Poisson deconvolution regularized via Plug-and-Play (PnP) denoising priors.

    * **Serial:** :class:`~cards.models.poisson_deconvolution_pnp_model.PoissonDeconvolutionPnpModel`
    * **Distributed:** :class:`~cards.models.poisson_deconvolution_pnp_model.DistributedPoissonDeconvolutionPnpModel`

Poisson Deconvolution (TV)
    Poisson deconvolution regularized via Total Variation (TV).

    * **Serial:** :class:`~cards.models.poisson_deconvolution_tv_model.PoissonDeconvolutionTvModel`
    * **Distributed:** :class:`~cards.models.poisson_deconvolution_tv_model.DistributedPoissonDeconvolutionTvModel`

Examples
--------
>>> #TODO: add example usage of the models here
"""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from .base_model import BaseDistributedModel, BaseModel
from .gaussian_deconvolution_pnp_model import (
    DistributedGaussianDeconvolutionPnpModel,
    GaussianDeconvolutionPnpModel,
)
from .gaussian_deconvolution_tv_model import (
    DistributedGaussianDeconvolutionTvModel,
    GaussianDeconvolutionTvModel,
)
from .gaussian_inpainting_pnp_model import (
    DistributedGaussianInpaintingPnpModel,
    GaussianInpaintingPnpModel,
)
from .gaussian_inpainting_tv_model import (
    DistributedGaussianInpaintingTvModel,
    GaussianInpaintingTvModel,
)
from .poisson_deconvolution_pnp_model import (
    DistributedPoissonDeconvolutionPnpModel,
    PoissonDeconvolutionPnpModel,
)
from .poisson_deconvolution_tv_model import (
    DistributedPoissonDeconvolutionTvModel,
    PoissonDeconvolutionTvModel,
)

__all__ = [
    "BaseDistributedModel",
    "BaseModel",
    "DistributedGaussianDeconvolutionPnpModel",
    "DistributedGaussianDeconvolutionTvModel",
    "DistributedGaussianInpaintingPnpModel",
    "DistributedGaussianInpaintingTvModel",
    "DistributedPoissonDeconvolutionPnpModel",
    "DistributedPoissonDeconvolutionTvModel",
    "GaussianDeconvolutionPnpModel",
    "GaussianDeconvolutionTvModel",
    "GaussianInpaintingPnpModel",
    "GaussianInpaintingTvModel",
    "PoissonDeconvolutionPnpModel",
    "PoissonDeconvolutionTvModel",
]

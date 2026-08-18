import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")

from gaussian_deconvolution_pnp import (
    GaussianDeconvObservationsHook,
    GaussianDeconvPnpMcmcHook,
    PnpDeconvGeometryHook,
)

from cards.core.simulation import Simulation


def main():

    geom_hk = PnpDeconvGeometryHook()
    obs_hk = GaussianDeconvObservationsHook()
    mcmc_hk = GaussianDeconvPnpMcmcHook()

    simu = Simulation.from_cli(geom_hk, obs_hk, mcmc_hk)
    simu.run()


if __name__ == "__main__":
    main()

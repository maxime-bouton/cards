# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import warnings


from gaussian_deconvolution_pnp import (
    GaussianDeconvObservationsHook,
    GaussianDeconvPnpMcmcHook,
    PnpDeconvGeometryHook,
)

from cards.core.simulation import Simulation

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")


def main():

    geom_hk = PnpDeconvGeometryHook()
    obs_hk = GaussianDeconvObservationsHook()
    mcmc_hk = GaussianDeconvPnpMcmcHook()

    simu = Simulation.from_cli(geom_hk, obs_hk, mcmc_hk)
    simu.run()


if __name__ == "__main__":
    main()

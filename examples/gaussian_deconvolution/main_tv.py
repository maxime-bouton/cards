# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from gaussian_deconvolution_tv import (
    GaussianDeconvObservationsHook,
    GaussianDeconvTvMcmcHook,
    TvDeconvGeometryHook,
)

from cards.core.simulation import Simulation

# TODO: to be fused with main_rework.py


def main():

    geom_hk = TvDeconvGeometryHook()
    obs_hk = GaussianDeconvObservationsHook()
    mcmc_hk = GaussianDeconvTvMcmcHook()

    simu = Simulation.from_cli(geom_hk, obs_hk, mcmc_hk)
    simu.run()


if __name__ == "__main__":
    main()

# python examples/gaussian_deconvolution/main_tv.py --config examples/gaussian_deconvolution/configs/config_gdec_128_tv.json
# python examples/gaussian_deconvolution/main_tv.py --config examples/gaussian_deconvolution/configs/config_gdec_128_tv.json --device gpu
# mpirun -np 2 python examples/gaussian_deconvolution/main_tv.py --config examples/gaussian_deconvolution/configs/config_gdec_128_tv.json --mode mpi
# mpirun -np 2 -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n python examples/gaussian_deconvolution/main_tv.py --config examples/gaussian_deconvolution/configs/config_gdec_128_tv.json --mode mpi --device gpu

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

from utils_inpainting import (
    build_obs_and_model_paths,
    gaussian_inpainting_params,
    generate_inpainting_observations,
)

from cards.utils.main_helper import run_main

if __name__ == "__main__":
    run_main(
        gaussian_inpainting_params,
        lambda d: {},
        build_obs_and_model_paths,
        lambda p, mode: generate_inpainting_observations(
            p["original_img_path"],
            p["mask_loss"],
            p["isnr"],
            p["seed_data"],
            p["obs_path"],
            mode=mode,
        ),
        module_name="utils_inpainting",
        save_picture=False,
        show_results=False,
    )

# python -m main --config config.json --mode serial --device cpu
# python -m main --config config.json --mode serial --device gpu
# mpirun -n 2 python -m mpi4py main.py --config config.json --mode mpi --device cpu
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --config config.json --mode mpi --device gpu

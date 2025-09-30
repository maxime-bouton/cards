from mcmc.utils.main_helper import run_main
from utils_inpainting import (
    gaussian_inpainting_params,
    build_obs_and_model_paths,
    generate_inpainting_observations,
)


if __name__ == "__main__":
    run_main(
        gaussian_inpainting_params,
        lambda d: {},
        build_obs_and_model_paths,
        lambda p: generate_inpainting_observations(
            p["original_img_path"],
            p["mask_loss"],
            p["isnr"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_inpainting",
    )


# mpirun -n 9 python -m mpi4py main.py
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi-gpu

from mcmc.utils.main_helper import run_main
from utils_data import (
    gaussian_deconvolution_params,
    define_slices,
    build_obs_and_model_paths,
    generate_gaussian_deconvolution_observations,
)

if __name__ == "__main__":
    run_main(
        gaussian_deconvolution_params,
        define_slices,
        build_obs_and_model_paths,
        generate_observations_fn=lambda p: generate_gaussian_deconvolution_observations(
            p["original_img_path"],
            p["kernel"],
            p["isnr"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_gaussian_deconvolution",
    )

# mpirun -n 9 python -m mpi4py main.py
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi-gpu

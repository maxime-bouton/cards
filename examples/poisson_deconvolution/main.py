from utils_poisson_deconvolution import (
    build_obs_and_model_paths,
    define_slices,
    generate_poisson_deconvolution_observations,
    poisson_deconvolution_params,
)

from cards.utils.main_helper import run_main

if __name__ == "__main__":
    run_main(
        poisson_deconvolution_params,
        define_slices,
        build_obs_and_model_paths,
        lambda p: generate_poisson_deconvolution_observations(
            p["original_img_path"],
            p["kernel"],
            p["dynamic_range"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_poisson_deconvolution",
        save_picture=False,
        show_results=False,
    )

# python -m main --config config.json --mode serial --device cpu
# python -m main --config config.json --mode serial --device gpu
# mpirun -n 2 python -m mpi4py main.py --config config.json --mode mpi --device cpu
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi --device gpu

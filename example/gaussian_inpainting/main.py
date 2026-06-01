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
        lambda p: generate_inpainting_observations(
            p["original_img_path"],
            p["mask_loss"],
            p["isnr"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_inpainting",
        save_picture=False,
        show_results=False,
    )

# python -m main --config config.json --mode serial --device cpu
# python -m main --config config.json --mode serial --device gpu
# mpirun -n 2 python -m mpi4py main.py --config config.json --mode mpi --device cpu
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --config config.json --mode mpi --device gpu

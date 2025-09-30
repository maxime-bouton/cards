from mcmc.utils.main_helper import run_main

from utils_poisson_deconvolution import (
    poisson_deconvolution_params,
    define_slices,
    build_obs_and_model_paths,
    generate_poisson_deconvolution_observations,
)


if __name__ == "__main__":
    run_main(
        poisson_deconvolution_params,
        define_slices,
        build_obs_and_model_paths,
        generate_observations_fn=lambda p: generate_poisson_deconvolution_observations(
            p["original_img_path"],
            p["kernel"],
            p["dynamic_range"],
            p["seed_data"],
            p["obs_path"],
        ),
        module_name="utils_poisson_deconvolution",
        save_picture=True,
    )

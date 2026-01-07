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

# FIXME: issue with PnP
# File "/home/pthouvenin/python/cards/example/poisson_deconvolution/main.py", line 12, in <module>
#     run_main(
#     ~~~~~~~~^
#         poisson_deconvolution_params,
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     ...<10 lines>...
#         save_picture=True,
#         ^^^^^^^^^^^^^^^^^^
#     )
#     ^
#   File "/home/pthouvenin/python/cards/src/cards/utils/main_helper.py", line 225, in run_main
#     main(
#     ~~~~^
#         config_args.mode,
#         ^^^^^^^^^^^^^^^^^
#     ...<9 lines>...
#         show_results=show_results,
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^
#     )
#     ^
#   File "/home/pthouvenin/python/cards/src/cards/utils/main_helper.py", line 103, in main
#     compute_fn(logger=logger, mode=mode, **args_main)
#     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/pthouvenin/python/cards/example/poisson_deconvolution/utils_poisson_deconvolution.py", line 417, in compute_pnp
#     denoiser = SerialDDFB(
#         image_size=np.asarray(gt_shape),
#         n_layers=denoiser_params["n_layers"],
#         n_features=denoiser_params["n_features"],
#     )
#   File "/home/pthouvenin/python/cards/src/cards/denoisers/serial_ddfb.py", line 39, in __init__
#     image_size[-3],
#     ~~~~~~~~~~^^^^
# IndexError: index -3 is out of bounds for axis 0 with size 2

# TODO: all TV configs run, PnP to do
# python -m main --config config.json --mode serial --device cpu
# python -m main --config config.json --mode serial --device gpu
# mpirun -n 2 python -m mpi4py main.py --config config.json --mode mpi --device cpu
# mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m mpi4py main.py --mode mpi --device gpu

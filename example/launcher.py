import collections.abc
import copy
import itertools
import json
import os
import subprocess
import time


def deep_update(d, u):
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def build_command(config_name, workers):
    """Constructs the command list for GPU execution based on worker count."""
    device = "gpu"
    base_cmd = ["python", "main.py"]

    mode = "serial" if workers == 1 else "mpi"

    script_args = ["--config", config_name, "--mode", mode, "--device", device]

    # serial execution (no MPI)
    if workers == 1:
        return base_cmd + script_args

    # else, mpi execution with CUDA-aware flags
    mpi_base = ["mpirun", "-np", str(workers)]
    mpi_gpu_flags = [
        "-x",
        "OMPI_MCA_pml=ucx",
        "-x",
        "OMPI_MCA_osc=ucx",
        "-x",
        "OMPI_MCA_opal_cuda_support=true",
        "-x",
        "UCX_MEMTYPE_CACHE=n",
    ]
    return mpi_base + mpi_gpu_flags + base_cmd + script_args


def main():
    applications = [
        "gaussian_deconvolution",
        "gaussian_inpainting",
        "poisson_deconvolution",
    ]

    observation_configs = [
        "config_128.json",
        # "config_2048.json",
        # "config_2896.json",
        # "config_4096.json",
    ]

    prior_configs = [
        "config_ddfb.json",
        "config_dncnn.json",
        # "config_drunet.json",
        "config_tv.json",
    ]

    workers = [1, 2, 4]

    # load common config (contains settings that are shared across all experiments)
    with open("config_common.json", "r") as f:
        common_config = json.load(f)

    # generate all combinations of experiments (Cartesian product)
    experiments = list(
        itertools.product(applications, observation_configs, prior_configs, workers)
    )

    print(f"Total experiments queued: {len(experiments)}\n" + "=" * 50)

    for app, obs_name, prior_name, w in experiments:
        obs_path = os.path.join(app, obs_name)
        prior_path = os.path.join(app, prior_name)

        # skip if config files are missing
        if not (os.path.exists(obs_path) and os.path.exists(prior_path)):
            print(f"[{app}] Missing config files. Skipping {obs_name} + {prior_name}")
            continue

        # load the observation and prior configs
        with open(obs_path, "r") as f:
            obs_data = json.load(f)
        with open(prior_path, "r") as f:
            prior_data = json.load(f)

        # merge configs (prior overwrites observation, which overwrites common)
        merged_config = copy.deepcopy(common_config)
        deep_update(merged_config, obs_data)
        deep_update(merged_config, prior_data)

        # save the merged config for record-keeping and reproducibility
        obs_prefix = obs_name.replace("config_", "").replace(".json", "")
        prior_prefix = prior_name.replace("config_", "").replace(".json", "")

        run_config_name = f"run_{obs_prefix}_{prior_prefix}_w{w}.json"
        run_config_path = os.path.join(app, run_config_name)

        with open(run_config_path, "w") as f:
            json.dump(merged_config, f, indent=4)

        # build and execute the command for this experiment
        cmd_list = build_command(config_name=run_config_name, workers=w)

        print(
            f"\n[LAUNCHING] App: {app} | Obs: {obs_prefix} | Prior: {prior_prefix} | Workers: {w}"
        )
        print(f"CMD: {' '.join(cmd_list)}")

        try:
            # set timeout to 10 hours (36000 seconds) to prevent runaway processes
            subprocess.run(cmd_list, cwd=app, check=True, timeout=36000)

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Experiment failed with exit code {e.returncode}")

        except subprocess.TimeoutExpired:
            print("\n[TIMEOUT] Experiment took too long and was killed.")

        finally:
            print("Sleeping for 1 second to let VRAM flush...")
            time.sleep(1)


if __name__ == "__main__":
    main()

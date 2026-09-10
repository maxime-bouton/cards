# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import argparse
import collections.abc
import copy
import itertools
import json
import subprocess
import textwrap
import time
from pathlib import Path

APPLICATIONS = [
    "gaussian_deconvolution",
    "gaussian_inpainting",
    "poisson_deconvolution",
]

OBS_CONFIGS = [
    "128.json",
    # "2048.json",
    # "2896.json",
    # "4096.json",
]

PRIOR_CONFIGS = [
    "ddfb.json",
    "dncnn.json",
    "drunet.json",
    "tv.json",
]

WORKERS = [1, 2, 4]
DEVICE = "gpu"

ABBS = {
    "gaussian_deconvolution": "gdec",
    "gaussian_inpainting": "ginp",
    "poisson_deconvolution": "pdec",
}


TIME_LIMITS = {
    "128": "00:30:00",
    "2048": "04:00:00",
    "2896": "08:00:00",
    "4096": "15:00:00",
}

ENV_SETTINGS = "module load <MODULE>; conda activate <PATH_TO_ENV>"


def get_time_limit(obs_prefix):
    """Returns the time limit based on the observation size, defaulting to 2 hours."""
    return TIME_LIMITS.get(obs_prefix, "02:00:00")


def _deep_update(d, u):
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _create_config_file(app, obs_path: Path, prior_path: Path, common_config):
    obs_prefix = obs_path.stem
    prior_prefix = prior_path.stem
    app_short = ABBS.get(app, app)

    config_name = f"config_{app_short}_{obs_prefix}_{prior_prefix}.json"

    output_dir = Path(app) / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / config_name

    if config_path.exists():
        print(f"[SKIP] {config_name} already exists.")
        return config_name

    with open(obs_path, "r") as f:
        obs_data = json.load(f)
    with open(prior_path, "r") as f:
        prior_data = json.load(f)

    merged_config = copy.deepcopy(common_config)
    _deep_update(merged_config, obs_data)
    _deep_update(merged_config, prior_data)

    with open(config_path, "w") as f:
        json.dump(merged_config, f, indent=4)

    print(f"[CREATE] {config_name} created.")
    return config_name


def build_command(config_name, workers, device=DEVICE):
    """Constructs the command list for GPU execution based on worker count."""

    rel_config_path = f"configs/{config_name}"

    base_cmd = ["python", "main.py"]
    mode = "serial" if workers == 1 else "mpi"
    script_args = ["--config", rel_config_path, "--mode", mode, "--device", device]

    if workers == 1:
        return base_cmd + script_args

    mpi_base = ["mpirun", "-np", str(workers)]

    if device == "gpu":
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
    else:
        return mpi_base + base_cmd + script_args


def generate_slurm_script(
    app,
    job_name,
    workers,
    config_name,
    time_limit,
    device=DEVICE,
    env_settings=ENV_SETTINGS,
):
    """Generates the SLURM script content."""

    rel_config_path = f"configs/{config_name}"

    if workers == 1:
        cmd_str = (
            f"python main.py --config {rel_config_path} --mode serial --device {device}"
        )
    else:
        cmd_str = f"mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np {workers} python main.py --config {rel_config_path} --mode mpi --device {device}"

    # textwrap.dedent removes the leading python indentation so SLURM reads the #SBATCH tags correctly
    slurm_content = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition=gpu_p13
        #SBATCH -C v100-32g
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --hint=nomultithread
        #SBATCH --output=logs/{job_name}_%j.out
        #SBATCH --error=logs/{job_name}_%j.err
        #SBATCH --ntasks-per-node={workers}
        #SBATCH --gres=gpu:{workers}
        #SBATCH --time={time_limit}

        module purge
        {env_settings}

        cd {app}
        set -x

        {cmd_str}
        """)
    return slurm_content


def main(args):
    with open("template_common.json", "r") as f:
        common_config = json.load(f)

    configs = list(itertools.product(APPLICATIONS, OBS_CONFIGS, PRIOR_CONFIGS))
    print(f"Processing {len(configs)} configurations...\n" + "-" * 40)

    for app, obs_name, prior_name in configs:
        obs_path = Path(app) / "templates" / "obs" / obs_name
        prior_path = Path(app) / "templates" / "priors" / prior_name

        if not obs_path.exists():
            print(f"[{app}] Missing {obs_name}. Skipping.")
            continue
        if not prior_path.exists():
            print(f"[{app}] Missing {prior_name}. Skipping.")
            continue

        config_name = _create_config_file(app, obs_path, prior_path, common_config)
        base_job_name = config_name.replace("config_", "").replace(".json", "")

        for w in WORKERS:
            job_name = f"{base_job_name}_w{w}"

            if args.run:
                cmd_list = build_command(config_name, w)
                print(f"\n[LAUNCHING LOCAL] {job_name}")
                try:
                    subprocess.run(cmd_list, cwd=app, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed with code {e.returncode}")
                finally:
                    time.sleep(2)

            if args.slurm or args.submit:
                (Path(app) / "slurm").mkdir(parents=True, exist_ok=True)

                time_limit = get_time_limit(obs_path.stem)
                slurm_content = generate_slurm_script(
                    app, job_name, w, config_name, time_limit, DEVICE
                )

                slurm_path = Path(app) / "slurm" / f"{job_name}.slurm"
                with open(slurm_path, "w") as f:
                    f.write(slurm_content)

                if args.slurm and not args.submit:
                    print(f"[GENERATED] {slurm_path}")

                if args.submit:
                    (Path(app) / "logs").mkdir(parents=True, exist_ok=True)
                    print(f"[SUBMITTING] {slurm_path}")
                    try:
                        subprocess.run(
                            ["sbatch", f"slurm/{job_name}.slurm"], cwd=app, check=True
                        )
                        time.sleep(0.5)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] SLURM submission failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment launcher and SLURM generator."
    )
    parser.add_argument(
        "--run", help="Build configs and run locally.", action="store_true"
    )
    parser.add_argument(
        "--slurm", help="Build configs and generate SLURM scripts.", action="store_true"
    )
    parser.add_argument(
        "--submit", help="Generate SLURM scripts and submit.", action="store_true"
    )

    args = parser.parse_args()

    main(args)

from pathlib import Path

import numpy as np

from mcmc.utils.utils import analyze_data
from mcmc.utils.utils_img import read_img_shape


def walk_and_process(root_dir: Path, burnin: int = 0):
    for h5_file in root_dir.rglob("data.h5"):
        data_dir = h5_file.parent
        poisson = h5_file.parent.parent.name.split("-")[0] == "poisson"

        if poisson:
            rg = int("".join([c for c in data_dir.name.split("-")[3] if c.isdigit()]))
        else:
            rg = 1
        y_shape = read_img_shape(h5_file, key="y")
        x_shape = read_img_shape(h5_file, key="x")
        diff = np.asarray(y_shape) - np.asarray(x_shape)
        slices = tuple(np.s_[(d // 2) or None : (-d // 2) or None] for d in diff)

        for dirpath, dirnames, _ in data_dir.walk():
            for dirname in dirnames:
                if dirname in ["serial-gpu", "mpi-gpu_2", "mpi-gpu_4"]:
                    full_dir_path = Path(dirpath) / dirname
                    count = sum(1 for f in full_dir_path.glob("checkpoint_*.h5"))
                    comm_size = (
                        int(dirname.split("_")[-1]) if "mpi-gpu" in dirname else 1
                    )
                    checkpoint_size = int(
                        "".join([c for c in dirpath.name.split("-")[0] if c.isdigit()])
                    )

                    if burnin < count:
                        print(f"Processing {full_dir_path} with {count} checkpoints")
                        analyze_data(
                            n_checkpoint=count,
                            checkpoint_size=checkpoint_size,
                            burnin=burnin,
                            save_path=str(full_dir_path),
                            obs_path=str(h5_file),
                            output_file_name="results",
                            slices=slices,
                            comm_size=comm_size,
                            obs_rg=rg,
                        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MMSE from checkpoint aggregation and related metrics."
    )
    parser.add_argument("--root_dir", type=Path, help="Root directory")
    parser.add_argument("--burnin", type=int, default=1, help="Burn-in period")
    args = parser.parse_args()

    walk_and_process(args.root_dir, args.burnin)

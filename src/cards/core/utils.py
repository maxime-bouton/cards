import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Select the implementation to use.",
        default="serial",
        type=str,
        choices={"serial", "mpi"},
    )
    parser.add_argument(
        "--device",
        help="Select the type of hardware to use (CPU or GPU).",
        default="cpu",
        type=str,
        choices={"cpu", "gpu"},
    )
    parser.add_argument(
        "--config",
        help="Config file containing the problem parameters. Expects a .json file.",
        default="config.json",
        type=str,
    )
    return parser.parse_args()

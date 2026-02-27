from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def save_tensor_as_image(tensor: np.ndarray, out_path: Path, range: int = 1):
    """Save a (C, H, W) numpy array as a PNG image."""
    if tensor.ndim != 3:
        raise ValueError("Expected tensor of shape (C, H, W)")

    C, H, W = tensor.shape
    if C not in [1, 3]:
        raise ValueError(f"Only 1 or 3 channel images supported, got C={C}")

    tensor = np.clip(tensor / range, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)

    if C == 1:
        img = Image.fromarray(tensor[0], mode="L")
    else:
        img = Image.fromarray(np.transpose(tensor, (1, 2, 0)), mode="RGB")

    img.save(out_path, format="JPEG")


def process_h5_file(file_path: Path, key: str):
    out_dir = file_path.parent
    print(f"Processing {file_path} for key '{key}'")

    try:
        with h5py.File(file_path, "r") as f:
            if key not in f:
                print(f"Warning: '{key}' not found in {file_path}")
                return
            rg = 1
            if key in ["x", "y"]:
                size = int(out_dir.name.split("-")[0])
                if key == "y" and "dynamic_range" in out_dir.name:
                    rg = int(
                        "".join([d for d in out_dir.name.split("-")[3] if d.isdigit()])
                    )
                out_filename = (
                    f"{key.lower()}_{out_dir.parent.name.replace('-', '_')}_{size}.jpg"
                )
            else:
                out_filename = f"{key.lower()}.jpg"

            array = f[key][()]
            if array.ndim == 2:
                array = array[np.newaxis, :, :]  # Convert (H, W) → (1, H, W)

            out_path = out_dir / out_filename
            save_tensor_as_image(array, out_path, rg)
            # print(f"Saved {key} from {file_path} to {out_path}")
    except OSError as e:
        print(f"Error processing {file_path}: {e}")


def walk_and_process(
    root_dir: Path,
    key: str,
    filename: str,
):
    for h5_file in root_dir.rglob(f"{filename}.h5"):
        process_h5_file(h5_file, key=key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 'x' and 'y' from data.h5 files to PNG images."
    )
    parser.add_argument(
        "--root_dir", type=Path, help="Root directory to search for data.h5 files"
    )
    parser.add_argument("--key", type=str, help="Key to extract from the HDF5 file")
    args = parser.parse_args()

    file_dict = {
        "X_mmse": "checkpoint_10",
        "y": "data",
        "x": "data",
        "interpolation": "data",
    }

    walk_and_process(args.root_dir, args.key, file_dict[args.key])

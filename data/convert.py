import argparse
import os

import h5py
import numpy as np
from PIL import Image


def convert_image_to_h5(
    input_path, output_path=None, dataset_name="x", keep_gray_channel=False
):
    """Converts an image file to an HDF5 container with PyTorch-friendly [C, H, W] shapes."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base_name, _ = os.path.splitext(input_path)
        output_path = f"{base_name}.h5"

    try:
        with Image.open(input_path) as img:
            img_format = img.format if img.format else "Unknown"

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            img_mode = img.mode
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            if img_mode == "RGB":
                # image loaded as (H, W, 3) and transposed to (3, H, W)
                img_array = np.transpose(img_array, (2, 0, 1))

            elif img_mode == "L":
                # grayscale as (H, W) expanded to (1, H, W) if --keep-gray-channel is set
                if keep_gray_channel:
                    img_array = np.expand_dims(img_array, axis=0)

        with h5py.File(output_path, "w") as h5f:
            dataset = h5f.create_dataset(
                name=dataset_name,
                data=img_array,
                compression="gzip",
                compression_opts=4,
            )

            # metadata
            dataset.attrs["original_filename"] = os.path.basename(input_path)
            dataset.attrs["original_format"] = img_format
            dataset.attrs["color_mode"] = img_mode
            dataset.attrs["is_chw_format"] = True

        print(f"✅ Saved to: '{output_path}'")
        print(
            f"📊 Dataset '{dataset_name}': Shape {img_array.shape}, Dtype {img_array.dtype}"
        )

    except Exception as e:
        print(f"❌ Failed to convert {input_path}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an image to HDF5 with [C, H, W] ordering."
    )
    parser.add_argument("input", help="Path to the input image file.")
    parser.add_argument(
        "-o", "--output", help="Path to the output .h5 file.", default=None
    )
    parser.add_argument(
        "-d", "--dataset", help="Name of the dataset inside the H5 file.", default="x"
    )

    parser.add_argument(
        "--keep-gray-channel",
        action="store_true",
        help="If the image is grayscale, enforce a [1, H, W] shape instead of [H, W].",
    )

    args = parser.parse_args()

    convert_image_to_h5(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset,
        keep_gray_channel=args.keep_gray_channel,
    )

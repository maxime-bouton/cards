import glob
import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from mcmc.denoisers.denoiser_loader import (
    load_pretrained_ddfb,
    load_pretrained_dncnn,
    load_pretrained_drunet,
)


def load_image_patches(img_folder, num_patches, patch_size):
    """
    Load images from a folder and extract random patches using PyTorch tools.

    Parameters
    ----------
    img_folder : str
        Path to the folder containing images
    num_patches : int
        Number of patches to extract
    patch_size : int
        Size of each patch (assuming square patches)

    Returns
    -------
    list
        List of extracted patches as tensors
    """
    print(f"Loading images and extracting {num_patches} patches...")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(img_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(img_folder, ext.upper())))

    if not image_paths:
        raise ValueError(f"No images found in {img_folder}")

    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.expand(3, -1, -1)),
        ]
    )

    clean_patches = []

    while len(clean_patches) < num_patches:
        img_path = random.choice(image_paths)
        try:
            img = read_image(img_path, mode=ImageReadMode.RGB)
            img = transform(img)
            clean_patches += extract_random_patches(img, 1, patch_size)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue

    return clean_patches[:num_patches]


def load_denoisers():
    """Load all pretrained denoising models.

    Returns
    -------
    dict
        Dictionary containing model information with keys for each denoiser.
    """
    ddfb4 = load_pretrained_ddfb(n_layers=4, n_features=64)
    ddfb19 = load_pretrained_ddfb(n_layers=19, n_features=64)
    ddfb20 = load_pretrained_ddfb(n_layers=20, n_features=64)
    dncnn = load_pretrained_dncnn()
    drunet = load_pretrained_drunet()

    return {
        "DDFB-4": {"model": ddfb4.to(DEVICE), "requires_sigma": True},
        "DDFB-19": {"model": ddfb19.to(DEVICE), "requires_sigma": True},
        "DDFB-20": {"model": ddfb20.to(DEVICE), "requires_sigma": True},
        "DnCNN": {"model": dncnn.to(DEVICE), "requires_sigma": False},
        "DRUNet": {"model": drunet.to(DEVICE), "requires_sigma": True},
    }


def get_model_size(model):
    """Calculate model size in MB.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    float
        Model size in megabytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


def get_model_params(model):
    """Count total trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_sigma_for_snr(clean, target_snr_db):
    """Compute noise standard deviation for target SNR.

    Parameters
    ----------
    clean : torch.Tensor
        Clean image tensor.
    target_snr_db : float
        Target signal-to-noise ratio in dB.

    Returns
    -------
    float
        Noise standard deviation.
    """
    signal_power = clean.pow(2).mean(dim=(-3, -2, -1))
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    return noise_power**0.5


def extract_random_patches(image, num_patches, patch_size):
    """Extract random patches from an image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W).
    num_patches : int
        Number of patches to extract.
    patch_size : int
        Size of square patches.

    Returns
    -------
    list
        List of patch tensors.
    """
    patches = []
    C, H, W = image.shape
    for _ in range(num_patches):
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)
        patch = image[:, top : top + patch_size, left : left + patch_size]
        patches.append(patch)
    return patches


def compute_snr(pred, target):
    """Compute Signal-to-Noise Ratio.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor.
    target : torch.Tensor
        Ground truth tensor.

    Returns
    -------
    float
        SNR in dB.
    """
    noise = pred - target
    target_pow = target.pow(2).mean(dim=(-3, -2, -1))
    noise_pow = noise.pow(2).mean(dim=(-3, -2, -1))
    return 10 * torch.log10(target_pow / noise_pow)


def create_comprehensive_plots(df, output_folder):
    """Create comprehensive visualization plots.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    output_folder : str
        Output directory path.
    """
    plt.style.use("seaborn-v0_8")

    base_colors = ["#2E8B57", "#4682B4", "#DAA520", "#DC143C", "#8A2BE2", "#FF8C00"]

    mean_snr = df.groupby("model")["output_snr"].mean().sort_values()
    model_order = mean_snr.index.tolist()

    model_to_color = {
        model: base_colors[i % len(base_colors)] for i, model in enumerate(model_order)
    }
    model_palette = [model_to_color[model] for model in model_order]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Denoising Model Benchmark Results", fontsize=16, fontweight="bold")

    metrics = ["output_snr", "output_psnr", "output_ssim"]
    metric_labels = ["SNR (dB)", "PSNR (dB)", "SSIM"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        sns.boxplot(
            data=df,
            x="model",
            hue="model",
            y=metric,
            order=model_order,
            palette=model_to_color,
            ax=axes[0, i],
            legend=False,
        )
        axes[0, i].set_title(f"{label} Distribution", fontweight="bold")
        axes[0, i].set_ylabel(label)
        axes[0, i].set_xlabel("Model")
        axes[0, i].tick_params(axis="x", rotation=45)

    model_summary = (
        df.groupby("model")
        .agg(
            {
                "output_snr": "mean",
                "execution_time_ms": "mean",
                "model_size_mb": "first",
                "model_params_m": "first",
            }
        )
        .reindex(model_order)
        .reset_index()
    )

    bar_metrics = ["execution_time_ms", "model_size_mb", "model_params_m"]
    bar_titles = ["Average Execution Time", "Model Size", "Model Parameters"]
    y_labels = ["Time (ms)", "Size (MB)", "Parameters (M)"]

    for i in range(3):
        axes[1, i].bar(
            model_summary["model"],
            model_summary[bar_metrics[i]],
            color=model_palette,
        )
        axes[1, i].set_title(bar_titles[i], fontweight="bold")
        axes[1, i].set_ylabel(y_labels[i])
        axes[1, i].set_xlabel("Model")
        axes[1, i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "comprehensive_benchmark.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_performance_efficiency_plot(df, output_folder):
    """Create performance vs efficiency scatter plots.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    output_folder : str
        Output directory path.
    """
    model_summary = (
        df.groupby("model")
        .agg(
            {
                "output_psnr": "mean",
                "output_ssim": "mean",
                "execution_time_ms": "mean",
                "model_size_mb": "first",
                "model_params_m": "first",
            }
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Performance vs Efficiency Trade-offs", fontsize=16, fontweight="bold")

    axes[0].scatter(
        model_summary["execution_time_ms"],
        model_summary["output_psnr"],
        s=model_summary["model_size_mb"] * 20,
        alpha=0.7,
        c=range(len(model_summary)),
        cmap="viridis",
    )
    axes[0].set_xlabel("Execution Time (ms)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs Speed (bubble size = model size)")

    for i, model in enumerate(model_summary["model"]):
        axes[0].annotate(
            model,
            (
                model_summary.iloc[i]["execution_time_ms"],
                model_summary.iloc[i]["output_psnr"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    axes[1].scatter(
        model_summary["model_params_m"],
        model_summary["output_ssim"],
        s=model_summary["execution_time_ms"] * 2,
        alpha=0.7,
        c=range(len(model_summary)),
        cmap="viridis",
    )
    axes[1].set_xlabel("Parameters (k)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs Model Size (bubble size = exec time)")

    for i, model in enumerate(model_summary["model"]):
        axes[1].annotate(
            model,
            (
                model_summary.iloc[i]["model_params_m"],
                model_summary.iloc[i]["output_ssim"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "performance_efficiency.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_summary_table(df, output_folder):
    """Create and save summary statistics table.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    output_folder : str
        Output directory path.
    """
    summary_stats = (
        df.groupby("model")
        .agg(
            {
                "sigma": ["mean", "std"],
                "input_snr": ["mean", "std"],
                "input_psnr": ["mean", "std"],
                "input_ssim": ["mean", "std"],
                "output_snr": ["mean", "std"],
                "output_psnr": ["mean", "std"],
                "output_ssim": ["mean", "std"],
                "execution_time_ms": ["mean", "std"],
                "model_size_mb": "first",
                "model_params_m": "first",
            }
        )
        .round(4)
    )

    mean_snr = df.groupby("model")["output_snr"].mean().sort_values()
    model_order = mean_snr.index.tolist()
    summary_stats = summary_stats.reindex(model_order)

    summary_stats.columns = ["_".join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()

    summary_stats_clean = pd.DataFrame(
        {
            "Model": summary_stats["model"],
            "In SNR (dB)": summary_stats["input_snr_mean"].apply(lambda x: f"{x:.2f}")
            + " ± "
            + summary_stats["input_snr_std"].apply(lambda x: f"{x:.2f}"),
            "In PSNR (dB)": summary_stats["input_psnr_mean"].apply(lambda x: f"{x:.2f}")
            + " ± "
            + summary_stats["input_psnr_std"].apply(lambda x: f"{x:.2f}"),
            "In SSIM": summary_stats["input_ssim_mean"].apply(lambda x: f"{x:.3f}")
            + " ± "
            + summary_stats["input_ssim_std"].apply(lambda x: f"{x:.3f}"),
            "Out SNR (dB)": summary_stats["output_snr_mean"].apply(lambda x: f"{x:.2f}")
            + " ± "
            + summary_stats["output_snr_std"].apply(lambda x: f"{x:.2f}"),
            "Out PSNR (dB)": summary_stats["output_psnr_mean"].apply(
                lambda x: f"{x:.2f}"
            )
            + " ± "
            + summary_stats["output_psnr_std"].apply(lambda x: f"{x:.2f}"),
            "Out SSIM": summary_stats["output_ssim_mean"].apply(lambda x: f"{x:.3f}")
            + " ± "
            + summary_stats["output_ssim_std"].apply(lambda x: f"{x:.3f}"),
            "Time (ms)": summary_stats["execution_time_ms_mean"].apply(
                lambda x: f"{x:.2f}"
            ),
            "Size (MB)": summary_stats["model_size_mb_first"].apply(
                lambda x: f"{x:.2f}"
            ),
            "Params (k)": summary_stats["model_params_m_first"].apply(
                lambda x: f"{x:.2f}"
            ),
        }
    )

    summary_stats_clean.to_csv(
        os.path.join(output_folder, "summary_statistics.csv"), index=False
    )
    print("\nSummary Statistics:")
    print(summary_stats_clean.to_string(index=False))


def main():
    """Main benchmark execution function."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    clean_patches = load_image_patches(IMG_FOLDER, NUM_PATCHES, PATCH_SIZE)
    clean_batch = torch.stack(clean_patches).to(DEVICE)

    sigma_batch = compute_sigma_for_snr(clean_batch, TARGET_SNR_DB).view(-1, 1, 1, 1)
    noisy_batch = clean_batch + sigma_batch * torch.randn_like(clean_batch)

    denoisers = load_denoisers()
    records = []

    print("Running batch denoising with performance profiling...")

    psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3)).to(
        DEVICE
    )
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(DEVICE)

    for model_name, entry in denoisers.items():
        model = entry["model"]
        requires_sigma = entry.get("requires_sigma", False)

        model_size_mb = get_model_size(model)
        model_params = get_model_params(model) / 1e3

        with torch.no_grad():
            inputs = noisy_batch.to(DEVICE)
            if requires_sigma:
                sigmas_device = sigma_batch.to(DEVICE)

            for _ in range(2):
                if requires_sigma:
                    _ = model(inputs, sigmas_device)
                else:
                    _ = model(inputs)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if requires_sigma:
                denoised_batch = model(inputs, sigmas_device)
            else:
                denoised_batch = model(inputs)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000

        in_snr = compute_snr(noisy_batch, clean_batch)
        in_psnr = psnr(noisy_batch, clean_batch)
        in_ssim = ssim(noisy_batch, clean_batch)

        out_snr = compute_snr(denoised_batch, clean_batch)
        out_psnr = psnr(denoised_batch, clean_batch)
        out_ssim = ssim(denoised_batch, clean_batch)

        for i in range(len(clean_batch)):
            if (
                torch.isnan(in_snr[i])
                or torch.isnan(in_psnr[i])
                or torch.isnan(in_ssim[i])
                or torch.isnan(out_snr[i])
                or torch.isnan(out_psnr[i])
                or torch.isnan(out_ssim[i])
            ):
                continue
            records.append(
                {
                    "model": model_name,
                    "patch_id": i,
                    "sigma": sigma_batch[i].item(),
                    "input_snr": in_snr[i].item(),
                    "input_psnr": in_psnr[i].item(),
                    "input_ssim": in_ssim[i].item(),
                    "output_snr": out_snr[i].item(),
                    "output_psnr": out_psnr[i].item(),
                    "output_ssim": out_ssim[i].item(),
                    "execution_time_ms": execution_time_ms,
                    "model_size_mb": model_size_mb,
                    "model_params_m": model_params,
                }
            )

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(OUTPUT_FOLDER, "detailed_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("Generating comprehensive visualizations...")
    create_comprehensive_plots(df, OUTPUT_FOLDER)
    create_performance_efficiency_plot(df, OUTPUT_FOLDER)
    create_summary_table(df, OUTPUT_FOLDER)

    print(f"Enhanced benchmark complete. Results saved to '{OUTPUT_FOLDER}'")
    print("Key outputs:")
    print("  - Detailed metrics: detailed_metrics.csv")
    print("  - Summary statistics: summary_statistics.csv")
    print("  - Comprehensive plots: comprehensive_benchmark.png")
    print("  - Efficiency analysis: performance_efficiency.png")


if __name__ == "__main__":
    IMG_FOLDER = "/data3/mbouton/BSDS500/BSDS500/data/images/test"
    IMG_FOLDER = "/data3/mbouton/DIV2K_valid_HR"
    OUTPUT_FOLDER = "produced_data/benchmark"
    PATCH_SIZE = 50
    NUM_PATCHES = 1000
    TARGET_SNR_DB = 20.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()

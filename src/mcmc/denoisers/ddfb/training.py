import argparse
import logging
import os
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def calculate_snr(clean_images, noisy_or_denoised_images):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between clean and noisy/denoised images.

    Parameters
    ----------
    clean_images : torch.Tensor
        Clean reference images
    noisy_or_denoised_images : torch.Tensor
        Noisy or denoised images to compare against clean images

    Returns
    -------
    snr : torch.Tensor
        Mean SNR value (in dB) across the batch, excluding images with zero noise power
    """
    signal_power = torch.sum(clean_images**2, dim=[1, 2, 3])
    noise = clean_images - noisy_or_denoised_images
    noise_power = torch.sum(noise**2, dim=[1, 2, 3])
    valid_mask = noise_power > 0

    if valid_mask.sum() > 0:
        ratio = signal_power[valid_mask] / noise_power[valid_mask]
        snr_values = 10 * torch.log10(ratio[ratio != 0])
        return snr_values.mean()
    else:
        return torch.tensor(float("nan"), device=clean_images.device)


def setup_logger(log_dir, log_filename):
    """
    Set up logger that writes to both console and file.

    Parameters
    ----------
    log_dir : str
        Directory path for saving log files
    log_filename : str
        Name of the log file

    Returns
    -------
    logger : logging.Logger
        Configured logger object
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("DDFB_trainer")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class DenoisingDataset(Dataset):
    """
    Dataset class for image denoising, extracts random patches from input images.

    Parameters
    ----------
    image_dir : str
        Directory containing the training images
    patch_size : int
        Size of image patches to extract (patches are square)
    transform : callable, optional
        Optional transform to be applied to the patches
    patches_per_image : int
        Number of patches to extract from each image
    grayscale : bool
        If True, convert images to grayscale
    """

    def __init__(
        self,
        image_dir,
        patch_size=50,
        transform=None,
        patches_per_image=1,
        grayscale=False,
    ):
        self.image_paths = []
        self.patch_size = patch_size
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.conversion = "L" if grayscale else "RGB"

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img_path = self.image_paths[img_idx]

        with Image.open(img_path).convert(self.conversion) as img:
            if img.width < self.patch_size or img.height < self.patch_size:
                img = img.resize(
                    (
                        max(self.patch_size, img.width),
                        max(self.patch_size, img.height),
                    )
                )

            left = random.randint(0, img.width - self.patch_size)
            top = random.randint(0, img.height - self.patch_size)
            patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))

            if self.transform:
                patch = self.transform(patch)

            clean = patch

            return clean


def add_noise(
    clean_images,
    sigma=None,
    min_sigma=0.0,
    max_sigma=0.1,
    same_noise_level=True,
    isnr=None,
):
    """
    Add Gaussian noise to images.

    Parameters
    ----------
    clean_images : torch.Tensor
        Clean images batch
    sigma : float, optional
        Noise standard deviation (if None, will be randomly sampled)
    min_sigma : float
        Minimum noise level when sampling randomly
    max_sigma : float
        Maximum noise level when sampling randomly
    same_noise_level : bool
        If True, all images in the batch will have the same noise level

    Returns
    -------
    noisy_images : torch.Tensor
        Noisy images batch
    sigmas : torch.Tensor
        Noise levels used for each image
    """
    batch_size = clean_images.size(0)

    if sigma is not None:
        sigmas = torch.full((batch_size, 1, 1, 1), sigma, device=clean_images.device)
    elif same_noise_level:
        sigma_val = random.uniform(min_sigma, max_sigma)
        sigmas = torch.full(
            (batch_size, 1, 1, 1), sigma_val, device=clean_images.device
        )
    elif isnr is not None:
        sigmas = compute_sigma_for_snr(clean_images, target_snr_db=isnr)
        sigmas = sigmas.view(batch_size, 1, 1, 1).to(device=clean_images.device)
    else:
        sigmas = (
            torch.rand(batch_size, 1, 1, 1, device=clean_images.device)
            * (max_sigma - min_sigma)
            + min_sigma
        )

    noise = torch.randn_like(clean_images) * sigmas
    noisy_images = clean_images + noise
    return noisy_images, sigmas


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


def evaluate(
    model, val_loader, criterion, device, same_noise_level, sigma=None, isnr=None
):
    """
    Evaluate the model on validation data.

    Parameters
    ----------
    model : torch.nn.Module
        The denoising model
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to run evaluation on
    same_noise_level : bool
        If True, all images in batch have same noise level
    sigma : float, optional
        Fixed noise level for validation

    Returns
    -------
    avg_loss : float
        Average validation loss
    avg_input_snr : float
        Average input SNR on validation set
    avg_output_snr : float
        Average output SNR on validation set
    """
    model.eval()
    total_loss = 0
    total_input_snr = 0
    total_output_snr = 0
    count = 0

    with torch.no_grad():
        for clean_images in val_loader:
            clean_images = clean_images.to(device)

            noisy_images, sigmas = add_noise(
                clean_images, sigma=sigma, same_noise_level=same_noise_level, isnr=isnr
            )

            denoised_images = model(noisy_images, sigmas)

            loss = criterion(denoised_images, clean_images) / 2

            input_snr = calculate_snr(clean_images, noisy_images)
            output_snr = calculate_snr(clean_images, denoised_images)

            total_loss += loss.item()
            if not torch.isnan(input_snr):
                total_input_snr += input_snr.item()
            if not torch.isnan(output_snr):
                total_output_snr += output_snr.item()
            count += 1

    avg_loss = total_loss / count
    avg_input_snr = total_input_snr / count
    avg_output_snr = total_output_snr / count

    return avg_loss, avg_input_snr, avg_output_snr


def train(
    args,
    model,
    str_model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    logger,
):
    """
    Train the DDFB model.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    model : torch.nn.Module
        The DDFB model
    str_model: str
        String representation of the model to save weights
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer for training
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to train on
    logger : logging.Logger
        Logger for training information

    Returns
    -------
    model : torch.nn.Module
        Trained model
    """
    best_val_loss = float("inf")
    best_output_snr = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_input_snr_sum = 0
        train_output_snr_sum = 0
        batch_count = 0
        valid_snr_count = 0
        start_time = time.time()

        for handler in logger.handlers:
            handler.stream.write("\n")
            handler.flush()
        logger.info(f"Epoch: {epoch + 1}/{args.epochs}")

        for batch_idx, clean_images in enumerate(train_loader):
            clean_images = clean_images.to(device)

            noisy_images, sigmas = add_noise(
                clean_images,
                sigma=args.fixed_sigma if args.fixed_sigma is not None else None,
                same_noise_level=args.same_noise_level,
                isnr=args.isnr,
            )

            optimizer.zero_grad()
            denoised_images = model(noisy_images, sigmas)

            loss = criterion(denoised_images, clean_images) / 2

            input_snr = calculate_snr(clean_images, noisy_images)
            output_snr = calculate_snr(clean_images, denoised_images)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if not torch.isnan(input_snr):
                train_input_snr_sum += input_snr.item()
                train_output_snr_sum += output_snr.item()
                valid_snr_count += 1
            batch_count += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_batch_loss = train_loss / batch_count
                avg_input_snr = train_input_snr_sum / max(1, valid_snr_count)
                avg_output_snr = train_output_snr_sum / max(1, valid_snr_count)

                train_loss = 0
                train_input_snr_sum = 0
                train_output_snr_sum = 0
                batch_count = 0
                valid_snr_count = 0

                logger.info(
                    f"\tBatch: {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_batch_loss:.3e} | "
                    f"Input SNR: {avg_input_snr:.2f} dB | "
                    f"Output SNR: {avg_output_snr:.2f} dB | "
                    f"Difference SNR: {avg_output_snr - avg_input_snr:.2f} dB | "
                    f"Noise levels: {sigmas.mean().item():.4f} | "
                    f"Noise range: {sigmas.min().item():.4f}-{sigmas.max().item():.4f}"
                )

        val_loss, val_input_snr, val_output_snr = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.same_noise_level,
            args.fixed_sigma,
            args.isnr,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        logger.info(
            f"Val Loss: {val_loss:.3e} | "
            f"Val Input SNR: {val_input_snr:.2f} dB | "
            f"Val Output SNR: {val_output_snr:.2f} dB | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_output_snr = val_output_snr

            checkpoint_path = os.path.join(args.checkpoint_dir, f"{str_model}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"{str_model}_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

    logger.info(
        f"Training completed. Best validation loss: {best_val_loss:.6f}, "
        f"Output SNR: {best_output_snr:.2f} dB"
    )
    return model


def initialize_ddfb_model(model):
    """
    Apply improved initialization to DDFB model's convolutional layers.

    Parameters
    ----------
    model : torch.nn.Module
        The DDFB model to initialize

    Returns
    -------
    model : torch.nn.Module
        Initialized model
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


def main():
    """
    Main function for DDFB model training.
    """
    parser = argparse.ArgumentParser(
        description="DDFB CNN Training for Image Denoising"
    )

    parser.add_argument(
        "--gray", action="store_true", help="Training for grayscale images"
    )

    parser.add_argument(
        "--n_layers", type=int, default=5, help="Number of layers in the DDFB"
    )

    parser.add_argument(
        "--n_features", type=int, default=64, help="Number of features in the DDFB"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the image directory",
        default="/path/to/training_dataset/",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        help="Path to the validation image directory",
        default="",
    )

    parser.add_argument(
        "--patch_size", type=int, default=50, help="Size of image patches"
    )
    parser.add_argument(
        "--patches_per_image", type=int, default=1, help="Number of patches per image"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--isnr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--same_noise_level",
        action="store_true",
        help="Use same noise level for all images in a batch",
    )
    parser.add_argument(
        "--fixed_sigma",
        type=float,
        default=None,
        help="Fixed noise level (if not specified, random in [0, 0.1])",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split ratio"
    )

    TRAINING_DIR = "training/"

    parser.add_argument(
        "--log_dir",
        type=str,
        default=TRAINING_DIR,
        help="Directory for saving logs",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=TRAINING_DIR,
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log interval (batches)"
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save interval (epochs)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger = setup_logger(args.log_dir, "training.log")

    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    full_dataset = DenoisingDataset(
        image_dir=args.data_dir,
        patch_size=args.patch_size,
        transform=transform,
        patches_per_image=args.patches_per_image,
        grayscale=args.gray,
    )

    if args.val_dir and os.path.exists(args.val_dir):
        train_dataset = full_dataset
        val_dataset = DenoisingDataset(
            image_dir=args.val_dir,
            patch_size=args.patch_size,
            transform=transform,
            patches_per_image=args.patches_per_image,
            grayscale=args.gray,
        )
        train_size = len(train_dataset)
        val_size = len(val_dataset)
    else:
        dataset_size = len(full_dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(
        f"Dataset loaded. Training samples: {train_size}, Validation samples: {val_size}"
    )

    from mcmc.denoisers.ddfb.network_ddfb import DDFB

    nch = 1 if args.gray else 3
    nla = args.n_layers
    nfe = args.n_features

    str_model = f"ddfb_nch{nch}_nla{nla}_nfe{nfe}"

    model = DDFB(C=nch, n_layers=nla, n_features=nfe)
    model = initialize_ddfb_model(model)
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info("Applied improved Kaiming initialization to DDFB model")
    model = model.to(device)

    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    logger.info("Starting training...")
    train(
        args,
        model,
        str_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        logger,
    )

    logger.info("Training finished!")


if __name__ == "__main__":
    main()

    # python src/mcmc/denoisers/ddfb/training.py --data_dir /data3/mbouton/imagenet/test/ --val_dir /data3/mbouton/DIV2K_valid_HR/ --n_layers 4 --n_features 64

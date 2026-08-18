r"""Utility functions to create and format logger outputs."""

# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais - **A
# Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse
# Problems**, [arxiv preprint](http://arxiv.org/abs/), October 2025.

# TODO: documentation

import logging
import re
import sys
from pathlib import Path


def get_null_logger(name: str = "null_logger") -> logging.Logger:
    r"""Create a null logger."""
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


class ColoredFormatter(logging.Formatter):
    r"""Custom formatter that adds colors based on log level."""

    RESET = "\033[0m"
    COLORS = {
        logging.CRITICAL: "\033[1;31m",
        logging.ERROR: "\033[31m",
        logging.WARNING: "\033[33m",
        logging.INFO: RESET,
        logging.DEBUG: "\033[36m",
    }

    def format(self, record):
        orig_msg = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{orig_msg}{self.RESET}"


class PlainFileFormatter(logging.Formatter):
    r"""Formatter that strips ANSI escape codes for clean file logging."""

    # Regex to match standard ANSI color/style escape sequences
    ANSI_RE = re.compile(r"\033\[[0-9;]*m")

    def format(self, record):
        orig_msg = super().format(record)
        return self.ANSI_RE.sub("", orig_msg)


def build_logger(
    rank,
    path: Path | None = None,
    level=logging.INFO,
    print_rank: int | None = 0,
):
    r"""
    Build a logger that writes to both a rank-specific file and the console (rank 0 only),
    with colored output based on log level.

    Parameters
    ----------
    rank : int
        MPI rank of the current process
    path : Path, optional
        Save the logs to the specified path
    level : int, optional
        Logging level, by default logging.INFO
    print_rank : int, optional
        If specified, only the logger for this rank will print to the console.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []

    if path is not None:
        file_formatter = PlainFileFormatter(
            "%(asctime)s - %(levelname)-8s - %(message)s"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if rank is not None and rank == print_rank:
        console_handler = logging.StreamHandler(sys.stdout)
        colored_formatter = ColoredFormatter(
            f"%(asctime)s - Rank {rank} - %(levelname)-8s - %(message)s"
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    return logger


class ProgressBar:
    r"""A standalone MPI progress bar that coordinates perfectly with standard loggers.

    Uses the 'Lift and Drop' pattern to erase itself before external logs
    are printed, keeping a clean pinned-bottom UI without intercepting the logger.
    """

    def __init__(self, total: int, desc: str = "Sampling", bar_len: int = 45) -> None:
        self.total = total
        self.desc = desc
        self.bar_len = bar_len
        self._is_visible = False

    def clear(self) -> None:
        r"""Erase the progress bar from the terminal."""
        if self._is_visible:
            # \033[1A : Move cursor UP one line
            # \033[2K : Erase the entire line
            sys.stdout.write("\033[1A\033[2K")
            self._is_visible = False

    def _format_time(self, seconds: float) -> str:
        r"""Format seconds into a readable time string."""
        if seconds <= 0:
            return "00:00"

        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)

        if h > 0:
            return f"{h:02d}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"

    def update(self, current: int, time_per_step: float | None = None) -> None:
        r"""Draw the progress bar on a new line."""
        percent = min(current / self.total, 1.0)
        filled = int(self.bar_len * percent)
        bar = "█" * filled + " " * (self.bar_len - filled)

        eta_str = ""
        if time_per_step is not None:
            remaining_steps = self.total - current
            eta_seconds = remaining_steps * time_per_step
            eta_str = f" | ETA: {self._format_time(eta_seconds)}"

        pbar_line = (
            f"{self.desc} |{bar}| {current}/{self.total} [{percent:>4.0%}]{eta_str}\n"
        )

        sys.stdout.write(pbar_line)
        sys.stdout.flush()
        self._is_visible = True

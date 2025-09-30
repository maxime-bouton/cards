import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors based on log level.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"

    def format(self, record):
        orig_msg = super().format(record)
        if record.levelno == logging.ERROR:
            return f"{ColoredFormatter.RED}{orig_msg}{ColoredFormatter.RESET}"
        elif record.levelno == logging.WARNING:
            return f"{ColoredFormatter.YELLOW}{orig_msg}{ColoredFormatter.RESET}"
        else:
            return orig_msg


def build_logger(
    rank,
    path: Path | None = None,
    level=logging.INFO,
    print_rank: int | None = 0,
):
    """
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
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if rank is not None and rank == print_rank:
        console_handler = logging.StreamHandler(sys.stdout)
        colored_formatter = ColoredFormatter(
            f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    return logger

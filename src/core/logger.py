import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from src.configs.paths import LOG_DIR_PATH

console = Console()


def get_logger(
    logger_name: str,
    log_dir: str | Path = LOG_DIR_PATH,
) -> logging.Logger:

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{logger_name}.log"

    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )

    file_handler = logging.FileHandler(log_file)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)

    return logger

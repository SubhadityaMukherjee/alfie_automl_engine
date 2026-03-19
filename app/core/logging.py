import logging
import os
from logging.handlers import RotatingFileHandler

_TEN_MB = 10 * 1024 * 1024


def configure_service_logging(service_name: str) -> None:
    """Set up a RotatingFileHandler for the given service name.

    Reads:
      - ALFIE_LOG_DIR   — directory for log files (default: ./logs)
      - ALFIE_LOG_LEVEL — logging level name (default: INFO)

    Creates logs/<service_name>.log with maxBytes=10 MB, backupCount=5.
    Attaches the handler to the root logger so all module loggers inherit it.
    Safe to call multiple times; duplicate handlers are not added.
    """
    log_dir = os.getenv("ALFIE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    log_level_name = os.getenv("ALFIE_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{service_name}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if called more than once
    existing_files = {
        h.baseFilename
        for h in root_logger.handlers
        if isinstance(h, RotatingFileHandler)
    }
    if log_file in existing_files:
        return

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_TEN_MB,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

import logging
import os
from logging.handlers import RotatingFileHandler


def _configure_logging_once() -> None:
    """Configure root logging to write to a single log file.

    Respects env:
      - ALFIE_LOG_FILE: path to the log file (default: ./alfie_app.log)
      - ALFIE_LOG_LEVEL: logging level name (default: INFO)
    """
    if getattr(_configure_logging_once, "configured", False):
        return

    log_file = os.getenv("ALFIE_LOG_FILE", os.path.join(os.getcwd(), "alfie_app.log"))
    log_level_name = os.getenv("ALFIE_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_name, logging.DEBUG)
    except Exception:
        log_level = logging.DEBUG

    # Ensure parent directory exists
    try:
        parent_dir = os.path.dirname(log_file) or "."
        os.makedirs(parent_dir, exist_ok=True)
    except Exception:
        # Fallback to current working directory if creation fails
        log_file = os.path.join(os.getcwd(), "alfie_app.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any pre-existing handlers to avoid duplicate logs
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    _configure_logging_once.configured = True  # type: ignore[attr-defined]


# Configure logging on import
_configure_logging_once()

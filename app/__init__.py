import logging
import os

# Fallback logging configuration — console only, WARNING level.
# Each service calls app.core.logging.configure_service_logging() inside its
# lifespan to add a rotating file handler for its own log file.
_log_level_name = os.getenv("ALFIE_LOG_LEVEL", "WARNING").upper()
_log_level = getattr(logging, _log_level_name, logging.WARNING)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

"""Logging configuration module."""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
        log_level: str = "INFO",
        log_dir: Optional[Path] = None,
        service_name: str = "llm_homology_api"
) -> logging.Logger:
    """
    Set up logging configuration with both stdout and timestamped file output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files. If None, uses ./logs/
        service_name: Name of the service for log filename

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{service_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_filename,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Log the startup information
    logger.info(f"Logging initialized. Log file: {log_filename}")
    logger.info(f"Console log level: {log_level.upper()}")
    logger.info(f"File log level: DEBUG")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If setup_logging hasn't been called,
    this will return a basic logger.

    Args:
        name: Logger name. If None, returns the root service logger.

    Returns:
        Logger instance
    """
    if name is None:
        name = "llm_homology_api"
    return logging.getLogger(name)
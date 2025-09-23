"""Logging configuration module."""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    service_name: str = "llm_homology_api",
    force_flush: bool = True  # New parameter
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

    # Force immediate flushing for console
    console_handler.flush = lambda: console_handler.stream.flush()

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

    # Force immediate flushing for file handler
    original_emit = file_handler.emit
    def flush_emit(record):
        original_emit(record)
        file_handler.flush()
    file_handler.emit = flush_emit

    logger.addHandler(file_handler)

    # Log the startup information
    logger.info(f"Logging initialized. Log file: {log_filename}")
    logger.info(f"Console log level: {log_level.upper()}")
    logger.info(f"File log level: DEBUG")
    logger.info(f"Force flush enabled: {force_flush}")

    # If force_flush is enabled, flush immediately
    if force_flush:
        for handler in logger.handlers:
            handler.flush()

    return logger


def flush_logger(logger_name: Optional[str] = None) -> None:
    """
    Manually flush a specific logger or all loggers.

    Args:
        logger_name: Name of logger to flush. If None, flushes all loggers.
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.flush()
    else:
        # Flush all loggers
        import sys

        # Flush all named loggers
        for name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                handler.flush()

        # Flush root logger
        for handler in logging.root.handlers:
            handler.flush()

        # Flush standard streams
        sys.stdout.flush()
        sys.stderr.flush()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance that works with uvicorn.
    Auto-configures basic logging if none exists.
    """
    if name is None:
        name = "llm_homology_api"

    logger = logging.getLogger(name)

    # If no handlers exist anywhere, set up basic logging
    if not logger.handlers and not logger.parent.handlers and not logging.root.handlers:
        # Set up basic console logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add to root logger so all loggers inherit it
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)

        logger.info("Auto-configured basic logging")

    return logger
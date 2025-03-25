"""
Logging Configuration

This module provides configuration for the application's logging system.
"""

import logging
import logging.config
import os
import sys
from typing import Dict, Optional

# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/agent.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": True
        },
        "ai_agent_framework": {
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
    }
}


def setup_logging(
    config: Optional[Dict] = None,
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_level: Optional[str] = None
) -> None:
    """
    Configure the logging system.
    
    Args:
        config: Optional custom logging configuration
        log_dir: Directory to store log files
        log_level: Default logging level
        console_level: Optional separate logging level for console output
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Start with default config
    log_config = DEFAULT_CONFIG.copy()
    
    # Update with custom config if provided
    if config:
        _deep_merge(log_config, config)
    
    # Update log directory in file handlers
    for handler_name in ["file", "error_file"]:
        if handler_name in log_config["handlers"]:
            filename = os.path.basename(log_config["handlers"][handler_name]["filename"])
            log_config["handlers"][handler_name]["filename"] = os.path.join(log_dir, filename)
    
    # Update log levels
    log_config["loggers"][""]["level"] = log_level
    log_config["loggers"]["ai_agent_framework"]["level"] = log_level
    
    if console_level:
        log_config["handlers"]["console"]["level"] = console_level
    
    # Apply configuration
    logging.config.dictConfig(log_config)
    
    # Add unhandled exception hook
    sys.excepthook = _log_uncaught_exceptions
    
    logging.info(f"Logging system initialized (level: {log_level})")


def _deep_merge(target: Dict, source: Dict) -> None:
    """
    Deep merge source dict into target dict.
    
    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # If both are dicts, recursively merge them
            _deep_merge(target[key], value)
        else:
            # Otherwise, source overwrites target
            target[key] = value


def _log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """
    Log uncaught exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = logging.getLogger("ai_agent_framework")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
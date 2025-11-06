"""
Structured logging setup for WarmStart.
Provides JSON-formatted logs with context tracking and optional emoji stripping.
"""

import logging
import sys
import os
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    event_dict["app"] = "warmstart"
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    if method_name == "debug":
        event_dict["level"] = "DEBUG"
    elif method_name == "info":
        event_dict["level"] = "INFO"
    elif method_name == "warning":
        event_dict["level"] = "WARNING"
    elif method_name == "error":
        event_dict["level"] = "ERROR"
    elif method_name == "critical":
        event_dict["level"] = "CRITICAL"
    return event_dict


def strip_non_ascii_processor(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Optionally strip non-ASCII characters (e.g., emojis) from log output.

    Enabled when environment variable WARMSTART_NO_EMOJI == "1".
    This operates conservatively by removing all non-ASCII characters to avoid
    Windows console encoding issues.
    """
    try:
        if os.environ.get("WARMSTART_NO_EMOJI", "0") == "1":
            for k, v in list(event_dict.items()):
                if isinstance(v, str):
                    event_dict[k] = v.encode("ascii", "ignore").decode("ascii")
    except Exception:
        # Never break logging pipeline
        pass
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'console')
        log_file: Optional file path for log output
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_app_context,
        add_log_level,
        strip_non_ascii_processor,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if log_format == "json":
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent logs.
    
    Example:
        bind_context(run_id="run_123", domain="legal")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove context variables.
    
    Example:
        unbind_context("run_id", "domain")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


class LoggerContext:
    """
    Context manager for temporary logging context.
    
    Example:
        with LoggerContext(run_id="run_123", generation=5):
            logger.info("Processing generation")
    """
    
    def __init__(self, **kwargs: Any):
        self.context = kwargs
    
    def __enter__(self):
        bind_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        unbind_context(*self.context.keys())

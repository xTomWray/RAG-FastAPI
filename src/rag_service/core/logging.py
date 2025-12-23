"""Structured logging configuration for the RAG service.

Provides configurable logging with support for both JSON format
(for production log aggregation) and console format (for development).

Example usage:
    from rag_service.core.logging import configure_logging, get_logger

    configure_logging(json_format=True, log_level="INFO")
    logger = get_logger(__name__)
    logger.info("Processing request", request_id="abc123", user_id=42)
"""

import logging
import sys
from typing import Any, Literal

import structlog
from structlog.types import Processor


def configure_logging(
    log_format: Literal["json", "console"] = "json",
    log_level: str = "INFO",
) -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors for the selected format.
    JSON format produces machine-parseable logs for log aggregators.
    Console format produces colorful, human-readable logs for development.

    Args:
        log_format: Output format - "json" for production, "console" for dev.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Common processors for all formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # JSON format for production
        processors: list[Processor] = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    # Use stdlib LoggerFactory so loggers have a .name attribute
    # (required by add_logger_name processor)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for libraries that use it
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound logger instance with structured logging support.
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to the current logging context.

    These values will be included in all subsequent log messages
    within the current async context (e.g., request handling).

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        bind_context(correlation_id="abc123", user_id=42)
        logger.info("Processing")  # Will include correlation_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Call this at the end of request handling to prevent context
    from leaking to subsequent requests.
    """
    structlog.contextvars.clear_contextvars()

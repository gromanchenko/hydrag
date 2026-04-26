"""Logging helpers for HydRAG."""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a propagation-only HydRAG logger."""
    logger_name = f"hydrag.{name}" if name else "hydrag"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = True
    return logger

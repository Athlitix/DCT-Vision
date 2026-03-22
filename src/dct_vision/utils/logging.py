"""Structured logging configuration for dct-vision."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g. logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

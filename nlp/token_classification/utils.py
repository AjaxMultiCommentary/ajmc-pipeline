"""A few general utilities for token_classification."""

import random
import logging
import numpy as np
import torch


def get_custom_logger(name: str,
                      level: int = logging.INFO,
                      fmt: str = "%(levelname)s - %(name)s -   %(message)s"):
    """Custom logging wraper, called each time a logger is declared in the package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """Sets seed for `random`, `np.random` and `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

"""A few general utilities for `nlp` a machine learning in general."""

import random
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed):
    """Sets seed for `random`, `np.random` and `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: Union[torch.device, str]) -> torch.device:
    """A simple function to create a `torch.device` from string. If `device` is allready, a `torch.device`,
    returns it unchanged."""

    if type(device) == str:
        return torch.device(device)
    elif type(device) == torch.device:
        return device
    else:
        raise





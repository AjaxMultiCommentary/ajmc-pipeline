from typing import Tuple, List
import numpy as np

from ajmc.commons.docstrings import docstrings, docstring_formatter


@docstring_formatter(**docstrings)
def compute_interval_overlap(i1: Tuple[int, int], i2: Tuple[int, int]):
    """Computes the overlap between two interevals defined by their start and their stop included.

    Args:
        i1: {interval}
        i2: {interval}
    Returns:
        int: The length of the overlap.
    """
    return max(min(i1[1], i2[1]) + 1 - max(i1[0], i2[0]), 0)  # Adding 1 as borders are included


@docstring_formatter(**docstrings)
def is_interval_within_interval(contained: Tuple[int, int], container: Tuple[int, int]) -> bool:
    """Checks if the `contained` interval is included in the `container` interval.

    Args:
        container: {interval}
        contained: {interval}
    """
    return contained[0] >= container[0] and contained[1] <= container[1]


@docstring_formatter(**docstrings)
def are_intervals_within_intervals(contained: List[Tuple[int, int]], container: List[Tuple[int, int]]) -> bool:
    """Applies `is_interval_within_interval` on a list of intervals, making sure that all the contained intervals
    are contained in one of the container intervals."""

    # todo 👁️ deal with overlapping TCs (e.g. chapter spanning over several pages but not entirely included in them.
    return all(
        [
            any([is_interval_within_interval(contained_i, container_i) for container_i in container])
            for contained_i in contained
        ]
    )


def safe_divide(dividend, divisor):
    """Simple division which return `np.nan` if `divisor` equals zero."""
    return dividend / divisor if divisor != 0 else np.nan

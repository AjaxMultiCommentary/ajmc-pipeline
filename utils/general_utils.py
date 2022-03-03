"""General utilities"""

import timeit
from typing import List, Tuple

import numpy as np


def lazy_property(func):
    """Decorator. Makes property computation lazy."""

    def inner(self, *args, **kwargs):
        private_attr = "_" + func.__name__
        if not hasattr(self, private_attr):
            setattr(self, private_attr, func(self, *args, **kwargs))
        return getattr(self, private_attr)

    return property(inner)


def timer(iterations: int = 3, number: int = 1_000):
    """Decorator with arguments. Computes the execution time of a function."""

    def timer_decorator(func):
        def inner(*args, **kwargs):
            statement = lambda: func(*args, **kwargs)
            for i in range(iterations):
                print(
                    f"""Func: {func.__name__}, Iteration {i}. Elapsed time for {number} executions :   {timeit.timeit(statement, number=number)}""")
            return statement()

        return inner

    return timer_decorator


def safe_divide(dividend, divisor):
    """Simple division which return `np.nan` if `divisor` equals zero."""
    return dividend / divisor if divisor != 0 else np.nan


RectangleType = List[Tuple[int, int]]
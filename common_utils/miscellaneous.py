"""Miscellaneous helpers and utilities."""

import timeit
from typing import List, Tuple, Iterable, Generator
import numpy as np


RectangleType = List[Tuple[int, int]]

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



def recursive_iterator(iterable: Iterable, iterable_types: Tuple[Iterable] = (list, tuple)) -> Generator:
    """Iterates recursively through an iterable potentially containing iterables of `iterable_types`."""
    for x in iterable:
        if isinstance(x, iterable_types):
            for y in recursive_iterator(x):
                yield y
        else:
            yield x


def get_unique_elements(iterable: Iterable, iterable_types: Tuple[Iterable] = (list, tuple)) -> List[str]:
    """Get the list of elements from any potentially recursive iterable."""
    return list(set([l for l in recursive_iterator(iterable, iterable_types)]))




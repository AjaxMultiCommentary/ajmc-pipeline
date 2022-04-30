"""Miscellaneous helpers and utilities."""

import json
import logging
import timeit
from typing import List, Tuple, Iterable, Generator
import numpy as np
from jsonschema import Draft6Validator

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


def validate_json_schema(schema_path: str = 'data/page.schema.json'):
    """Validates a json schema against `Draft6Validator`"""

    with open(schema_path, "r") as file:
        schema = json.loads(file.read())

    Draft6Validator.check_schema(schema)


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
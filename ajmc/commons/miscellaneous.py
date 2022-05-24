"""Miscellaneous helpers and utilities."""

import json
import logging
import timeit
from typing import List, Tuple, Iterable, Generator
import numpy as np
import pandas as pd
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


# Toldo : One could also add automatic typing, if necessary (i.e. taking type hints directly from params.
def docstring_formatter(**kwargs):
    """Decorator with arguments used to format the docstring of a functions.

    `docstring_formatter` is a decorator with arguments, which means that it takes any set of `kwargs` as argument and
    returns a decorator. It should therefore always be called with parentheses (unlike traditional decorators - see
    below). It follows the grammar of `str.format()`, i.e. `{my_format_value}`.
    grammar.

    Example:
        For instance, this code :

        ```Python
        @docstring_formatter(greeting = 'hello')
        def my_func():
            "A simple greeter that says {greeting}"
            # Do your stuff
        ```

        Is actually equivalent with :

        ```Python
        def my_func():
            "A simple greeter that says {greeting}"
            # Do your stuff

        my_func.__doc__ = my_func.__doc__.format(greeting = 'hello')
        ```
    """

    def inner_decorator(func):
        func.__doc__ = func.__doc__.format(**kwargs)
        return func

    return inner_decorator


def timer(iterations: int = 3, number: int = 1_000):
    """Decorator with arguments. Computes the execution time of a function.

    `timer` is a decorator with arguments, which means that it takes any set of `iterations` and `number` as arguments
    and returns a decorator. It should therefore always be called with parentheses (unlike traditional decorators).
    """

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


def read_google_sheet(sheet_id: str, sheet_name: str, **kwargs) -> pd.DataFrame:
    """A simple function to read a google sheet in a `pd.DataFrame`.

    Works at 2022-05-17. See https://towardsdatascience.com/read-data-from-google-sheets-into-pandas-without-the-google-sheets-api-5c468536550
    for more info.
    """

    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url, **kwargs)

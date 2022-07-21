"""Miscellaneous helpers and utilities."""
from functools import wraps
import json
import logging
import timeit
from typing import List, Tuple, Iterable, Generator
import pandas as pd
from jsonschema import Draft6Validator

RectangleType = List[Tuple[int, int]]


def lazy_property(func):
    """Decorator. Makes property computation lazy."""

    private_attr = '_' + func.__name__

    @wraps(func)
    def fget(self):
        if not hasattr(self, private_attr):
            setattr(self, private_attr, func(self))
        return getattr(self, private_attr)

    def fset(self, value):
        setattr(self, private_attr, value)

    def fdel(self):
        delattr(self, private_attr)

    return property(fget, fset, fdel, func.__doc__)


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


def split_list(list_: list, n: int, pad: object) -> List[List[object]]:
    """Divides a list into a list of lists with n elements, padding the last chunk with `pad`."""
    chunks = []
    for x in range(0, len(list_), n):
        chunk = list_[x: n + x]

        if len(chunk) < n:
            chunk += [pad for _ in range(n - len(chunk))]

        chunks.append(chunk)

    return chunks


def aligned_print(*args, **kwargs):
    """Prints `args`, respecting custom spaces between each arg. Used to print aligned rows in a for loop.

    Args:
        kwargs: should contain space : List[int] : the list of space between each arg.
    """
    if 'spaces' not in kwargs.keys():
        kwargs['spaces'] = [20] * len(args)

    to_print = ''
    for arg, space in zip(args, kwargs['spaces']):
        to_print += str(arg) + (' ' * max(space - len(str(arg)), 1))
    print(to_print)



"""Miscellaneous helpers and utilities."""
import inspect
from functools import wraps
import json
import logging
import timeit
from typing import List, Iterable, Generator, Callable, Type, Optional
import pandas as pd
from jsonschema import Draft6Validator
from ajmc.commons.docstrings import docstring_formatter, docstrings


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


def recursive_iterator(iterable: Iterable, iterable_types: Iterable[Type[Iterable]] = (list, tuple)) -> Generator:
    """Iterates recursively through an iterable potentially containing iterables of `iterable_types`."""
    for x in iterable:
        if any([isinstance(x, iterable_type) for iterable_type in iterable_types]):
            for y in recursive_iterator(x):
                yield y
        else:
            yield x


def get_unique_elements(iterable: Iterable, iterable_types: Iterable[Type[Iterable]] = (list, tuple)) -> List[str]:
    """Get the list of elements from any potentially recursive iterable."""
    return list(set([l for l in recursive_iterator(iterable, iterable_types)]))


def validate_json_schema(schema_path: str = 'data/page.schema.json'):
    """Validates a json schema against `Draft6Validator`"""

    with open(schema_path, "r") as file:
        schema = json.loads(file.read())

    Draft6Validator.check_schema(schema)


formatter = logging.Formatter("%(levelname)s - %(name)s -   %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(formatter)


def get_custom_logger(name: str,
                      level: int = logging.INFO):
    """Custom logging wraper, called each time a logger is declared in the package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


@docstring_formatter(**docstrings)
def read_google_sheet(sheet_id: str, sheet_name: str, **kwargs) -> pd.DataFrame:
    """A simple function to read a google sheet in a `pd.DataFrame`.

    Works at 2022-09-29. See https://towardsdatascience.com/read-data-from-google-sheets-into-pandas-without-the-google-sheets-api-5c468536550
    for more info.

    Args:
        sheet_id: {sheet_id}
        sheet_name: {sheet_name}
    """

    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url, **kwargs)


def split_list(list_: list, n: int, pad: object = False) -> List[List[object]]:
    """Divides a list into lists with n elements, pads the last chunk with `pad` if the latter is not `False`.

    Args:
        list_: the list to split
        n: the number of elements in each chunk
        pad: the object to pad the last chunk with. If `False`, no padding is performed.
    """
    chunks = []
    for x in range(0, len(list_), n):
        chunk = list_[x: n + x]

        if len(chunk) < n and pad is not False:
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


def prefix_command_with_conda_env(command: str,
                                  env_name: str,
                                  conda_install_dir: 'Path'):
    return f'source {conda_install_dir / "etc/profile.d/conda.sh"}; conda activate {env_name}; ' + command


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
        if hasattr(self, private_attr):
            delattr(self, private_attr)

    return property(fget, fset, fdel, func.__doc__)


def lazy_init(func):
    """Set attributes for required arguments and defaulted keyword argument which are not None at instantiation.

    Example:
        ```python
        @lazy_init
        def __init__(self, hello, world = None):
            pass
        ```

        is actually equivalent to :

        ```python
        def __init__(self, hello, world = None):
            self.hello = hello

            if world is not None:
                self.world = world
        ```

    Note:
        Warning, this does not handle `*args`.

    """
    specs = inspect.getfullargspec(func)
    assert specs.varargs is None, "`lazy_init` does not handle *args"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        defaults_args_len = len(specs.defaults) if specs.defaults else 0
        required_args_len = len(specs.args) - defaults_args_len
        required_args_names = specs.args[1:required_args_len]
        defaults_args_names = specs.args[required_args_len:]

        # Start with required args
        required_from_args = [(name, value) for name, value in zip(required_args_names, args)]
        required_from_kwargs = [(name, value) for name, value in kwargs.items() if name in required_args_names]
        for name, value in required_from_args + required_from_kwargs:
            setattr(self, name, value)

        # For defaulted args and potential **kwargs, only add if not None
        def_from_args = [(n, v) for n, v in zip(defaults_args_names, args[required_args_len:])]
        def_from_kwargs = [(n, v) for n, v in kwargs.items() if n in defaults_args_names]
        for name, value in def_from_args + def_from_kwargs:
            setattr(self, name, value)

        # Add potential **kwargs
        for name, value in kwargs.items():
            if name not in required_args_names + defaults_args_names:
                setattr(self, name, value)

        func(self, *args, **kwargs)

    return wrapper


def lazy_attributer(attr_name: str, func: Callable, attr_decorator: Callable = lambda x: x):
    """Parametrized decorator returning a decorator which adds the attribute of
    name `attr_name` and of value `func` to the `class_` it decorizes.

    Example:
        ```python
        @lazy_attributer('greeting', lambda self: f'Bonjour {self.name}', property)
        class Student:
            ...
        ```
        is actually equivalent to :

        ```python
        class Student:
            ...

            @property
            def greeting(self):
                return f'Bonjour {self.name}'
        ```

    """

    def set_attribute(class_):
        setattr(class_, attr_name, attr_decorator(func))
        return class_

    return set_attribute


class LazyObject:
    """An object that computes attributes lazily using `compute_function`.

    The set of possible attributes is infinite by default, but can be constrained by setting `constrained_attrs`.
    Otherwise, any called attribute will be created and computing on the fly.

    Example:
        >>> my_lazy_object = LazyObject(lambda attr_name: attr_name + ' has been computed')
        >>> my_lazy_object.hello
        'hello has been computed'
        >>> my_lazy_object.another_greeting_word
        'another_greeting_word has been computed'
    """

    @lazy_init
    def __init__(self,
                 compute_function: Callable,
                 constrained_attrs: Optional[List[str]] = None,
                 **kwargs):
        """Initializes the object.

        Args:
            compute_function: The function to call to compute the attributes.
            constrained_attrs: Constrains the list of possible attributes. The given attributes will be the only one itered upon.
             If `None`, any attribute can be computed.
            **kwargs: Pass kwargs to manually set attributes.
        """
        pass

    def __getattr__(self, attribute):
        if attribute not in self.__dict__:
            if self.constrained_attrs is None or attribute in self.constrained_attrs:  # If constrained, only compute if in constrained attrs
                self.__dict__[attribute] = self.compute_function(attribute)  # Compute and set attribute lazily
            else:
                raise AttributeError(
                    f"""Attribute {attribute} is not in the list of allowed attributes: {self.constrained_attrs}""")
        return self.__dict__[attribute]

    def __dir__(self):
        return ['compute_function',
                'constrained_attrs'] + self.constrained_attrs if self.constrained_attrs is not None else []

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __iter__(self):
        if self.constrained_attrs is None:
            raise TypeError(
                f'You are trying to iterate on a {self.__class__.__name__} but the attributes to iter upon are not defined (self.constrained_attrs is None).')
        else:
            for attr in self.constrained_attrs:
                yield attr, getattr(self, attr)


def inline_def(func, name, doc=None):
    """Returns a function with a new name and docstring.

    Args:
        func: the function to rename
        name: the new name
        doc: the new docstring
    """
    func.__name__ = name
    if doc is not None:
        func.__doc__ = doc
    return func


def log_to_file(log_message: str, log_file: 'Path'):
    """Appends `log_message` to `log_file`"""
    with open(log_file, "a+") as tmp_file:
        tmp_file.write(log_message)
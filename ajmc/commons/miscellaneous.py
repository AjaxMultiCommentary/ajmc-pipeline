"""Miscellaneous helpers and utilities."""

import json
import logging
import re
import timeit
from pathlib import Path
from typing import Generator, Iterable, List, Type

from jsonschema import Draft6Validator

from ajmc.commons import variables as vs

formatter = logging.Formatter("%(levelname)s - %(name)s -   %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)


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


def validate_json_schema(schema_path: Path = vs.SCHEMA_PATH):
    """Validates a json schema against `Draft6Validator`"""
    Draft6Validator.check_schema(json.loads(schema_path.read_text(encoding='utf-8')))


def get_custom_logger(name: str,
                      level: int = logging.INFO):
    """Custom logging wraper, called each time a logger is declared in the package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


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
                                  conda_install_dir: Path):
    """Prefixes a command with with conda env activation command usable in a subshell."""
    return f'source {conda_install_dir / "etc/profile.d/conda.sh"}; conda activate {env_name}; ' + command


def log_to_file(log_message: str, log_file: Path):
    """Appends `log_message` to `log_file`"""
    with open(log_file, "a+") as tmp_file:
        tmp_file.write(log_message)


def get_imports(output_file: Path = None):
    """Get all imports from the package."""

    imports = []
    for pyfile in vs.PACKAGE_DIR.rglob('*.py'):
        text = pyfile.read_text(encoding='utf-8')
        imports += re.findall(r'\nfrom .+? import .*\n', text)
        imports += re.findall(r'\nimport .+?\n', text)

    imports = [imp for imp in imports if 'ajmc' not in imp]
    imports = [re.sub(r'\nfrom (.+?) import .*?\n', r'\1', imp) for imp in imports]
    imports = [re.sub(r'\nimport (.+?)[ \n]', r'\1', imp) for imp in imports]
    imports = [re.sub(r'(.+?)\..+', r'\1', imp) for imp in imports]
    imports = set(imports)

    if output_file:
        output_file.write_text('\n'.join(sorted(imports)), encoding='utf-8')

    return imports

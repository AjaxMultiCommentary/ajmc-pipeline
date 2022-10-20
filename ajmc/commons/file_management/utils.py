"""File management tools and utilities, such as moving/renaming/replacing files, get paths..."""

import os
import shutil
from string import ascii_letters
from datetime import datetime
from typing import Tuple, Optional, List, Union, Callable

from ajmc.commons import variables
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.commons.variables import PATHS
from ajmc.commons.docstrings import docstring_formatter, docstrings
from pathlib import Path
logger = get_custom_logger(__name__)

NUMBERS_LETTERS = '0123456789' + ascii_letters


def int_to_x_based_code(number: int,
                        base: int = 62,
                        fixed_min_len: Union[bool, int] = False,
                        symbols: str = NUMBERS_LETTERS) -> str:
    """Converts an integer to an x-based code.

    Args:
        number: The number to be converted
        base: The base of the code (default: 62)
        fixed_min_len: If int, the minimum length of the code, prepending symbols[0] if necessary.
        symbols: A string of symbols, mapped to integers (starting from 0) to be used to represent numbers in the
        resulting code (default: NUMBERS_LETTERS)

    Returns:
        The x-based code as a string.

    Examples:
        >>> int_to_x_based_code(3,base=2,symbols='01') # Binary code for 3
        '11'
        >>> int_to_x_based_code(11, base=12, symbols='0123456789AB', fixed_min_len=3) # Base 12 code for 11
        '00B'
    """

    # Start the by getting the highest power of 62 that is smaller or equal to the number
    power = 1
    powers = [0]
    while base ** power <= number:
        powers.append(power)
        power += 1

    # Then, we can start building the code
    code = ''
    for i in reversed(powers):
        divisor = number // (base ** i)
        remainder = number % (base ** i)
        number = remainder
        code += symbols[divisor]

    # If the code is shorter than the fixed length, we add leading zeros
    if fixed_min_len:
        code = symbols[0] * (fixed_min_len - len(code)) + code

    return code


def get_62_based_datecode(date: Optional[datetime] = None):
    """Returns a 62-based code based on the date and time.

    This function is used to generate a unique id for each OCR run, based on the date and time. It takes the form of a
    6 digits numbers, where each digit is a letter or a number.
        - The first digit correspond to the last number of the year (e.g. 1 for 2021).
        - The second digit correspond to the month (e.g. 1 for January, B for december).
        - The third digit correspond to the day of the month (e.g. 1 for the 1st, A for 11, K for 21...).
        - The fourth digit correspond to the hour (e.g. 1 for 1am, A for 10am... etc).
        - The fifth digit correspond to the minute (e.g. 1 for 1min, B for 11min... etc).
        - The sixth digit correspond to the second (e.g. 1 for 1sec, C for 12sec... etc).

    Note:
        Base 62 is set to default, as it allows for displaying hours, minutes and seconds in a single digit.
        It corresponds to the 10 arabic numbers,the 26 latin lower-case letters and the 26 latin capitals.

    Args:
        date: The date to be converted to a 62-based code. If None, the current date is used.

    Returns:
        The 62-based code as a string.

    Examples:
        >>> get_62_based_datecode(datetime(year=2021,month=1, day=1, hour=1, minute=1, second=1))
        "111111"
        >>> get_62_based_datecode(datetime(year=2021,month=12, day=31, hour=23, minute=59, second=59))
        "1cvnXX"
    """
    date = datetime.now() if date is None else date

    datecode = ''
    datecode += int_to_x_based_code(int(str(date.year)[-1]))
    datecode += int_to_x_based_code(date.month)
    datecode += int_to_x_based_code(date.day)
    datecode += int_to_x_based_code(date.hour)
    datecode += int_to_x_based_code(date.minute)
    datecode += int_to_x_based_code(date.second)

    return datecode


def verify_path_integrity(path: str, path_pattern: str) -> None:
    """Verify the integrity of an `ocr_path` with respect to ajmc's folder structure.
    Args:
        path: The path to be tested
        path_pattern: The pattern to be respected (see `commons.variables.FOLDER_STRUCTURE_PATHS`).
    """
    dirs = path.strip('/').split('/')  # Stripping trailing and leading '/' before splitting
    dirs_pattern = path_pattern.strip('/').split('/')

    for dir_, pattern in zip(reversed(dirs), reversed(dirs_pattern)):
        # Make sure the detected commentary id is known
        if pattern == '[commentary_id]':
            assert dir_ in variables.COMMENTARY_IDS, f"""The commentary id ({dir_}) detected 
            in the provided path ({path}) does not match any known commentary_id. """

        # Skip other placeholder patterns (e.g. '[ocr_run]').
        elif pattern[0] == '[' and pattern[-1] == ']':
            continue

        # Verify fixed patterns
        else:
            assert pattern == dir_, f"""The provided path ({path}) does not seem to be compliant with AJMC's 
            folder structure. Please make sure you are observing the following pattern: \n {path_pattern} """


def parse_ocr_path(path: str) -> Tuple[str, str, str]:
    """Extracts the base_path, commentary_id and ocr_run from an AJMC compliant OCR-outputs path."""
    dirs = path.rstrip('/').split('/')  # Stripping trailing '/' before splitting
    dirs_pattern = variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'].strip('/').split('/')
    base = '/'.join(dirs[:-len(dirs_pattern)])
    rest = dirs[-len(dirs_pattern):]
    commentary_id = rest[dirs_pattern.index('[commentary_id]')]
    ocr_run = rest[dirs_pattern.index('[ocr_run]')]

    return base, commentary_id, ocr_run


@docstring_formatter(**docstrings)
def find_file_by_name(base_name: str,
                      directory: str = None) -> Optional[str]:
    """Gets the path to a file from its base name.

    Args:
        base_name: The base name of the file to be found.
        directory: {directory} in which to look for the file.

    Returns:
        The absolute path to the file.
    """

    files = [f for f in os.listdir(directory) if base_name in f]

    assert len(
        files) <= 1, f"""There are {len(files)} files matching the name {base_name} in {directory}. Please check."""

    if len(files) == 0:
        logger.debug(f"""Page_id {base_name} matches no file in {directory}, skipping...""")
        return None

    else:
        return os.path.join(directory, files[0])


@docstring_formatter(**docstrings)
def guess_ocr_format(ocr_path: str) -> str:
    """Guesses the ocr-format of a file.

    Args:
        ocr_path: {ocr_path}

    Returns:
        The ocr-format of the file, either 'pagexml', 'krakenhocr' or 'tesshocr'.
    """

    if ocr_path[-3:] == 'xml':
        return 'pagexml'
    elif ocr_path[-4:] == 'html':
        return 'krakenhocr'
    elif ocr_path[-4:] == 'hocr':
        return 'tesshocr'
    else:
        raise NotImplementedError("""The format could not be identified. It looks like the format is neither 
        `tesshocr`, nor `krakenhocr` nor `pagexml`, which are the only formats this module deals with.""")


@docstring_formatter(**docstrings)
def move_files_in_each_commentary_dir(relative_src: str,
                                      relative_dst: str,
                                      base_dir: str = PATHS['base_dir']):
    """Moves/rename files/folders in the folder structure.

    Args:
        relative_src: relative path of the source file/folder, from commentary root (e.g. `'ocr/groundtruth'`)
        relative_dst: relative path of the destination file/folder,  from commentary root (e.g. `'ocr/groundtruth'`)
        base_dir: {base_dir}
    """

    for dir_name in walk_dirs(base_dir):
        if os.path.exists(os.path.join(base_dir, dir_name, relative_src)):
            # Moves the file/folder
            shutil.move(os.path.join(base_dir, dir_name, relative_src),
                        os.path.join(base_dir, dir_name, relative_dst))


@docstring_formatter(**docstrings)
def create_folder_in_each_commentary_dir(relative_dir_path: str,
                                         base_dir: str = PATHS['base_dir']):
    """Creates an empty directory in each commentary directory.

    Args:
        relative_dir_path: relative path of the directory to be created, from the base_dir.
        base_dir: {base_dir}
    """
    for dir_path in walk_dirs(base_dir, prepend_base=True):
        os.makedirs(os.path.join(dir_path, relative_dir_path), exist_ok=True)


def merge_subdirectories(parent_dir: str, destination_dir: str):
    """Recursive function to merge the content of all sub-directories in `parent_dir` into one"""
    os.makedirs(destination_dir, exist_ok=True)
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            shutil.copy(os.path.join(root, file), destination_dir)
        for dir_ in dirs:
            merge_subdirectories(os.path.join(root, dir_), destination_dir)


# Todo change this to return a Path object
def walk_dirs(path: str, prepend_base: bool = False, recursive: bool = False) -> List[str]:
    """Walks over the dirs in path."""
    for root, dirs, files in os.walk(path):
        for dir_ in sorted(dirs):
            if prepend_base:
                yield os.path.join(root, dir_)
            else:
                yield dir_
        if not recursive:
            break


def walk_files(path: str,
               filter: Optional[Callable[[Path], bool]] = None,
               recursive: bool = False):
    """Walks over the files in path.

    Args:
        path: The path to walk over.
        filter: A function that takes a filename as input and returns a boolean.
        recursive: Whether to walk recursively or not.
    """
    for root, dirs, files in os.walk(path):
        for filename in sorted(files):
            path = Path(os.path.join(root, filename))
            if filter is not None:
                if filter(path):
                    yield path
            else:
                yield path

        if not recursive:
            break
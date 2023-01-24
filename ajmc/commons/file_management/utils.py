"""File management tools and utilities, such as moving/renaming/replacing files, get paths..."""
# CHECKED 2023-01-24
import shutil
from datetime import datetime
from pathlib import Path
from string import ascii_letters
from typing import Callable, List, Optional, Union

from ajmc.commons import variables as vs
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)

def int_to_x_based_code(number: int,
                        base: int = 62,
                        fixed_min_len: Union[bool, int] = False,
                        symbols: str = '0123456789' + ascii_letters) -> str:
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
        code = code.rjust(fixed_min_len, symbols[0])

    return code


def get_62_based_datecode(date: Optional[datetime] = None) -> str:
    """Returns a 62-based code based on the date and time.

    This function is mainly used to generate a unique id for each OCR run, based on the date and time. It takes the form of a
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


@docstring_formatter(**docstrings)
def move_files_in_each_commentary_dir(relative_src_path: Union[str, Path],
                                      relative_dst_path: Union[str, Path],
                                      base_dir: Path = vs.COMMS_DATA_DIR):
    """Moves/rename files/folders in the folder structure.

    Args:
        relative_src_path: relative path of the source file/folder, from commentary base_dir (e.g. `'ocr/groundtruth'`)
        relative_dst_path: relative path of the destination file/folder,  from commentary base_dir (e.g. `'ocr/groundtruth'`)
        base_dir: {base_dir}
    """

    for dir_ in walk_dirs(base_dir):
        abs_src_path = dir_ / relative_src_path
        abs_dst_path = dir_ / relative_dst_path
        if abs_src_path.exists():
            logger.info(f"Moving {abs_src_path} to {abs_dst_path}")
            abs_dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(abs_src_path, abs_dst_path)


def walk_dirs(path: Path, recursive: bool = False) -> List[Path]:
    """Walks over the dirs in path."""
    for dir_ in path.glob('*'):
        if dir_.is_dir():
            yield dir_
            if recursive:
                for dir__ in walk_dirs(dir_, recursive):
                    yield dir__


def walk_files(parent_dir: Path,
               filter_func: Optional[Callable[[Path], bool]] = None,
               recursive: bool = False):
    """Walks over the files in parent_dir.

    Args:
        parent_dir: The path to walk over.
        filter_func: A function that takes a filename as input and returns a boolean.
        recursive: Whether to walk recursively or not.
    """
    for path in parent_dir.glob('**/*' if recursive else '*'):
        if path.is_file():
            if filter_func is None or filter_func(path):
                yield path
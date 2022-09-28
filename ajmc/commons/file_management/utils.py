"""File management tools and utilities, such as moving/renaming/replacing files, get paths..."""

import os
import shutil
from string import ascii_letters
from datetime import datetime
from typing import Tuple, Optional, List

from ajmc.commons import variables
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.commons.variables import PATHS

logger = get_custom_logger(__name__)

NUMBERS_LETTERS = '0123456789' + ascii_letters


def int_to_62_based_code(number: int) -> str:
    assert number < 62**3, 'This function is implemented for numbers up to `62**3`.'
    code = ''
    for i in [3844, 62, 1]:  # 62**2, 62**1, 62**0
        num = number // i
        assert num <= 62
        code += NUMBERS_LETTERS[num]
        number = number - (num * i)
    return code


def get_62_based_datecode(date: Optional[datetime] = None):

    if date is None:
        date = datetime.now()

    datecode = ''

    datecode += str(date.year)[-1]
    datecode += NUMBERS_LETTERS[date.month]
    datecode += NUMBERS_LETTERS[date.day]

    day_seconds = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).seconds

    datecode += int_to_62_based_code(day_seconds)

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


def get_path_from_id(page_id: str, dir_: str = None) -> str:
    """Gets the path to a page file (image, ocr...) from its id"""

    files = [f for f in os.listdir(dir_) if page_id in f]

    assert len(files) <= 1, f"""There are {len(files)} files matching the name {page_id} in {dir_}. Please check."""

    if len(files) == 0:
        logger.debug(f"""Page_id {page_id} matches no file in {dir_}, skipping...""")
        return ""

    else:
        return os.path.join(dir_, files[0])


def guess_ocr_format(ocr_path: str) -> str:
    """Guesses the ocr-format of a file.

    Args:
        ocr_path: Absolute path to an ocr output file"""

    if ocr_path[-3:] == 'xml':
        return 'pagexml'
    elif ocr_path[-4:] == 'html':
        return 'krakenhocr'
    elif ocr_path[-4:] == 'hocr':
        return 'tesshocr'
    else:
        raise NotImplementedError("""The format could not be identified. It looks like the format is neither 
        `tesshocr`, nor `krakenhocr` nor `pagexml`, which are the only formats this module deals with.""")


def move_files_in_each_commentary_dir(relative_src: str,
                                      relative_dst: str,
                                      base_dir: str = PATHS['base_dir']):
    """Moves/rename files/folders in the folder structur

    Args:
        relative_src: relative path of the source file/folder, from commentary root (e.g. `'ocr/groundtruth'`)
        relative_dst: relative path of the destination file/folder,  from commentary root (e.g. `'ocr/groundtruth'`)

    Returns:
        None
    """

    for dir_name in walk_dirs(base_dir):
        if os.path.exists(os.path.join(base_dir, dir_name, relative_src)):
            # Moves the file/folder
            shutil.move(os.path.join(base_dir, dir_name, relative_src),
                        os.path.join(base_dir, dir_name, relative_dst))


def create_folder_in_each_commentary_dir(relative_dir_path:str):
    for dir_name in next(os.walk(PATHS['base_dir']))[1]:
        os.makedirs(os.path.join(PATHS['base_dir'], dir_name, relative_dir_path), exist_ok=True)


def merge_subdirectories(parent_dir: str, destination_dir: str):
    """Recursive function to merge the content of all sub-directories in `parent_dir` into one"""
    os.makedirs(destination_dir, exist_ok=True)
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            shutil.copy(os.path.join(root, file), destination_dir)
        for dir_ in dirs:
            merge_subdirectories(os.path.join(root, dir_), destination_dir)


def walk_dirs(path: str, prepend_base: bool = False) -> List[str]:
    """Walks over the dirs in path."""
    if prepend_base:
        return [os.path.join(path, dir_) for dir_ in sorted(next(os.walk(path))[1])]
    else:
        return sorted(next(os.walk(path))[1])
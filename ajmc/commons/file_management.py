"""File management tools and utilities, such as moving/renaming/replacing files, get paths..."""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from string import ascii_letters
from typing import Callable, List, Optional, Union, Tuple, Set

import pandas as pd

from ajmc.commons import variables as vs
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_ajmc_logger

logger = get_ajmc_logger(__name__)


def int_to_x_based_code(number: int,
                        base: int = 62,
                        fixed_min_len: Union[bool, int] = False,
                        symbols: str = '0123456789' + ascii_letters) -> str:
    """Converts an integer to an x-based code.

    Examples:
        >>> int_to_x_based_code(3,base=2,symbols='01') # Binary code for 3
        '11'
        >>> int_to_x_based_code(11, base=12, symbols='0123456789AB', fixed_min_len=3) # Base 12 code for 11
        '00B'

    Args:
        number: The number to be converted
        base: The base of the code (default: 62)
        fixed_min_len: If int, the minimum length of the code, prepending symbols[0] if necessary.
        symbols: A string of symbols, mapped to integers (starting from 0) to be used to represent numbers in the
        resulting code (default: NUMBERS_LETTERS)

    Returns:
        The x-based code as a string.
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

    Examples:
        >>> get_62_based_datecode(datetime(year=2021,month=1, day=1, hour=1, minute=1, second=1))
        "111111"
        >>> get_62_based_datecode(datetime(year=2021,month=12, day=31, hour=23, minute=59, second=59))
        "1cvnXX"

    This function is mainly used to generate a unique id for each OCR run, based on the date and time. It takes the form of a
    6 digits numbers, where each digit is a letter or a number.
    * The first digit correspond to the last number of the year (e.g. 1 for 2021).
    * The second digit correspond to the month (e.g. 1 for January, B for december).
    * The third digit correspond to the day of the month (e.g. 1 for the 1st, A for 11, K for 21...).
    * The fourth digit correspond to the hour (e.g. 1 for 1am, A for 10am... etc).
    * The fifth digit correspond to the minute (e.g. 1 for 1min, B for 11min... etc).
    * The sixth digit correspond to the second (e.g. 1 for 1sec, C for 12sec... etc).

    Note:
        Base 62 is set to default, as it allows for displaying hours, minutes and seconds in a single digit.
        It corresponds to the 10 arabic numbers,the 26 latin lower-case letters and the 26 latin capitals.

    Args:
        date: The date to be converted to a 62-based code. If None, the current date is used.

    Returns:
        The 62-based code as a string.
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
                                      root_dir: Path = vs.COMMS_DATA_DIR):
    """Moves/rename files/folders in the folder structure.

    Args:
        relative_src_path: relative path of the source file/folder, from commentary root_dir (e.g. ``'ocr/groundtruth'``)
        relative_dst_path: relative path of the destination file/folder,  from commentary root_dir (e.g. ``'ocr/groundtruth'``)
        root_dir: {root_dir}
    """

    for dir_ in walk_dirs(root_dir):
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


def clean_all_ocr_runs_outputs_dirs():
    """Cleans all ocr/runs/xxxxxxx/outputs dirs from lace outputs and other weird files."""
    for comm_dir in walk_dirs(vs.COMMS_DATA_DIR):
        comm_runs_dir = vs.get_comm_ocr_runs_dir(comm_dir.name)
        if comm_runs_dir.is_dir():
            for ocr_run_dir in walk_dirs(comm_runs_dir):
                outputs_dir = vs.get_comm_ocr_outputs_dir(comm_dir.name, ocr_run_dir.name)
                for path in outputs_dir.glob('*'):
                    if comm_dir.name not in path.name and path.suffix not in ['.sh', '']:
                        command = f'rm -rf {path}'
                        print(command)
                        os.system(command)


def find_replace_in_all_comm_dirs(file_rel_path: str,
                                  old_pattern: str,
                                  new_pattern: str,
                                  check: bool = False):
    for dir_ in walk_dirs(vs.COMMS_DATA_DIR):
        file_path = vs.get_comm_root_dir(dir_.name) / file_rel_path

        if file_path.exists():
            text = file_path.read_text(encoding='utf-8')
            text = text.replace(old_pattern, new_pattern)
            if check:
                print(f'Would write {file_path}, replacing {old_pattern} with {new_pattern}')
                if input('Type enter to continue, or anything else to stop') != '':
                    continue
            print(f'Writing {file_path}')
            file_path.write_text(text, encoding='utf-8')


@docstring_formatter(**docstrings)
def check_via_spreadsheet_conformity(comm_id: str,
                                     check_comm_only: bool = False) -> Tuple[Set[str], Set[str]]:
    """Verifies that ``via_project`` actually contains the page ids listed in ``sheet_page_ids`` and vice-versa.

    This function is used to make sure that the pages marked as groundtruth in spreadsheets are actually present in the
    respective via_project and vice-versa. If there are differences between the sets of via and spreadsheet pages,
    prints a logging.error. In any case, return the two sets of difference.

    Args:
        comm_id: {commentary_id}
        check_comm_only: Whether to check only the pages where only commentary regions are annotated.

    Returns:
         A tuple containing two sets of str:
             1. The difference ``sheet_pages - via_pages``.
             2. The difference ``via_pages - sheet_pages``.
    """

    via_project = json.loads(vs.get_comm_via_path(comm_id).read_text(encoding='utf-8'))
    sheet = get_olr_gt_spreadsheet()
    sheet_pages = set(sheet['page_id'][sheet['commentary_id'] == comm_id])

    via_full_gt_pages = []  # This contains the pages which are entirely annotated
    via_comm_gt_pages = []  # This contains the pages where only commentary regions are annotated

    for v in via_project['_via_img_metadata'].values():

        # If the page has annotations which are neither commentary nor undefined, append to full pages
        if any([r['region_attributes']['text'] not in ['commentary', 'undefined'] for r in v['regions']]):
            via_full_gt_pages.append(v['filename'].split('.')[0])

        # if the page has only commentary or undefined annotations, append to commentary pages
        elif all([r['region_attributes']['text'] in ['commentary', 'undefined'] for r in v['regions']]) and \
                any([r['region_attributes']['text'] in ['commentary'] for r in v['regions']]):
            via_comm_gt_pages.append(v['filename'].split('.')[0])

    via_pages = set(via_comm_gt_pages) if check_comm_only else set(via_full_gt_pages)

    diff_sheet_via = sheet_pages.difference(via_pages)
    diff_via_sheet = via_pages.difference(sheet_pages)

    if diff_sheet_via:
        print(f"""The following pages are in annotated in sheet 
        but not in via : \n{diff_sheet_via}\n""")

    if diff_via_sheet:
        print(f"""The following pages are in annotated in via 
        but not in sheet : \n{diff_via_sheet}\n""")

    if not diff_sheet_via and not diff_via_sheet:
        print("""OLR checking passed : pages in via and sheet are identical.""")

    return diff_sheet_via, diff_via_sheet


def check_ocr_gt_spreadsheet_conformity(comm_id: str):
    """Checks that a commentary's ocr gt directory contains the same pages as the spreadsheet."""

    sheet = get_ocr_gt_spreadsheet()
    sheet_gt_page_ids = set(sheet['page_id'][sheet['commentary_id'] == comm_id])

    via_project = json.loads(vs.get_comm_via_path(comm_id).read_text(encoding='utf-8'))
    via_gt_page_ids = set([Path(p['filename']).stem for p in via_project['_via_img_metadata'].values()])

    diff_via_sheet = via_gt_page_ids.difference(sheet_gt_page_ids)
    diff_sheet_via = sheet_gt_page_ids.difference(via_gt_page_ids)

    if diff_via_sheet:
        print(f"""The following pages are in annotated in drive 
        but not in sheet : \n{diff_via_sheet}\n""")

    if diff_sheet_via:
        print(f"""The following pages are in annotated in sheet 
        but not in drive : \n{diff_sheet_via}\n""")

    if not diff_via_sheet and not diff_sheet_via:
        print("""OCR checking passed : Pages in drive and sheet are identical.""")


def data_sanity_check():
    """Performs a sanity check on ajmc base data.

    This function notably checks:
    - That the ajmc's folder structure is respected (i.e. that requested files and folders exist).
    - That the data is in the correct format (e.g. that images have the right extension).
    - That data is compliant with spreadsheets.
    """

    # Check all the commentaries are listed in vs.ALL_COMM_IDS
    drive_comm_ids = set([dir_.name for dir_ in walk_dirs(vs.COMMS_DATA_DIR)])
    code_comm_ids = set(vs.ALL_COMM_IDS)

    if drive_comm_ids.difference(code_comm_ids):
        logger.warning(f"Commentaries ids in the drive but not in the code: {drive_comm_ids.difference(code_comm_ids)}")
    if code_comm_ids.difference(drive_comm_ids):
        logger.warning(f"Commentaries ids in the code but not in the drive: {code_comm_ids.difference(drive_comm_ids)}")

    # Check that all the commentaries have the right folder structure
    for comm_id in vs.ALL_COMM_IDS:

        print(f"\n\nChecking commentary {comm_id}...".center(40, '-'))

        if not vs.get_comm_img_dir(comm_id).exists():
            logger.warning(f"Commentary {comm_id} does not have an image folder.")
        if not vs.get_comm_ocr_runs_dir(comm_id).exists():
            logger.warning(f"Commentary {comm_id} does not have an ocr folder.")
        if not vs.get_comm_canonical_dir(comm_id).exists():
            logger.warning(f"Commentary {comm_id} does not have a canonical folder.")

        # Check that all the images have the right extension
        for img_path in vs.get_comm_img_dir(comm_id).glob(f'{comm_id}*'):
            if img_path.suffix != vs.DEFAULT_IMG_EXTENSION:
                logger.warning(f"Image {img_path} has a wrong extension.")

        # Check OCR spreadsheet conformity with vias
        check_ocr_gt_spreadsheet_conformity(comm_id)

        # Check via-project conformity
        check_via_spreadsheet_conformity(comm_id)


@docstring_formatter(**docstrings)
def read_google_sheet(sheet_id: str, sheet_name: str, **kwargs) -> pd.DataFrame:
    """A simple function to read a google sheet in a ``pd.DataFrame``.

    Works at 2022-09-29. See https://towardsdatascience.com/read-data-from-google-sheets-into-pandas-without-the-google-sheets-api-5c468536550
    for more info.

    Args:
        sheet_id: {sheet_id}
        sheet_name: {sheet_name}

    Returns:
        The sheet as a ``pd.DataFrame``.
    """

    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url, **kwargs)


_OLR_GT_SPREADSHEET = None


def get_olr_gt_spreadsheet() -> pd.DataFrame:
    """Returns the OLR spreadsheet as a ``pd.DataFrame``."""

    global _OLR_GT_SPREADSHEET
    if _OLR_GT_SPREADSHEET is None:
        _OLR_GT_SPREADSHEET = read_google_sheet(vs.SPREADSHEETS['olr_gt'], 'olr_gt')
    return _OLR_GT_SPREADSHEET


_OCR_GT_SPREADSHEET = None


def get_ocr_gt_spreadsheet() -> pd.DataFrame:
    """Returns the OCR spreadsheet as a ``pd.DataFrame``."""

    global _OCR_GT_SPREADSHEET
    if _OCR_GT_SPREADSHEET is None:
        _OCR_GT_SPREADSHEET = read_google_sheet(vs.SPREADSHEETS['ocr_gt'], 'ocr_gt')
    return _OCR_GT_SPREADSHEET


_METADATA_SPREADSHEET = None


def get_metadata_spreadsheet() -> pd.DataFrame:
    """Returns the metadata spreadsheet as a ``pd.DataFrame``."""

    global _METADATA_SPREADSHEET
    if _METADATA_SPREADSHEET is None:
        _METADATA_SPREADSHEET = read_google_sheet(vs.SPREADSHEETS['metadata'], 'metadata')
    return _METADATA_SPREADSHEET


def get_commit_hash(dir_: Path) -> str:
    """Returns the commit hash of a directory."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dir_).decode().strip()


def has_file_changed(repo_path: Union[str, Path], file_path: Union[str, Path], commit_hash1: str, commit_hash2: str):
    """
    Check if a specific file has changed between two commits using subprocess.

    Args:
    - repo_path (str): Path to the Git repository.
    - file_path (str): Path to the file in the repository.
    - commit_hash1 (str): Hash of the first commit.
    - commit_hash2 (str): Hash of the second commit.

    Returns:
    - bool: True if the file has changed, False otherwise.
    """
    diff_output = subprocess.check_output(f"git -C {repo_path} diff {commit_hash1} {commit_hash2} -- {file_path}",
                                          shell=True,
                                          stderr=subprocess.STDOUT,
                                          text=True)

    # If there are changes, the diff_output will not be empty
    return bool(diff_output)


def check_change_in_commentary_data(comm_id: str, current_commit: str, last_commit: str):
    """Checks if the commentary data has changed between two commits"""
    return has_file_changed(vs.COMMS_DATA_DIR, vs.get_comm_via_path(comm_id), current_commit, last_commit)


def check_change_in_ne(comm_id: str, current_commit: str, last_commit: str):
    """Checks if the files related to a specific in the NE corpus has changed between two commits"""
    return any([has_file_changed(vs.NE_CORPUS_DIR, p, current_commit, last_commit)
                for p in vs.NE_CORPUS_DIR.glob(f'{comm_id}*.xmi')])


def check_change_in_lemlink(comm_id: str, current_commit: str, last_commit: str):
    return any([has_file_changed(vs.LEMLINK_CORPUS_DIR, p, current_commit, last_commit)
                for p in vs.LEMLINK_CORPUS_DIR.glob(f'{comm_id}*.xmi')])

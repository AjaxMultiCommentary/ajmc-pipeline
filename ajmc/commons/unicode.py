"""This file contains unicode variables and functions which serve processing unicode characters."""

import re
from typing import List, Tuple


def get_all_chars_from_range(start: str, end: str) -> str:
    """Get all characters from a range of unicode characters.

    Args:
        start (str): The first character in the range.
        end (str): The last character in the    range.

    Returns:
        str: A string containing all characters in the range.
    """
    return ''.join([chr(ordinal) for ordinal in range(ord(start), ord(end) + 1)])


def get_all_chars_from_ranges(ranges: List[Tuple[str, str]]) -> str:
    """Get all characters from a list of ranges of unicode characters.

    Args:
        ranges (list): A list of tuples of unicode characters ranges.

    Returns:
        str: A string containing all characters in the ranges.
    """
    return ''.join([get_all_chars_from_range(start, end) for start, end in ranges])


CHARSETS_RANGES = {
    'latin': [('A', 'Z'), ('a', 'z'), ('\u00C0', '\u00FF'), ('\u0152', '\u0152'), ('\u0153', '\u0153')],
    'greek': [('\u0386', '\u038A'), ('\u038C', '\u038C'), ('\u038E', '\u03A1'), ('\u03A3', '\u03E1'),  # standard greek, no coptic/separate diacritics
              ('\u1F00', '\u1F15'), ('\u1F18', '\u1F1D'), ('\u1F20', '\u1F45'), ('\u1F48', '\u1F4D'),  # polytonic greek...
              ('\u1F50', '\u1F57'), ('\u1F59', '\u1F59'), ('\u1F5B', '\u1F5B'), ('\u1F5D', '\u1F5D'),
              ('\u1F5F', '\u1F7D'), ('\u1F80', '\u1FB4'), ('\u1FB6', '\u1FBC'), ('\u1FBE', '\u1FBE'),
              ('\u1FC2', '\u1FC4'), ('\u1FC6', '\u1FCC'), ('\u1FD0', '\u1FD3'), ('\u1FD6', '\u1FDB'),
              ('\u1FE0', '\u1FEC'), ('\u1FF2', '\u1FF4'), ('\u1FF6', '\u1FFC'), ('\u2126', '\u2126'),
              ('\u0300', '\u0300'), ('\u0301', '\u0301'), ('\u0313', '\u0313'), ('\u0314', '\u0314'),
              ('\u0345', '\u0345'), ('\u0342', '\u0342'), ('\u0308', '\u0308')],
    'numeral': [('0', '9')],
    'punctuation': [('\u0020', '\u002F'), ('\u003A', '\u003F'), ('\u005B', '\u0060'), ('\u007B', '\u007E'), ('\u00A8', '\u00A8'),
                    ('\u00B7', '\u00B7')]
}

CHARSETS_CHARS = {charset: get_all_chars_from_ranges(ranges) for charset, ranges in CHARSETS_RANGES.items()}

CHARSETS_PATTERNS = {charset: re.compile(rf'[{charset_chars}]', re.UNICODE) for charset, charset_chars in CHARSETS_CHARS.items()}

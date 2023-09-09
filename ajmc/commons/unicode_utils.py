"""This file contains unicode variables and functions which serve processing unicode characters."""

import re
import unicodedata
from typing import List, Tuple, Callable

from ajmc.commons.arithmetic import safe_divide


def harmonise_ligatures(text: str) -> str:
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff')
    text = text.replace('ﬃ', 'ffi')
    text = text.replace('ﬄ', 'ffl')
    text = text.replace('ﬅ', 'ft')
    text = text.replace('ﬆ', 'st')
    return text


def harmonise_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


def harmonise_punctuation(text: str) -> str:
    text = text.replace('═', '=')
    text = text.replace('‟', '"')
    text = text.replace('⸗', '—')
    text = text.replace('═', '=')
    text = text.replace('●', '•')
    text = text.replace('⟨', '〈')
    text = text.replace('⟩', '〉')
    text = text.replace('‐', '-')
    text = text.replace('‑', '-')
    text = text.replace('‒', '-')
    text = text.replace('―', '-')
    text = text.replace('‥', '..')
    text = text.replace('…', '...')
    text = text.replace('‧', '·')
    text = text.replace('′', "'")  # prime
    text = text.replace('″', '"')  # double prime
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('͵', ',')  # greek lower numeral sign to comma)
    text = text.replace('ʹ', "'")
    text = text.replace('ʺ', '"')
    text = text.replace('ʻ', "'")
    text = text.replace('ʼ', "'")
    text = text.replace('ʽ', "'")
    text = text.replace('ˈ', "'")
    text = text.replace('ˊ', "'")
    text = text.replace('ˋ', "'")
    text = text.replace('ˌ', ",")
    text = text.replace('\x92', "'")
    return text


def harmonise_non_printable(text: str) -> str:
    text = text.replace('\x92', "'")
    text = text.replace('', 'ï')
    text = text.replace('', 'ï')
    text = text.replace('', 'ï')
    text = text.replace('­', '-')
    text = text.replace('­', '-')
    return text


def harmonise_miscellaneous_symbols(text: str) -> str:
    text = text.replace('', 'ï')
    text = text.replace('', 'ï')
    text = text.replace('', 'ï')
    text = text.replace('­', '-')
    text = text.replace('­', '-')
    text = text.replace('⁓', '~')
    text = text.replace('∼', '~')
    text = text.replace('➳', '→')
    text = text.replace('⇒', '→')
    text = text.replace('⇔', '↔')
    text = text.replace('⇐', '←')
    text = text.replace('⇔', '↔')
    text = text.replace('➤', '→')
    text = text.replace('˖', '+')
    text = text.replace('ʼ', "'")
    text = text.replace('×', 'x')
    text = text.replace('‟', '"')
    text = text.replace('‛', "'")
    text = text.replace('ϰ', 'κ')
    text = text.replace('ϱ', 'ρ')
    text = text.replace('ϑ', 'θ')
    text = text.replace('‟', '"')
    text = text.replace('ꝙ', 'q')
    text = text.replace('ꝛ', 'r')
    text = text.replace('Ꝙ', 'Q')
    text = text.replace('Ꝛ', 'R')
    text = text.replace('ꝓ', 'p')
    text = text.replace('Ꝑ', 'P')
    text = text.replace('🄰', 'A')
    text = text.replace('🄱', 'B')
    text = text.replace('🄲', 'C')
    text = text.replace('🄳', 'D')
    text = text.replace('🄴', 'E')
    text = text.replace('🄵', 'F')
    text = text.replace('🄶', 'G')
    text = text.replace('🄷', 'H')
    text = text.replace('🄸', 'I')
    text = text.replace('🄹', 'J')
    text = text.replace('🄺', 'K')
    text = text.replace('🄻', 'L')
    text = text.replace('🄼', 'M')
    text = text.replace('🄽', 'N')
    text = text.replace('🄾', 'O')
    text = text.replace('🄿', 'P')
    text = text.replace('🅀', 'Q')
    text = text.replace('🅁', 'R')
    text = text.replace('🅂', 'S')
    text = text.replace('🅃', 'T')
    text = text.replace('🅄', 'U')
    text = text.replace('🅅', 'V')
    text = text.replace('🅆', 'W')
    text = text.replace('🅇', 'X')
    text = text.replace('🅈', 'Y')
    text = text.replace('🅉', 'Z')
    text = text.replace('⸢', '[')
    text = text.replace('⸣', ']')
    text = text.replace('⸤', '[')
    text = text.replace('⌋', ']')
    text = text.replace('⌈', '[')
    text = text.replace('⸥', ']')
    text = text.replace('⁄', '/')
    text = text.replace('µ', 'μ')
    return text


def harmonise_unicode(text: str,
                      harmonise_functions: Tuple[Callable[[str], str]] = (
                              harmonise_punctuation, harmonise_miscellaneous_symbols, harmonise_ligatures),
                      harmonise_space_chars: bool = True) -> str:
    """Harmonise unicode characters.

    Note:
        This function takes an ``NFC`` string and returns an ``NFC`` string.

    Args:
        text (str): The text to harmonise.
        harmonise_functions (tuple): A tuple of functions to apply to the text. Each function should take an NFC string as input and return an NFC string as output.
        harmonise_space_chars (bool): Whether to harmonise space characters.

    Returns:
        str: The harmonised text (an ``NFC`` string).
    """
    for function in harmonise_functions:
        text = function(text)
    if harmonise_space_chars:
        text = harmonise_spaces(text)
    return text


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
              ('\u0345', '\u0345'), ('\u0342', '\u0342'), ('\u0308', '\u0308'), ('·', '·'), ('\u0384', '\u0384')],
    'numeral': [('0', '9')],
    'punctuation': [('\u0020', '\u002F'), ('\u003A', '\u003F'), ('\u005B', '\u0060'), ('\u007B', '\u007E'), ('\u00A8', '\u00A8'),
                    ('\u00B7', '\u00B7')]
}

CHARSETS_CHARS_NFC = {charset: get_all_chars_from_ranges(ranges) for charset, ranges in CHARSETS_RANGES.items()}

CHARSETS_CHARS_NFD = {charset: ''.join(set(unicodedata.normalize('NFD', charset_chars))) for charset, charset_chars in CHARSETS_CHARS_NFC.items()}

CHARSETS_PATTERNS = {charset: re.compile(rf'[{charset_chars}]', re.UNICODE) for charset, charset_chars in CHARSETS_CHARS_NFC.items()}


def chunk_string_by_charsets(string: str, fallback: str = 'latin'):
    """Chunk a string by character set, returning a list of tuples of the form (chunk, charset).

    Example:
        >>> chunk_string_by_charsets('Hello Γειά σου Κόσμε World')
        [('Hello ', 'latin'), ('Γειά σου Κόσμε ', 'greek'), ('World', 'latin')]

    Args:
        string (str): The string to chunk.

    Returns:
        list: A list of tuples of the form (chunk, charset).
    """

    chunks = []
    chunk = string[0]
    chunk_charset = get_char_charset(chunk, fallback=fallback)

    for char in string[1:]:
        char_charset = get_char_charset(char, fallback=fallback)

        if any([re.match(r'\s', char),
                char_charset == chunk_charset]):
            chunk += char

        else:
            chunks.append((chunk, chunk_charset))
            chunk, chunk_charset = char, char_charset

    chunks.append((chunk, chunk_charset))
    return chunks


def get_char_charset(char: str, fallback: str = 'fallback') -> str:
    """Returns the charset of a character, if any, ``fallback`` otherwise."""
    for charset_name, charset_chars in CHARSETS_CHARS_NFC.items():
        if char in charset_chars:
            return charset_name
    else:
        return fallback


def count_chars_by_charset(string: str, charset: str) -> int:
    """Counts the number of chars by unicode characters set.

    Example:
        ``count_chars_by_charset('γεια σας, world', 'greek')`` returns ``7`` as there are 7 greek
        chars in ``string``.

    Args:
        string: a NFC-normalized string (default). For NFD-normalized strings, use ``count_chars_by_charset_nfd``.
        charset: should be ``'greek'``, ``'latin'``, ``'numeral'``, ``'punctuation'``.

    Returns:
        int: the number of charset-matching characters in ``string``.
    """
    return sum([c in CHARSETS_CHARS_NFC[charset] for c in string])


def count_chars_by_charset_nfd(string: str, charset: str) -> int:
    """Counts the number of chars by unicode characters set.

    Example:
        ``count_chars_by_charset('γεια σας, world', 'greek')`` returns ``7`` as there are 7 greek
        chars in ``string``.

    Args:
        string: a NFD-normalized string. For NFC-normalized strings, use ``count_chars_by_charset``.
        charset: should be ``'greek'``, ``'latin'``, ``'numeral'``, ``'punctuation'``.

    Returns:
        int: the number of charset-matching characters in ``string``.
    """
    return sum([c in CHARSETS_CHARS_NFD[charset] for c in string])


def is_charset_string(string: str,
                      charset: str,
                      threshold: float = 0.5,
                      strict: bool = True) -> bool:
    """Returns True if more than ``threshold`` of chars in string are in ``charset``, False otherwise.

    Args:
        string: self explanatory
        charset: should be ``'greek'``, ``'latin'``, ``'numeral'``, ``'punctuation'`` or a valid ``re``-pattern,
                    for instance ``r'([\u00F4-\u00FF])'``
        threshold: the threshold above which the function returns True
        strict: if True, only chars in ``charset`` are considered, if False, chars in ``charset``, ``'numeral'`` and
                ``'punctuation'`` are considered.
    """

    if strict:
        return safe_divide(count_chars_by_charset(string=string, charset=charset), len(string)) >= threshold
    else:
        return safe_divide(sum(count_chars_by_charset(string, charset_) for charset_ in [charset, 'numeral', 'punctuation']),
                           len(string)) >= threshold


def is_charset_string_nfd(string: str,
                          charset: str,
                          threshold: float = 0.5,
                          strict: bool = True) -> bool:
    """Returns True if more than ``threshold`` of chars in string are in ``charset``, False otherwise.

    Args:
        string: a NFD-normalized string. For NFC-normalized strings, use ``is_charset_string``.
        charset: should be ``'greek'``, ``'latin'``, ``'numeral'``, ``'punctuation'``.
        threshold: the threshold above which the function returns True
        strict: if True, only chars in ``charset`` are considered, if False, chars in ``charset``, ``'numeral'`` and
                ``'punctuation'`` are considered.
    """

    if strict:
        return count_chars_by_charset_nfd(string=string, charset=charset) / len(string) >= threshold
    else:
        return sum(count_chars_by_charset_nfd(string, charset_) for charset_ in [charset, 'numeral', 'punctuation']) / len(string) >= threshold


def get_char_unicode_name(char: str) -> str:
    """Returns the unicode name of a character."""
    try:
        return unicodedata.name(char)
    except:
        return ''

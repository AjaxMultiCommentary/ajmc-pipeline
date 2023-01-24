"""A bunch of ocr tools."""

import re
from pathlib import Path
from typing import Union

from ajmc.commons import variables as vs
from ajmc.commons.arithmetic import safe_divide
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.file_management.utils import get_62_based_datecode


def harmonise_unicode(text: str):
    text = re.sub(r"᾽", "’", text)
    text = re.sub(r"ʼ", "’", text)
    text = re.sub(r"'", "’", text)
    text = re.sub(r"—", "-", text)
    text = re.sub(r"„", '"', text)

    return text


def is_greek_char(char: str) -> bool:
    """Returns True if char is a Greek character, False otherwise."""
    return bool(re.match(vs.CHARSETS['greek'], char))


def is_latin_char(char: str) -> bool:
    """Returns True if char is a Latin character, False otherwise."""
    return bool(re.match(vs.CHARSETS['latin'], char))


def is_punctuation_char(char: str) -> bool:
    """Returns True if char is a punctuation character, False otherwise."""
    return bool(re.match(vs.CHARSETS['punctuation'], char))


def is_number_char(char: str) -> bool:
    """Returns True if char is a number character, False otherwise."""
    return bool(re.match(vs.CHARSETS['numbers'], char))


def count_chars_by_charset(string: str, charset: str) -> int:
    """Counts the number of chars by unicode characters set.

    Example:
        `count_chars_by_charset('γεια σας, world', 'greek')` returns `7` as there are 7 greek
        chars in `string`.

    Args:
        string: self explanatory
        charset: should be `'greek'`, `'latin'`, `'numbers'`, `'punctuation'` or a valid `re`-pattern,
                 for instance `r'([\u00F4-\u00FF])'`

    Returns:
        int: the number of charset-matching characters in `string`.
    """
    try:
        pattern = vs.CHARSETS[charset]
    except KeyError:
        pattern = re.compile(charset, flags=re.UNICODE)

    return len(re.findall(pattern, string))


def is_greek_string(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of alphabet chars in strin are Greek, False otherwise."""
    alpha_text = "".join([c for c in text if c.isalpha()])  # cleaning the text from non-alphabetical characters
    if alpha_text:
        proportion_greek_chars = count_chars_by_charset(string=alpha_text, charset='greek') / len(alpha_text)
        return proportion_greek_chars >= threshold
    else:
        return False


def is_latin_string(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of alphabet chars in strin are Latin, False otherwise."""
    alpha_text = "".join([c for c in text if c.isalpha()])  # cleaning the text from non-alphabetical characters
    if alpha_text:
        proportion_latin_chars = count_chars_by_charset(string=alpha_text, charset='latin') / len(alpha_text)
        return proportion_latin_chars >= threshold
    else:
        return False


def is_punctuation_string(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of chars in strin are punctuation, False otherwise."""
    if text:
        proportion_punctuation_chars = safe_divide(count_chars_by_charset(string=text, charset='punctuation'), len(text))
        return proportion_punctuation_chars >= threshold
    else:
        return False


def is_number_string(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of chars in strin are numbers, False otherwise."""
    alphanum_text = "".join([c for c in text if c.isalnum()])  # cleaning the text from non-alphabetical characters
    if alphanum_text:
        proportion_numbers = count_chars_by_charset(string=alphanum_text, charset='numbers') / len(alphanum_text)
        return proportion_numbers >= threshold
    else:
        return False

@docstring_formatter(**docstrings)
def get_kraken_command(commentary_id: str, model_path: Union[str, Path]) -> str:
    """LEGACY. Returns the command to be executed by Kraken.

    Args:
        commentary_id: {commentary_id}
        model_path: the path to the model to be used.
    """

    ocr_outputs_dir = vs.get_comm_ocr_runs_dir(commentary_id) / (get_62_based_datecode() + '_' + model_path.stem)
    ocr_outputs_dir.mkdir(parents=True, exist_ok=True)

    img_dir = vs.get_comm_img_dir(commentary_id)
    img_paths = sorted([p for p in img_dir.glob(f'*{vs.DEFAULT_IMG_EXTENSION}')])
    ocr_paths = [(ocr_outputs_dir / p.name).with_suffix('.hocr') for p in img_paths]

    file_list = ' '.join([f'-i {img} {ocr}' for img, ocr in zip(img_paths, ocr_paths)])

    return f'kraken {file_list} -h segment ocr --model model_path'


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

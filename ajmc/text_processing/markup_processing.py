"""
This module contains all the functions to read OCR- and OLR-output, for all the
formats (json, hocr, page-XML...)
"""

from typing import List, Union

import bs4

from ajmc.commons import variables as vs
from ajmc.commons.geometry import Shape


# ===========================  COORDS EXTRACTERS  ============================
def get_hocr_element_bbox(element: bs4.element.Tag) -> Shape:
    """Extract bbox from ``title='...; bbox X1 Y1 X2 Y2; ...'``"""
    coords = [int(num) for el in element['title'].split(';') if el.strip().startswith('bbox')
              for num in el.strip().split()[1:]]
    return Shape([(coords[0], coords[1]), (coords[2], coords[3])])


def get_pagexml_element_bbox(element: bs4.element.Tag) -> Shape:
    points = element.find('pc:Coords')['points'].split()  # A List[str]
    return Shape([tuple(int(coord) for coord in point.split(',')) for point in points])


def get_json_element_bbox(element: dict) -> Shape:
    return Shape.from_xyxy(*element['xyxy'])


def get_element_bbox(element: Union[bs4.element.Tag, dict], ocr_format: str = 'hocr') -> Shape:
    """Generic extractor. Extracts the bbox of a markup element."""
    if ocr_format in ['hocr', 'html']:
        return get_hocr_element_bbox(element)
    elif ocr_format == "xml":
        return get_pagexml_element_bbox(element)
    elif ocr_format == "json":
        return get_json_element_bbox(element)
    else:
        raise NotImplementedError(f'Accepted formats are {vs.OCR_OUTPUTS_EXTENSIONS}.')





# =========================  TEXT EXTRACTERS  =================================
def get_element_text(element: Union[bs4.element.Tag, dict], ocr_format: str = 'hocr') -> str:
    """Generic extractor. Extracts the text from a markup element."""
    if ocr_format in ['hocr', 'html']:
        return element.text
    elif ocr_format == "xml":
        return element.find('pc:TextEquiv', recursive=False).find('pc:Unicode', recursive=False).contents[0]
    elif ocr_format == "json":
        return element['text']
    else:
        raise NotImplementedError(f'Accepted formats are {vs.OCR_OUTPUTS_EXTENSIONS}.')


# ===========================  ELEMENT EXTRACTERS  ============================
def find_all_tesshocr_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with ``name`` in ``element``.

    Args:
        element: A ``bs4.element.Tag`` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of ``bs4.element.Tag``.
    """

    if 'word' in name:
        return element.find_all(attrs={'class': 'ocrx_word'})
    elif 'line' in name:
        return element.find_all(attrs={'class': ['ocr_line', 'ocrx_line', 'ocr_textfloat', 'ocr_header']})
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_krakenhocr_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with ``name`` in ``element``.

    Args:
        element: A ``bs4.element.Tag`` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of ``bs4.element.Tag``.
    """

    if 'word' in name:
        return element.find_all(attrs={'class': 'ocr_word'})
    elif 'line' in name:
        return element.find_all(attrs={'class': ['ocr_line', 'ocrx_line', 'ocr_textfloat', 'ocr_header']})
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_pagexml_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with ``name`` in ``element``.

    Args:
        element: A ``bs4.element.Tag`` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of ``bs4.element.Tag``.
    """

    if 'word' in name:
        return element.find_all(name='pc:Word')
    elif 'line' in name:
        return element.find_all(name='pc:TextLine')
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_json_elements(element: dict, name: str) -> List[dict]:
    """Finds all sub-elements with ``name`` in ``element``.

    Args:
        element: A ``dict`` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of ``dict``.
    """

    if 'word' in name:
        return element['words']
    elif 'line' in name:
        return element
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_elements(element: Union[bs4.element.Tag, bs4.BeautifulSoup, dict], name: str, format: str = 'hocr') -> List[
    Union[bs4.element.Tag, dict]]:
    """Generic extractor. Finds all sub-elements with ``name`` in ``element``.

    Args:
        element: A ``bs4.element.Tag`` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.
        format: The ocr-output format to consider: 'tesshocr', 'krakenhocr' or 'xml'.

    Returns:
        A list of ``bs4.element.Tag``.
    """

    if format == 'hocr':
        return find_all_tesshocr_elements(element, name)
    elif format == 'html':
        return find_all_krakenhocr_elements(element, name)
    elif format == 'xml':
        return find_all_pagexml_elements(element, name)
    elif format == 'json':
        return find_all_json_elements(element, name)
    else:
        raise NotImplementedError(f'Accepted formats are {vs.OCR_OUTPUTS_EXTENSIONS}.')

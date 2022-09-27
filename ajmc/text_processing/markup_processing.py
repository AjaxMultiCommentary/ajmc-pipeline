"""
This module contains all the functions to read OCR- and OLR-output, for all the
formats (json, hocr, page-XML...)
"""

import bs4
from ajmc.commons.geometry import Shape
from typing import List, Union


# ===========================  GENERIC PARSER  ============================
def parse_markup_file(path: str) -> bs4.BeautifulSoup:
    """Generic parser which works for PageXML and HOCR files"""
    with open(path, 'r', encoding="utf-8") as f:
        return bs4.BeautifulSoup(f.read(), 'xml')


# ===========================  COORDS EXTRACTERS  ============================
def get_hocr_element_bbox(element: bs4.element.Tag) -> Shape:
    """Extract bbox from `title='...; bbox X1 Y1 X2 Y2; ...'`"""
    coords = [int(num) for el in element['title'].split(';') if el.strip().startswith('bbox')
              for num in el.strip().split()[1:]]
    return Shape([(coords[0], coords[1]), (coords[2], coords[3])])


def get_pagexml_element_bbox(element: bs4.element.Tag) -> Shape:
    points = element.find('pc:Coords')['points'].split()  # A List[str]
    return Shape([tuple(int(coord) for coord in point.split(',')) for point in points])


def get_element_bbox(element: bs4.element.Tag, ocr_format: str = 'tesshocr') -> Shape:
    """Generic extractor. Extracts the bbox of a markup element."""
    if 'hocr' in ocr_format:
        return get_hocr_element_bbox(element)
    elif ocr_format == "pagexml":
        return get_pagexml_element_bbox(element)
    else:
        raise NotImplementedError("""Accepted formats are 'tesshocr', 'krakenhocr' and 'pagexml'.""")


# =========================  TEXT EXTRACTERS  =================================
def get_element_text(element: bs4.element.Tag, ocr_format: str = 'tesshocr') -> str:
    """Generic extractor. Extracts the text from a markup element."""
    if 'hocr' in ocr_format:
        return element.text
    elif ocr_format == "pagexml":
        return element.find('pc:TextEquiv', recursive=False).find('pc:Unicode', recursive=False).contents[0]
    else:
        raise NotImplementedError("""Accepted formats are 'tesshocr', 'krakenhocr' and 'pagexml'.""")


# ===========================  ELEMENT EXTRACTERS  ============================
def find_all_tesshocr_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    if 'word' in name:
        return element.find_all(attrs={'class': 'ocrx_word'})
    elif 'line' in name:
        return element.find_all(attrs={'class': ['ocr_line', 'ocrx_line', 'ocr_textfloat', 'ocr_header']})
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_krakenhocr_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    if 'word' in name:
        return element.find_all(attrs={'class': 'ocr_word'})
    elif 'line' in name:
        return element.find_all(attrs={'class': ['ocr_line', 'ocrx_line', 'ocr_textfloat', 'ocr_header']})
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_pagexml_elements(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all sub-elements with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    if 'word' in name:
        return element.find_all(name='pc:Word')
    elif 'line' in name:
        return element.find_all(name='pc:TextLine')
    else:
        raise NotImplementedError("""Accepted elements are 'lines' and 'words'.""")


def find_all_elements(element: Union[bs4.element.Tag, bs4.BeautifulSoup], name: str, format: str = 'tesshocr') -> List[bs4.element.Tag]:
    """Generic extractor. Finds all sub-elements with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.
        format: The ocr-output format to consider: 'tesshocr', 'krakenhocr' or 'pagexml'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    if format == 'tesshocr':
        return find_all_tesshocr_elements(element, name)
    elif format == 'krakenhocr':
        return find_all_krakenhocr_elements(element, name)
    elif format == 'pagexml':
        return find_all_pagexml_elements(element, name)
    else:
        raise NotImplementedError("""Accepted formats are 'tesshocr', 'krakenhocr' and 'pagexml'.""")

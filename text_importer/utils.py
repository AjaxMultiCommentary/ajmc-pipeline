"""`text_importer` utility functions."""

import bs4

from oclr.utils.geometry import Shape


def parse_xml(path: str) -> bs4.BeautifulSoup:
    """Generic parser which works for PageXML and HOCR files"""
    with open(path, 'r') as f:
        return bs4.BeautifulSoup(f.read(), 'xml')


def get_hocr_tag_coords(element: bs4.element.Tag) -> Shape:
    """Extract coords from `title='...; bbox X1 Y1 X2 Y2; ...'`"""
    coords = [int(num) for el in element['title'].split(';') if el.strip().startswith('bbox')
              for num in el.strip().split()[1:]]
    return Shape.from_points([(coords[0], coords[1]), (coords[2], coords[3])])
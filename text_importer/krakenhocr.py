"""Support for Kraken and Lace/Ciaconna HOCR type.

Notes:
    Kraken/Lace output is less consistent than Tesseract. Hence, main differences with `TessHocr` include:
        - Some naming difference
        - Automatic reshaping of word and line boxes to their minimal bounding rectangle
        - Deletion of empty words

"""

from typing import List
import bs4
from commons.custom_typing_types import ElementType
from oclr.utils.geometry import is_point_within_rectangle, Shape, get_bounding_rectangle_from_points
from text_importer.base_classes import Element, Page, Commentary
from commons.utils import lazy_property
from text_importer.utils import parse_xml, get_hocr_tag_coords


class KrakenHocrCommentary(Commentary):

    def __init__(self, commentary_id: str):
        super().__init__(commentary_id)

    def _create_page(self, page_id):
        return KrakenHocrPage(page_id, self)


class KrakenHocrPage(Page):

    def __init__(self, page_id: str, commentary: KrakenHocrCommentary = None):
        super().__init__(page_id, commentary)

    def _create_commentary(self):
        return KrakenHocrCommentary(self.id.split('_')[0])

    def _create_element(self, markup):
        return KrakenHocrElement(markup, self)

    def _parse(self):
        return parse_xml(self.ocr_path)

    def _get_coords(self):
        raise NotImplementedError('Page coords are not provided by Kraken/Lace Hocr files')

    def _find_tags(self, name: str) -> List[ElementType]:
        return find_all_krakenhocr_tags(self.markup, name)


class KrakenHocrElement(Element):
    """A class representing OCR elements such as lines, words and characters."""

    def __init__(self, element: 'bs4.element.Tag', page: KrakenHocrPage):
        self.id = element['id']
        self._markup = element
        self.page = page

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def contours(self):
        return find_included_contours(self, self.page.image.contours)

    @lazy_property
    def words(self):
        return shrink_elements_to_contours(self._get_children('words'), self.contours)

    def _get_children(self, name: str = None) -> List['KrakenHocrElement']:
        return [self.__class__(el, self.page) for el in self._find_tags(name)]

    def _get_parent(self, name: str = None):
        if name == 'page':
            return self.page
        elif name == 'commentary':
            return self.page.commentary
        else:
            raise NotImplementedError

    def _get_image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    def _get_coords(self):
        return get_hocr_tag_coords(self.markup)

    def _parse(self):
        return self._markup

    def _get_text(self):
        return self.markup.text

    def _find_tags(self, name: str) -> List[ElementType]:
        return find_all_krakenhocr_tags(element=self.markup, name=name)


def find_all_krakenhocr_tags(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all tags with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    name = 'ocr_word' if 'word' in name.lower() \
        else 'ocr_line' if 'line' in name.lower() \
        else name

    return element.find_all(attrs={'class': name})



def find_included_contours(element:ElementType, contours: List[Shape]) -> List[Shape]:
    e_contours = [c for c in contours
                  if any([is_point_within_rectangle(p, element.coords.bounding_rectangle) for p in c.points])
                  and max([p[1] for p in c.points]) <= element.coords.bounding_rectangle[2][1]
                  and min([p[1] for p in c.points]) >= element.coords.bounding_rectangle[0][1]]
    return e_contours

def shrink_elements_to_contours(elements: ElementType, contours: List[Shape]) -> List[ElementType]:

    for el in elements:
        # find contours with at least which have at least one point in word and which would not make the box taller
        e_contours = [c for c in contours
                      if any([is_point_within_rectangle(p, el.coords.bounding_rectangle) for p in c.points])
                      and max([p[1] for p in c.points]) <= el.coords.bounding_rectangle[2][1]
                      and min([p[1] for p in c.points]) >= el.coords.bounding_rectangle[0][1]]

        # get the words new coords
        word_points = [p for c in e_contours for p in c.points]
        word_points = word_points if word_points else [[0, 0], [0, 0], [0, 0], [0, 0]]

        el._coords = Shape(get_bounding_rectangle_from_points(word_points))

    return elements



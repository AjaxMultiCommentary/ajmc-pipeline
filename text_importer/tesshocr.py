from typing import List
import bs4
from commons.types import ElementType
from oclr.utils.geometry import Shape
from text_importer.base_classes import Element, Page, Commentary
from text_importer.file_management import get_page_ocr_path
from commons.utils import lazy_property
from text_importer.utils import parse_xml, get_hocr_tag_coords


class TessHocrCommentary(Commentary):

    def __init__(self, commentary_id: str):
        super().__init__(commentary_id)

    def _create_page(self, page_id):
        return TessHocrPage(page_id, self)


class TessHocrPage(Page):

    def __init__(self, page_id: str, commentary: TessHocrCommentary = None):
        super().__init__(page_id, commentary)

    def _create_commentary(self):
        return TessHocrCommentary(self.id.split('_')[0])

    def _create_element(self, markup):
        return TessHocrElement(markup, self)

    def _parse(self):
        return parse_xml(self.ocr_path)

    def _get_coords(self):
        return get_hocr_tag_coords(self.markup.find(attrs={'class': 'ocr_page'}))

    def _find_tags(self, name: str) -> List[ElementType]:
        return find_all_tesshocr_tags(self.markup, name)


class TessHocrElement(Element):
    """A class representing OCR elements such as lines, words and characters."""

    def __init__(self, element: 'bs4.element.Tag', page: TessHocrPage):
        self.id = element['id']
        self._markup = element
        self.page = page

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def words(self):
        return self._get_children('words')

    def _get_children(self, name: str = None) -> List['TessHocrElement']:
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
        return find_all_tesshocr_tags(element=self.markup, name=name)


def find_all_tesshocr_tags(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all tags with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    if 'line' not in name.lower():
        name = 'ocrx_word' if 'word' in name.lower() \
            else 'ocr_carea' if 'region' in name.lower() \
            else name

        return element.find_all(attrs={'class': name})

    else:
        return element.find_all(attrs={'class': ['ocr_line', 'ocr_']})




# #%%
# from oclr.utils.image_processing import draw_rectangles
# import cv2
#
# page = TessHocrPage('sophoclesplaysa05campgoog_0146')
# matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.regions], page.image.matrix.copy(), (255, 0, 0), 3)
# matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.lines], matrix, (0, 255, 0))
# matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.words], matrix)
# cv2.imwrite('/Users/sven/Desktop/testtesseract.png', matrix)
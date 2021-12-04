from typing import List
import bs4
from commons.types import ElementType, PathType
from oclr.utils.geometry import Shape
from text_importer.base_classes import Element, Page, Commentary
from commons.utils import lazy_property
from text_importer.file_management import get_page_ocr_path
from text_importer.utils import parse_xml


class PagexmlCommentary(Commentary):

    def __init__(self, commentary_id: str):
        super().__init__(commentary_id)

    def _create_page(self, page_id):
        return PagexmlPage(page_id, self)


class PagexmlPage(Page):

    def __init__(self, page_id: str, commentary: PagexmlCommentary = None):
        super().__init__(page_id, commentary)

    def _create_commentary(self):
        return PagexmlCommentary(self.id.split('_')[0])

    def _create_element(self, markup):
        return PagexmlElement(markup, self)

    def _parse(self):
        return parse_xml(self.ocr_path)

    def _get_coords(self):
        p = self.markup.find('pc:Page')
        return Shape.from_xywh(0, 0, w=int(p['imageWidth']), h=int(p['imageHeight']))

    def _find_tags(self, name: str) -> List[ElementType]:
        return find_all_pagexml_tags(self.markup, name)



class PagexmlElement(Element):
    """A class representing OCR elements such as lines, words and characters."""

    def __init__(self, element: 'bs4.element.Tag', page: PagexmlPage):
        self.id = get_pagexml_tag_id(element)
        self._markup = element
        self.page = page

    @lazy_property
    def ocr_path(self):
        return get_page_ocr_path(page_id=self.id, ocr_format='pagexml')

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def words(self):
        return self._get_children('words')

    def _get_children(self, name: str = None) -> List['PagexmlElement']:
        return [self.__class__(el, self.page) for el in find_all_pagexml_tags(element=self.markup, name=name)]

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
        return get_pagexml_tag_coords(self.markup)

    def _parse(self):
        return self._markup

    def _get_text(self):
        return get_pagexml_tag_text(self.markup)

    def _find_tags(self, name: str) -> List[ElementType]:
        return find_all_pagexml_tags(element=self.markup, name=name)




def get_pagexml_tag_coords(element: bs4.element.Tag) -> Shape:
    points = element.find('pc:Coords')['points'].split()  # A List[str]
    return Shape([tuple(int(coord) for coord in point.split(',')) for point in points])


def get_pagexml_tag_text(element: bs4.element.Tag) -> str:
    """Retrieves the content of pc:TextRegion, pc:TextLine or pc:Word tags"""
    return element.find('pc:TextEquiv', recursive=False).find('pc:Unicode', recursive=False).contents[0]


def get_pagexml_tag_id(element: bs4.element.Tag) -> str:
    return element['id']


def find_all_pagexml_tags(element: bs4.element.Tag, name: str) -> List[bs4.element.Tag]:
    """Finds all tags with `name` in `element`.

    Args:
        element: A `bs4.element.Tag` to search within.
        name: The name of the tags to search for, e.g. 'lines', 'words'.

    Returns:
        A list of `bs4.element.Tag`s.
    """

    name = 'pc:Word' if 'word' in name.lower() \
        else 'pc:TextLine' if 'line' in name.lower() \
        else 'pc:TextRegion' if 'region' in name.lower() \
        else name

    return element.find_all(name=name)

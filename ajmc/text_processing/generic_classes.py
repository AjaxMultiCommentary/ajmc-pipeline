import re
from abc import abstractmethod
from typing import List, Optional, Type, Union, Iterable

from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.image import draw_textcontainers
from ajmc.commons.miscellaneous import lazy_property, LazyObject, recursive_iterator
from ajmc.commons.variables import CHILD_TYPES, PARENT_TYPES, TEXTCONTAINER_TYPES
from ajmc.olr.utils import get_olr_splits_page_ids
from ajmc.text_processing.cas_utils import export_commentary_to_xmis


class TextContainer:
    """Generic object for ocr and canonical representations of text containers."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'commentary':
                self.parents.commentary = v
            else:
                setattr(self, k, v)

    @abstractmethod
    @docstring_formatter(**docstrings)
    def _get_children(self, children_type) -> List[Optional[Type['TextContainer']]]:
        """Gets the children of `self` which are of the given `children_type`.

        Args:
            children_type: {children_type}

        Returns:
            A list of children.
        """
        pass

    @abstractmethod
    @docstring_formatter(**docstrings)
    def _get_parent(self, parent_type) -> Optional[Type['TextContainer']]:
        """Gets the `TextContainer` of type `parent_type` which has self as a child.

        Note:
            Unlike `_get_children`, `get_parent` returns a single text container and not lists of
            text containers, as each text container can only have one parent.

        Args:
            parent_type: {parent_type}
        """
        pass

    @lazy_property
    def children(self) -> LazyObject:
        return LazyObject(compute_function=self._get_children, constrained_attrs=CHILD_TYPES)

    @lazy_property
    def parents(self) -> LazyObject:
        return LazyObject(compute_function=self._get_parent, constrained_attrs=PARENT_TYPES)

    @lazy_property
    def type(self) -> str:
        """Generic method to get a `TextContainer`'s type."""
        return re.findall(r'[A-Z][a-z]+', self.__class__.__name__)[-1].lower()

    # todo 👁️ there should be a possibility to add various space chars at the end of words
    # todo 👁️ there should be a possibility to de-hyphenate
    @lazy_property
    def text(self) -> str:
        """Generic method to get a `CanonicalTextContainer`'s text."""
        return ' '.join([w.text for w in self.children.words])


class Commentary:

    def _get_parent(self, parent_type: str) -> Optional[Type['TextContainer']]:
        return None  # A commentary has no parents

    def get_page(self, page_id: str) -> Optional['CanonicalPage']:
        """A simple shortcut to get a page from its id."""
        return [p for p in self.children.pages if p.id == page_id][0]

    def to_xmis(self: Type['Commentary'],
                make_jsons: bool,
                make_xmis: bool,
                json_dir: Optional[str] = None,
                xmi_dir: Optional[str] = None,
                region_types: Union[List[str], str] = 'all'):
        export_commentary_to_xmis(self, make_jsons, make_xmis, json_dir, xmi_dir, region_types)

    @lazy_property
    def olr_groundtruth_pages(self) -> List['CanonicalPage']:
        """A list of `CanonicalPage` objects containing the groundtruth of the OLR."""
        page_ids = get_olr_splits_page_ids(self.id)
        return [p for p in self.children.pages if p.id in page_ids]


class Page:

    def draw_textcontainers(self,
                            tc_types: List[str] = CHILD_TYPES,
                            output_path: Optional[str] = None):
        draw = self.image.matrix.copy()

        for type in tc_types:
            draw = draw_textcontainers(draw, output_path, *getattr(self.children, type))

        return draw

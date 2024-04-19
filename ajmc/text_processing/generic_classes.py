import re
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Type, Union, Callable

import cv2
import unicodedata
from lazy_objects.lazy_objects import lazy_property, LazyObject

from ajmc.commons import variables as vs, image as ajmc_img
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.ocr import variables as ocr_vs

logger = get_ajmc_logger(__name__)


class TextContainer:
    """``TextContainer`` is the mother class for all text containers.

    Note:
        A text container is a container for text. It can be a page, a line, a word, a character, etc.The mother class therefor contains all the
        methods and attributes that are common to all text containers: ``children``, ``parents``, ``type``, ``text``, etc. Please refer to the documentation
        of each of these attributes and methods for more information.

    Warning:
        The ``TextContainer`` class is abstract and should not be directly instantiated. Instead, use one of its children classes.
    """

    def __init__(self, **kwargs):
        """Initializes the ``TextContainer``."""
        for k, v in kwargs.items():
            if k in vs.TEXTCONTAINER_TYPES:
                setattr(self.parents, k, v)
            else:
                setattr(self, k, v)


    @abstractmethod
    @docstring_formatter(**docstrings)
    def _get_children(self, children_type) -> List[Optional[Type['TextContainer']]]:
        """Gets the children of ``self`` which are of the given ``children_type``.

        Args:
            children_type: {children_type}

        Returns:
            A list of children.
        """
        pass

    @abstractmethod
    @docstring_formatter(**docstrings)
    def _get_parent(self, parent_type) -> Optional[Type['TextContainer']]:
        """Gets the ``TextContainer`` of type ``parent_type`` which has self as a child.

        Note:
            Unlike ``_get_children``, ``get_parent`` returns a single text container and not lists of
            text containers, as each text container can only have one parent.

        Args:
            parent_type: {parent_type}
        """
        pass

    @lazy_property
    def children(self) -> LazyObject:
        return LazyObject(compute_function=self._get_children, constrained_attrs=vs.CHILD_TYPES)

    @lazy_property
    def parents(self) -> LazyObject:
        return LazyObject(compute_function=self._get_parent, constrained_attrs=vs.TEXTCONTAINER_TYPES)

    @lazy_property
    def type(self) -> str:
        """The type of ``TextContainer`` (e.g. 'page', 'line', 'word', 'region'.)."""
        return re.findall(r'[A-Z][a-z]+', self.__class__.__name__)[-1].lower()

    @lazy_property
    def text(self) -> str:
        """Generic method to get a ``CanonicalTextContainer``'s text."""
        return ' '.join([w.text for w in self.children.words])


class Commentary(TextContainer):

    def _get_parent(self, parent_type: str) -> None:
        return None  # A commentary has no parents, just here to implement abstractmethod

    def get_page(self, page_id: str) -> Optional[vs.PageType]:
        """A simple shortcut to get a page from its id."""
        return [p for p in self.children.pages if p.id == page_id][0]

    def get_section(self, section_type: str) -> Optional[Type['TextContainer']]:
        """A simple shortcut to get a section from its type."""
        try:
            return [s for s in self.children.sections if section_type in s.section_types][0]
        except IndexError:
            return None

    def export_ocr_gt_file_pairs(self,
                                 output_dir: Optional[Union[str, Path]] = None,
                                 unicode_format: str = 'NFC'):
        """Exports png-txt file pairs for each line in each ocr groundtruth page of the commentary.

        Args:
            output_dir: The directory to which the file pairs should be exported. If None, files are written to the
            default output directory (``vs.get_comm_ocr_gt_pairs_dir``).
            unicode_format: The unicode format to which the text should be normalized. See
            https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize for more information.
        """

        # Define output directory
        output_dir = vs.get_comm_ocr_gt_pairs_dir(self.id) if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over groundtruth pages
        for page in self.ocr_gt_pages:
            for i, line in enumerate(page.children.lines):
                line.image.write(output_dir / f'{page.id}_{i}{ocr_vs.IMG_EXTENSION}')
                (output_dir / f'{page.id}_{i}{ocr_vs.GT_TEXT_EXTENSION}').write_text(unicodedata.normalize(unicode_format, line.text),
                                                                                     encoding='utf-8')

    @lazy_property
    def root_dir(self) -> Path:
        """The directory containing the commentary's files."""
        return vs.get_comm_root_dir(self.id)

    @lazy_property
    def img_dir(self) -> Path:
        """The directory containing the commentary's images."""
        return vs.get_comm_img_dir(self.id)

    @lazy_property
    def olr_gt_pages(self) -> List[vs.PageType]:
        """A list of ``CanonicalPage`` objects containing the groundtruth of the OLR."""
        return [p for p in self.children.pages if p.id in self.olr_gt_page_ids]

    @lazy_property
    def ner_gt_pages(self) -> List[vs.PageType]:
        """A list of ``CanonicalPage`` objects containing the groundtruth of the NER."""

        return [p for p in self.children.pages if p.id in self.ner_gt_page_ids]

    @lazy_property
    def lemlink_gt_pages(self) -> List[vs.PageType]:
        """A list of ``CanonicalPage`` objects containing the groundtruth of the lemmatization."""

        return [p for p in self.children.pages if p.id in self.lemlink_gt_page_ids]


    def get_duplicates(self):
        comm_diffs = {}
        for page in self.children.pages:
            page_diffs = {}
            for child_type in ['regions', 'lines', 'words']:
                boxes = []
                for child in getattr(page.children, child_type):
                    try:
                        if child.bbox.bbox not in boxes:
                            boxes.append(child.bbox.bbox)
                        else:
                            print(f'Page {page.id} has duplicated {child_type}: {child.text} at {child.bbox.bbox}')
                    except ValueError:
                        print(f'Page {page.id} has a {child_type} with no bbox: {child.text}')

                page_diffs[child_type] = len(getattr(page.children, child_type)) - len(boxes)

            comm_diffs[page.id] = page_diffs

            if any(page_diffs.values()):
                print(f'************** Page {page.id} ************** ')
                for k, v in page_diffs.items():
                    print(f'{k}: {v}')
        return comm_diffs

    def safe_check_sections(self):
        errors = 0
        previous_end = self.children.sections[0].end
        for s in self.children.sections[1:]:
            if s.start <= previous_end:
                print(f'Overlapping sections: {previous_end} and {s.start}')
                errors += 1
            elif s.start == previous_end + 1:
                pass
            else:
                print(f'Missing page between {previous_end} and {s.start}')
                errors += 1
            previous_end = s.end
        if s.end != len(self.children.pages):
            print(f'Missing end pages between {s.end} and {len(self.children.pages)}')
            errors += 1

        return not errors > 0


    def is_safe(self):
        comm_diff = self.get_duplicates()
        for page_id, diffs in comm_diff.items():
            if any(diffs.values()):
                return False
        return True


class Page:

    def draw_textcontainers(self,
                            tc_types: List[str] = vs.CHILD_TYPES,
                            output_path: Optional[Union[str, Path]] = None,
                            text_getter: Optional[Callable] = None) -> 'np.ndarray':
        """Draws the text containers of the page on the page's image.

        Args:
            tc_types: A list of text container types to draw. By default, all text containers are drawn.
            output_path: The path to which the image should be saved. If None, the image is not saved.
        """
        draw = self.image.matrix.copy()

        for type_ in tc_types:
            draw = ajmc_img.draw_textcontainers(draw, None, text_getter, *getattr(self.children, type_))

        if output_path is not None:
            cv2.imwrite(str(output_path), draw)

        return draw

    @lazy_property
    def number(self) -> int:
        """The page number, such as it appears in the page's id.

        Warning:
            This number doesn't correspond to the page number as it appears in the scanned book !
        """
        return int(self.id.split('_')[-1])

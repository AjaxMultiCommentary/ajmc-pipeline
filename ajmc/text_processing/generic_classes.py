import re
import unicodedata
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Type, Union

import cv2

from ajmc.commons import variables
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.image import draw_textcontainers
from ajmc.commons.miscellaneous import get_custom_logger, lazy_init, lazy_property, LazyObject, read_google_sheet
from ajmc.olr.utils import get_olr_splits_page_ids

logger = get_custom_logger(__name__)


# @dataclass
class TextContainer:
    """Generic object for ocr and canonical representations of text containers."""

    @lazy_init
    def __init__(self, **kwargs):
        if hasattr(self, 'commentary'):
            self.parents.commentary = self.commentary
            del self.commentary

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
        return LazyObject(compute_function=self._get_children, constrained_attrs=variables.CHILD_TYPES)

    @lazy_property
    def parents(self) -> LazyObject:
        return LazyObject(compute_function=self._get_parent, constrained_attrs=variables.TEXTCONTAINER_TYPES)

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

    def get_page(self, page_id: str) -> Optional['Page']:
        """A simple shortcut to get a page from its id."""
        return [p for p in self.children.pages if p.id == page_id][0]

    @lazy_property
    def olr_groundtruth_pages(self) -> List['CanonicalPage']:
        """A list of `CanonicalPage` objects containing the groundtruth of the OLR."""
        page_ids = get_olr_splits_page_ids(self.id)
        return [p for p in self.children.pages if p.id in page_ids]

    @lazy_property
    def ocr_groundtruth_pages(self) -> List['CanonicalPage']:
        """A list of `CanonicalPage` objects containing the groundtruth of the OCR."""
        ocr_gt = read_google_sheet(variables.SPREADSHEETS['ocr_gt'], 'ocr_gt')
        page_ids = ocr_gt['page_id'][ocr_gt['commentary_id'] == self.id].tolist()
        return [p for p in self.children.pages if p.id in page_ids]

    def export_ocr_gt_file_pairs(self,
                                 output_dir: Optional[Union[str, Path]] = None,
                                 unicode_format: str = 'NFC'):
        """Exports png-txt file pairs for each line in each ocr groundtruth page of the commentary.

        Args:
            output_dir: The directory to which the file pairs should be exported. If None, files are written to the
            default output directory (`variables.get_comm_ocr_gt_pairs_dir`).
            unicode_format: The unicode format to which the text should be normalized. See
            https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize for more information.
        """

        # Define output directory
        output_dir = variables.get_comm_ocr_gt_pairs_dir(self.id) if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over groundtruth pages
        for page in self.ocr_groundtruth_pages:
            for i, line in enumerate(page.children.lines):
                line.image.write(str(output_dir / f'{page.id}_{i}.png'))
                (output_dir / f'{page.id}_{i}.gt.txt').write_text(unicodedata.normalize(unicode_format, line.text),
                                                                  encoding='utf-8')


class Page:

    def draw_textcontainers(self,
                            tc_types: List[str] = variables.CHILD_TYPES,
                            output_path: Optional[Union[str, Path]] = None) -> 'np.ndarray':
        draw = self.image.matrix.copy()

        for type in tc_types:
            draw = draw_textcontainers(draw, *getattr(self.children, type))

        if output_path is not None:
            cv2.imwrite(str(output_path), draw)

        return draw

    @lazy_property
    def number(self) -> int:
        """The page number."""
        return int(self.id.split('_')[-1])

    # def to_canonical_v1(self) -> Dict[str, Any]:
    #     """Creates canonical data, as used for INCEpTION. """
    #     logger.warning(
    #         'You are creating a canonical data version 1. For version 2, use `OcrCommentary.to_canonical()`.')
    #     data = {'id': self.id,
    #             'iiif': 'None',
    #             'cdate': strftime('%Y-%m-%d %H:%M:%S'),
    #             'regions': []}
    #
    #     for r in self.children.regions:
    #         r_dict = {'region_type': r.region_type,
    #                   'bbox': list(r.bbox.xywh),
    #                   'lines': [
    #                       {
    #                           'bbox': list(l.bbox.xywh),
    #                           'words': [
    #                               {
    #                                   'bbox': list(w.bbox.xywh),
    #                                   'text': w.text
    #                               } for w in l.children.words
    #                           ]
    #
    #                       } for l in r.children.lines
    #                   ]
    #                   }
    #         data['regions'].append(r_dict)
    #
    #     return data

    # def to_xmi_json(self):

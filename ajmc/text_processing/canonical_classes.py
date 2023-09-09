"""This module contains objects for the manipulation of canonical textcontainers."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

from jinja2 import Environment, PackageLoader
from lazy_objects.lazy_objects import lazy_property, LazyObject

from ajmc.commons import variables as vs
from ajmc.commons.arithmetic import is_interval_within_interval
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.geometry import get_bbox_from_points, Shape
from ajmc.commons.image import AjmcImage
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.text_processing.generic_classes import Commentary, Page, TextContainer

logger = get_ajmc_logger(__name__)


class CanonicalCommentary(Commentary):

    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: Optional[str],
                 children: Optional['LazyObject'],
                 images: Optional[List[AjmcImage]],
                 ocr_run_id: Optional[str] = None,
                 ocr_gt_page_ids: Optional[List[str]] = None,
                 **kwargs):
        """Initialize a ``CanonicalCommentary``.

        Args:
            id: The id of the commentary.
            children: A ``LazyObject`` containing the children of the commentary. Can be manually set after init.
            images: A list of ``AjmcImage`` objects. Can be instantiated after init.
            info: A dictionary containing additional information about the commentary.
            kwargs: {kwargs_for_properties}

        Note:
            Parameters ``id``, ``children``, and ``images`` are required, as they cannot be computed ex nihilo. They can
            however be set ``None`` and then be computed later on.
        """
        super().__init__(id=id,
                         children=children,
                         images=images,
                         ocr_run_id=ocr_run_id,
                         ocr_gt_page_ids=ocr_gt_page_ids,
                         **kwargs)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]):
        """Instantiate a ``CanonicalCommentary`` from a json file.

        Args:
            json_path: The path to a canonical/v2 json file containing a commentary and respecting the
            ajmc folder structure.
        """
        json_path = Path(json_path)
        if not json_path.match(f'{vs.get_comm_canonical_dir("*") / "*.json"}'):
            logger.warning(f"The provided ``json_path`` ({json_path}) is not compliant with ajmc's folder structure.")

        logger.debug(f'Importing canonical commentary from {json_path}')
        can_json = json.loads(json_path.read_text(encoding='utf-8'))

        # Create the (empty) commentary and populate its info
        commentary = cls(id=can_json['id'],
                         children=None,
                         images=None,
                         ocr_run_id=can_json['ocr_run_id'],
                         ocr_gt_page_ids=can_json['ocr_gt_page_ids'])

        # Automatically determinates paths
        commentary.base_dir = vs.get_comm_base_dir(commentary.id)
        img_dir = vs.get_comm_img_dir(commentary.id)

        # Set its images
        commentary.images = [
            AjmcImage(id=img['id'], path=img_dir / (img['id'] + vs.DEFAULT_IMG_EXTENSION), word_range=img['word_range'])
            for img in can_json['children']['pages']
        ]

        # Set its children
        commentary.children = LazyObject(
                compute_function=lambda x: x,
                constrained_attrs=vs.CHILD_TYPES,
                **{tc_type: [get_tc_type_class(tc_type)(commentary=commentary, **tc)
                             for tc in can_json['children'][tc_type]]
                   for tc_type in vs.CHILD_TYPES})

        return commentary

    def to_json(self, output_path: Optional[Union[str, Path]] = None) -> dict:
        """Exports self to canonical json format.

        Args:
            output_path: The path to which the json should be exported. Leave empty to export to default location

        Returns:
            The json as a dictionary
        """

        data = {'id': self.id,
                'ocr_run_id': self.ocr_run_id,
                'ocr_gt_page_ids': self.ocr_gt_page_ids,
                'children': {child_type: [tc.to_json() for tc in getattr(self.children, child_type)]
                             for child_type in vs.CHILD_TYPES}}

        if output_path is None:
            output_path = vs.get_comm_canonical_default_path(self.id, self.ocr_run_id)

        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

        return data

    def to_alto(self,
                output_dir: Union[str, Path],
                children_types: List[str],
                region_types_mapping: Dict[str, str],
                region_types_ids: Dict[str, str],
                ocr_gt_only: bool = False,
                olr_gt_only: bool = False,
                copy_images: bool = False,
                region_types: List[str] = vs.ROIS):
        """A wrapper to export self.children.pages to alto.

        Args:
            output_dir: The directory to which the alto files should be exported.
            children_types: The types of children to export.
            region_types_mapping: A dictionary mapping the region types to the alto types, e.g. vs.REGION_TYPES_TO_SEGMONTO.
            region_types_ids: A dictionary mapping the region types to the alto ids, e.g. vs.SEGMONTO_TO_VALUE_IDS.
            ocr_gt_only: If True, only the ocr_gt pages will be exported. Notice that ocr_gt is a subset of olr_gt.
            olr_gt_only: If True, only the olr_gt pages will be exported.
            copy_images: If True, the images will be copied to the output directory.
            region_types: The types of regions to export.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_pages = self.olr_gt_pages if olr_gt_only else self.ocr_gt_pages if ocr_gt_only else self.children.pages
        for p in selected_pages:
            p.to_alto(output_path=output_dir / (p.id + '.xml'),
                      children_types=children_types,
                      region_types_mapping=region_types_mapping,
                      region_types_ids=region_types_ids,
                      regions_types=region_types)
            if copy_images:
                p.image.write(output_dir / (p.image.id + vs.DEFAULT_IMG_EXTENSION))

    def _get_children(self, children_type) -> List[Optional[Type['TextContainer']]]:
        raise NotImplementedError('``CanonicalCommentary.children`` must be set at __init__.')

    @lazy_property
    def ocr_gt_pages(self) -> List[Type['CanonicalPage']]:
        """A list of ``CanonicalPage`` objects containing the groundtruth of the OCR."""

        return [p for p in self.children.pages if p.id in self.ocr_gt_page_ids]


class CanonicalTextContainer(TextContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        pass

    def _get_children(self, children_type: str) -> List[Optional[Type['CanonicalTextContainer']]]:
        """Fetches the ``TextContainers`` in the parent commentary  which are included in ``self.text_container``.

        Note:
            - This methods works with word ranges, NOT with coordinates.
            - This methods does retrieve elements which overlap only partially with ``self``.
        """

        if self.type == 'word':  # Special efficiency hack for words
            return []

        if children_type == 'words':  # Special efficiency hack for words
            return self.parents.commentary.children.words[self.word_range[0]:self.word_range[1] + 1]

        # General case
        return [tc for tc in getattr(self.parents.commentary.children, children_type)
                if is_interval_within_interval(contained=tc.word_range, container=self.word_range)
                and self.id != tc.id]

    def _get_parent(self, parent_type: str) -> Optional[Type['CanonicalTextContainer']]:

        if parent_type == 'commentary':
            raise AttributeError('``parents.commentary`` cannot be computed ex nihilo. It must be set manually.')

        parents = [tc for tc in getattr(self.parents.commentary.children, vs.TC_TYPES_TO_CHILD_TYPES[parent_type])
                   if is_interval_within_interval(contained=self.word_range, container=tc.word_range)
                   and self.id != tc.id]

        return parents[0] if len(parents) > 0 else None

    @lazy_property
    def id(self) -> str:
        """Generic method to create a ``CanonicalTextContainer``'s id."""
        return self.type + '_' + str(self.index)

    @lazy_property
    def index(self) -> int:
        """Generic method to get a ``CanonicalTextContainer``'s index in its parent commentary's children list."""
        return getattr(self.parents.commentary.children, vs.TC_TYPES_TO_CHILD_TYPES[self.type]).index(self)

    @lazy_property
    def word_range(self) -> Tuple[int, int]:
        return self.word_range

    @lazy_property
    def bbox(self) -> Shape:
        """Generic method to get a ``CanonicalTextContainer``'s bbox."""
        if len(self.children.words) == 0:
            return Shape([(0, 0), (0, 0)])
        else:
            return Shape(get_bbox_from_points([xy for w in self.children.words for xy in w.bbox.bbox]))

    @lazy_property
    def image(self) -> AjmcImage:
        """Generic method to create a ``CanonicalTextContainer``'s image."""
        return self.parents.page.image.crop(self.bbox.bbox)


class CanonicalSection(CanonicalTextContainer):

    def __init__(self,
                 commentary,
                 section_types,
                 section_title,
                 **kwargs):
        super().__init__(commentary=commentary,
                         section_types=section_types,
                         section_title=section_title,
                         **kwargs)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'id': self.id,
                'section_types': self.section_types,
                'section_title': self.section_title,
                'word_range': self.word_range}


class CanonicalPage(Page, CanonicalTextContainer):

    def __init__(self, id: str, word_range: Tuple[int, int], commentary: CanonicalCommentary, **kwargs):
        super().__init__(id=id, word_range=word_range, commentary=commentary, **kwargs)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'id': self.id, 'word_range': self.word_range}

    def to_alto(self,
                output_path: Path,
                children_types: List[str],
                region_types_mapping: Dict[str, str],
                region_types_ids: Dict[str, str],
                regions_types: List[str] = vs.ROIS):
        """Exports a page to ALTO-xml.

        Args:
            output_path: self-explanatory.
            children_types: The types of children to be exported. Must be a subset of ['regions', 'lines', 'words'].
            region_types_mapping: A dictionary mapping the types of regions to the types of regions in the ALTO-xml, for instance variables.REGION_TYPES_TO_SEGMONTO
            region_types_ids: A dictionary mapping the values of ``region_types_mapping`` to the ids of regions in the ALTO-xml, for instance variables.REGION_TYPES_TO_SEGMONTO_ID
            regions_types: The types of regions to be exported, for instance variables.ROIS. This allows for filtering only the regions of interested, excluded regions types like 'undefined'.
        """
        env = Environment(loader=PackageLoader('ajmc', 'data/templates'),
                          trim_blocks=True,
                          lstrip_blocks=True,
                          autoescape=True)
        template = env.get_template('alto.xml.jinja2')

        # xml_formatter = xmlformatter.Formatter(indent="1", indent_char="\t", encoding_output="UTF-8", correct=True)
        alto_xml_data = template.render(page=self,
                                        children_types=children_types,
                                        region_types=regions_types,
                                        region_types_mapping=region_types_mapping,
                                        region_types_ids=region_types_ids)
        output_path.write_text(alto_xml_data, encoding='utf-8')
        # formatted_xml = xml_formatter.format_string(alto_xml_data.replace('\n', ""))
        # f.write(formatted_xml.decode('utf-8'))

    @lazy_property
    def image(self) -> AjmcImage:  # Special case of page's images
        return [img for img in self.parents.commentary.images if img.id == self.id][0]


class CanonicalRegion(CanonicalTextContainer):

    def __init__(self, word_range: Tuple[int, int], commentary: CanonicalCommentary, region_type: str, **kwargs):
        super().__init__(word_range=word_range, commentary=commentary, region_type=region_type, **kwargs)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'word_range': self.word_range, 'region_type': self.region_type}


class CanonicalLine(CanonicalTextContainer):

    def __init__(self, word_range: Tuple[int, int], commentary: CanonicalCommentary, **kwargs):
        super().__init__(word_range=word_range, commentary=commentary, **kwargs)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'word_range': self.word_range}


class CanonicalWord(CanonicalTextContainer):

    def __init__(self, text: str, bbox: Iterable[Iterable[int]], commentary: CanonicalCommentary, **kwargs):
        super().__init__(text=text, commentary=commentary, **kwargs)
        self.bbox = Shape(bbox)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'bbox': self.bbox.bbox, 'text': self.text}

    @lazy_property
    def word_range(self):
        return self.index, self.index


# todo 👁️ not very elegant. try to revise.
def get_tc_type_class(tc_type) -> Type[CanonicalTextContainer]:
    if not tc_type.endswith('s'):
        return globals()[f'Canonical{tc_type.capitalize()}']
    else:
        if tc_type.endswith('ies'):
            return globals()[f'Canonical{tc_type[:-3].capitalize()}y']
        else:
            return globals()[f'Canonical{tc_type[:-1].capitalize()}']


class CanonicalAnnotation(CanonicalTextContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def text(self):
        return ' '.join([w.text for w in self.children.words])[self.shifts[0]:self.shifts[1]]

    def get_text_window(self, window_size=5):
        return ' '.join(self.parents.commentary.children.words[
                        self.word_range[0] - window_size:self.word_range[1] + window_size + 1])

    def bbox(self) -> None:
        logger.warning('``CanonicalAnnotation``s have no bbox.')
        return None


class CanonicalEntity(CanonicalAnnotation):

    def __init__(self,
                 commentary: 'CanonicalCommentary',
                 word_range: Tuple[int, int],
                 shifts: Tuple[int, int],
                 transcript: Optional[str],
                 label: str,
                 wikidata_id: Optional[str]):
        super().__init__(word_range=word_range,
                         commentary=commentary,
                         shifts=shifts,
                         transcript=transcript,
                         label=label,
                         wikidata_id=wikidata_id)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {'word_range': self.word_range,
                'shifts': self.shifts,
                'transcript': self.transcript,
                'label': self.label,
                'wikidata_id': self.wikidata_id}

    def bbox(self) -> None:
        logger.warning('``CanonicalEntity``s have no bbox.')
        return None


class CanonicalSentence(CanonicalAnnotation):

    def __init__(self,
                 commentary: 'CanonicalCommentary',
                 word_range: Tuple[int, int],
                 shifts: Tuple[int, int],
                 corrupted: Optional[str],
                 incomplete_continuing: str,
                 incomplete_truncated: Optional[str]):
        super().__init__(word_range=word_range,
                         commentary=commentary,
                         shifts=shifts,
                         corrupted=corrupted,
                         incomplete_continuing=incomplete_continuing,
                         incomplete_truncated=incomplete_truncated)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int], bool]]:
        return {'word_range': self.word_range,
                'shifts': self.shifts,
                'corrupted': self.corrupted,
                'incomplete_continuing': self.incomplete_continuing,
                'incomplete_truncated': self.incomplete_truncated}

    def bbox(self) -> None:
        logger.warning('``CanonicalSentence``s have no bbox.')
        return None


class CanonicalHyphenation(CanonicalAnnotation):

    def __init__(self,
                 commentary: 'CanonicalCommentary',
                 word_range: Tuple[int, int],
                 shifts: Tuple[int, int]):
        super().__init__(word_range=word_range,
                         commentary=commentary,
                         shifts=shifts)

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int], bool]]:
        return {'word_range': self.word_range,
                'shifts': self.shifts}

    def bbox(self) -> None:
        logger.warning('``CanonicalHyphenation``s have no bbox.')
        return None

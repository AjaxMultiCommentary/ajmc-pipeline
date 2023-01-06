"""
This module handles OCR outputs. Notice that it has to cope with the inconsistencies and vagaries of OCR outputs. Hence,
eventhough the code is not very elegant, I would not recommend to change it without a very good reason and absolute
confidence in your changes.
"""

import re
import json
import os
from pathlib import Path
from time import strftime
from typing import Dict, Optional, List, Union, Any, Tuple, Type
import bs4.element
from abc import abstractmethod

from tqdm import tqdm

from ajmc.commons.miscellaneous import lazy_property, get_custom_logger, LazyObject
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons import variables
from ajmc.commons.geometry import (
    Shape,
    is_bbox_within_bbox_with_threshold,
    is_bbox_within_bbox, adjust_bbox_to_included_contours, get_bbox_from_points
)
from ajmc.commons.image import AjmcImage, draw_textcontainers
from ajmc.commons.variables import PATHS, CHILD_TYPES, ANNOTATION_LAYERS
from ajmc.olr.utils import sort_to_reading_order, get_page_region_dicts_from_via
import jsonschema

from ajmc.text_processing.canonical_classes import CanonicalCommentary, CanonicalWord, CanonicalPage, CanonicalRegion, \
    CanonicalLine, CanonicalEntity, CanonicalSentence, CanonicalHyphenation, CanonicalSection
from ajmc.text_processing import cas_utils
from ajmc.text_processing.generic_classes import Commentary, TextContainer, Page
from ajmc.text_processing.markup_processing import parse_markup_file, get_element_bbox, \
    get_element_text, find_all_elements
from ajmc.commons.file_management.utils import verify_path_integrity, parse_ocr_path, find_file_by_name, \
    guess_ocr_format

logger = get_custom_logger(__name__)


class OcrCommentary(Commentary, TextContainer):
    """`OcrCommentary` objects reprensent a single ocr-run of on a commentary, i.e. a collection of page OCRed pages."""

    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: Optional[str] = None,
                 ocr_dir: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 via_path: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 groundtruth_dir: Optional[str] = None,
                 ocr_run: Optional[str] = None,
                 **kwargs):
        """Default constructor, where custom paths can be provided.

        This is usefull when you want to instantiate a `OcrCommentary` without using `ajmc`'s folder structure. Note that
        only the requested paths must be provided. For instance, the object will be created if you do not provided the
        path to the images. But logically, you won't be able to access fonctionnalities that requires it (for instance
        `OcrCommentary.children.pages[0].image`).

        Args:
            id: The id of the commentary (e.g. sophoclesplaysa05campgoog).
            ocr_dir: {ocr_dir}
            via_path: {via_path}
            image_dir: {image_dir}
            groundtruth_dir: {groundtruth_dir}
        """
        super().__init__(id=id, ocr_dir=ocr_dir, base_dir=base_dir, via_path=via_path, image_dir=image_dir,
                         groundtruth_dir=groundtruth_dir, ocr_run=ocr_run, **kwargs)

    @classmethod
    @docstring_formatter(output_dir_format=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])
    def from_ajmc_structure(cls, ocr_dir: str):
        """Use this method to construct a `OcrCommentary`-object using ajmc's folder structure.

        Args:
            ocr_dir: Path to the directory in which ocr-outputs are stored. It should end with
            {output_dir_format}.
        """

        verify_path_integrity(path=ocr_dir, path_pattern=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])
        base_dir, id, ocr_run = parse_ocr_path(path=ocr_dir)

        return cls(id=id,
                   ocr_dir=ocr_dir,
                   base_dir=os.path.join(base_dir, id),
                   via_path=os.path.join(base_dir, id, variables.PATHS['via_path']),
                   image_dir=os.path.join(base_dir, id, variables.PATHS['png']),
                   groundtruth_dir=os.path.join(base_dir, id, variables.PATHS['groundtruth']),
                   ocr_run=ocr_run)

    def to_canonical(self, include_ocr_groundtruth: bool = True) -> CanonicalCommentary:
        """Export the commentary to a `CanonicalCommentary` object.

        Note:
            This pipeline must cope with the fact that the OCR may not be perfect. For instance, it may happen that a word
            is empty, or that coordinates are fuzzy. It hence relies on OcrPage.optimize() to fix these issues. Though this code
            is far from elegant, I wouldn't recommend touching it unless you are 100% sure of what you are doing.

        Returns:
            A `CanonicalCommentary` object.
        """

        # We start by creating an empty `CanonicalCommentary`
        can = CanonicalCommentary(id=self.id, children=None, images=[], info={})

        # We fill the metadata
        if hasattr(self, 'ocr_run'):  # todo, why should this be an if ?
            can.info['ocr_run'] = self.ocr_run
            can.info['base_dir'] = self.base_dir

        # We now populate the children and images
        children = {k: [] for k in CHILD_TYPES}
        w_count = 0
        if include_ocr_groundtruth:
            gt_ids = [p.id for p in self.ocr_groundtruth_pages]

        for i, p in enumerate(tqdm(self.children.pages, desc=f'Canonizing {can.id}')):

            if include_ocr_groundtruth and p.id in gt_ids:
                p = self.ocr_groundtruth_pages[gt_ids.index(p.id)]

            p.optimise()
            p_start = w_count
            for r in p.children.regions:
                r_start = w_count
                for l in r.children.lines:
                    l_start = w_count
                    for w in l.children.words:
                        w.index = w_count  # for later use with annotations
                        children['words'].append(CanonicalWord(text=w.text, bbox=w.bbox.bbox, commentary=can))
                        w_count += 1  # Hence w_count - 1 below

                    children['lines'].append(CanonicalLine(word_range=(l_start, w_count - 1), commentary=can))

                children['regions'].append(CanonicalRegion(word_range=(r_start, w_count - 1),
                                                           commentary=can,
                                                           region_type=r.region_type))

            children['pages'].append(CanonicalPage(id=p.id, word_range=(p_start, w_count - 1), commentary=can))

            # Adding images
            can.images.append(AjmcImage(id=p.id, path=Path(p.image_path), word_range=(p_start, w_count - 1)))

            # Adding entities
            for ent in p.children.entities:
                if ent.children.words:
                    children['entities'].append(
                        CanonicalEntity(word_range=(ent.children.words[0].index, ent.children.words[-1].index),
                                        commentary=can,
                                        shifts=ent.shifts,
                                        transcript=ent.transcript,
                                        label=ent.label,
                                        wikidata_id=ent.wikidata_id))
            # Adding sentences
            for s in p.children.sentences:
                if s.children.words:
                    children['sentences'].append(
                        CanonicalSentence(word_range=(s.children.words[0].index, s.children.words[-1].index),
                                          commentary=can,
                                          shifts=s.shifts,
                                          corrupted=s.corrupted,
                                          incomplete_continuing=s.incomplete_continuing,
                                          incomplete_truncated=s.incomplete_truncated))

            # Adding hyphenations
            for h in p.children.hyphenations:
                if h.children.words:
                    children['hyphenations'].append(
                        CanonicalHyphenation(word_range=(h.children.words[0].index, h.children.words[-1].index),
                                             commentary=can,
                                             shifts=h.shifts))

            p.reset()  # We reset the page to free up memory


        # Adding sections
        for s in self.children.sections:
            children['sections'].append(
                CanonicalSection(word_range=(s.children.words[0].index, s.children.words[-1].index),
                                 commentary=can,
                                 section_type=s.section_type,
                                 section_title=s.section_title))

        can.children = LazyObject((lambda x: x), constrained_attrs=CHILD_TYPES, **children)

        return can

    def _get_children(self, children_type):

        if children_type == 'pages':
            pages = []
            for file in [f for f in os.listdir(self.ocr_dir) if f[-4:] in ['.xml', 'hocr', 'html']]:
                pages.append(OcrPage(ocr_path=os.path.join(self.ocr_dir, file),
                                     id=file.split('.')[0],
                                     image_path=find_file_by_name(file.split('.')[0], self.image_dir),
                                     commentary=self))

            return sorted(pages, key=lambda x: x.id)

        # Todo : not implemented yet
        elif children_type == 'sections':
            # sections_path = Path(variables.PATHS['base_dir']) / self.id / 'sections.json'
            # sections = json.loads(sections_path.read_text(encoding='utf-8'))
            # return [RawSection(self, **s) for s in sections]
            return []


        else:  # For other children, them from each page
            return [tc for p in self.children.pages for tc in getattr(p.children, children_type)]


    @lazy_property  # Todo 👁️ This should not be maintained anymore
    def ocr_groundtruth_pages(self) -> Union[List['OcrPage'], list]:
        """The commentary's pages which have a groundtruth file in `self.paths['groundtruth']`."""
        pages = []
        for file in [f for f in os.listdir(self.groundtruth_dir) if f.endswith('.html')]:
            pages.append(OcrPage(ocr_path=os.path.join(self.groundtruth_dir, file),
                                 id=file.split('.')[0],
                                 image_path=find_file_by_name(file.split('.')[0], self.image_dir),
                                 commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def via_project(self) -> dict:
        with open(self.via_path, 'r') as file:
            return json.load(file)

    @lazy_property
    def images(self) -> List[AjmcImage]:
        return [p.image for p in self.children.pages]


class OcrPage(Page, TextContainer):
    """A class representing a commentary page."""

    def __init__(self,
                 ocr_path: str,
                 id: Optional[str] = None,
                 image_path: Optional[str] = None,
                 commentary: Optional[OcrCommentary] = None,
                 **kwargs):
        super().__init__(ocr_path=ocr_path, id=id, image_path=image_path, commentary=commentary, **kwargs)

    def _get_children(self, children_type):
        if children_type == 'regions':
            return [OlrRegion.from_via(via_dict=r, page=self)
                    for r in get_page_region_dicts_from_via(self.id, self.parents.commentary.via_project)]

        # Lines and words must be retrieved together
        elif children_type in ['lines', 'words']:
            w_count = 0
            lines = []
            words = []
            for l_markup in find_all_elements(self.markup, 'line', self.ocr_format):
                line = OcrLine(markup=l_markup, page=self)
                line.word_ids = []
                for w_markup in find_all_elements(l_markup, 'word', self.ocr_format):
                    line.word_ids.append(w_count)
                    words.append(OcrWord(id=w_count, markup=w_markup, page=self))
                    w_count += 1
                lines.append(line)

            self.children.words = words
            self.children.lines = lines

            return getattr(self.children, children_type)

        elif children_type in ['entities', 'sentences', 'hyphenations']:
            try:
                rebuild = cas_utils.import_page_rebuild(self.id)
            except:
                logger.warning(f'No rebuild file found for page {self.id}')
                return []
            cas = cas_utils.import_page_cas(self.id)
            if cas is not None:
                annotations = cas_utils.safe_import_page_annotations(self.id, cas, rebuild,
                                                                     ANNOTATION_LAYERS[children_type])

                if children_type == 'entities':
                    return [RawEntity.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]
                elif children_type == 'sentences':
                    return [RawSentence.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]
                elif children_type == 'hyphenations':
                    return [RawHyphenation.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]

            else:  # page has no CAS
                return []

        else:
            return []

    def _get_parent(self, parent_type):
        raise NotImplementedError('OcrPage.parents must be set at __init__ or manually.')

    def to_canonical_v1(self) -> Dict[str, Any]:
        """Creates canonical data, as used for INCEpTION. """
        logger.warning(
            'You are creating a canonical data version 1. For version 2, use `OcrCommentary.to_canonical()`.')
        data = {'id': self.id,
                'iiif': 'None',
                'cdate': strftime('%Y-%m-%d %H:%M:%S'),
                'regions': []}

        for r in self.children.regions:
            r_dict = {'region_type': r.region_type,
                      'bbox': list(r.bbox.xywh),
                      'lines': [
                          {
                              'bbox': list(l.bbox.xywh),
                              'words': [
                                  {
                                      'bbox': list(w.bbox.xywh),
                                      'text': w.text
                                  } for w in l.children.words
                              ]

                          } for l in r.children.lines
                      ]
                      }
            data['regions'].append(r_dict)

        return data

    def to_json(self, output_dir: str, schema_path: str = variables.PATHS['schema']):
        """Validate `self.to_canonical_v1` and serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        jsonschema.validate(instance=self.to_canonical_v1(), schema=schema)

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(self.to_canonical_v1(), f, indent=4, ensure_ascii=False)

    def reset(self):
        """Resets the page to free up memory."""
        delattr(self, 'children')
        delattr(self, 'image')
        delattr(self, 'text')

    def optimise(self, debug_dir: Optional[str] = None):
        """Optimises coordinates and reading order.

        Args:
            debug_dir: If given, the page will be saved to this directory for debugging purposes.

        Note:
            - This function changes the page in place.
            - Like `OcrCommentary.to_canonical`, this function must cope with the vagaries of the OCR output. Though its
            code is far from slick, I wouldn't recommend trying to improve it unless you are 100% sure that you know what
            you are doing.
        """

        if debug_dir is not None:
            _ = draw_textcontainers(self.image.matrix.copy(), self,
                                    os.path.join(debug_dir, self.id + '_raw.png'))

        logger.warning("You are optimising a page, bboxes and children are going to be changed")
        self.reset()

        # Process words
        self.children.words = [w for w in self.children.words if re.sub(r'\s+', '', w.text) != '']
        for w in self.children.words:
            w.text = w.text.strip()  # Remove leading and trailing whitespace (happens sometimes)
            w.adjust_bbox()

        # Process lines
        self.children.lines = [l for l in self.children.lines if l.children.words]
        for l in self.children.lines:
            l.adjust_bbox()

        # Process regions
        self.children.regions = [r for r in self.children.regions
                                 if r.region_type not in ['undefined', 'line_number_commentary']
                                 and r.children.words]
        for r in self.children.regions:
            r.adjust_bbox()

        # Cut lines according to regions
        for r in self.children.regions:
            r.children.lines = []

            for l in self.children.lines:
                # If the line is entirely in the region, append it
                if is_bbox_within_bbox(contained=l.bbox.bbox,
                                       container=r.bbox.bbox):
                    l.region = r  # Link the line to its region
                    r.children.lines.append(l)

                # If the line is only partially in the region, handle the line splitting problem.
                elif any([is_bbox_within_bbox(w.bbox.bbox, r.bbox.bbox)
                          for w in l.children.words]):

                    # Create the new line and append it both to region and page lines
                    new_line = OcrLine(markup=None,
                                       page=self,
                                       word_ids=[w.id for w in l.children.words
                                                 if is_bbox_within_bbox(w.bbox.bbox, r.bbox.bbox)])
                    new_line.adjust_bbox()
                    new_line.region = r
                    r.children.lines.append(new_line)

                    # Actualize the old line
                    l.children.words = [w for w in l.children.words
                                        if w.id not in new_line.word_ids]
                    l.adjust_bbox()

            r.children.lines.sort(key=lambda l: l.bbox.xywh[1])

        # Actualize global page reading order
        ## Create fake regions for lines with no regions
        for l in self.children.lines:
            if not hasattr(l, 'region'):
                line_region = OlrRegion(region_type='line_region',
                                        bbox=Shape(l.bbox.bbox),
                                        page=self)
                line_region.children.lines = [l]
                self.children.regions.append(line_region)

        self.children.regions = sort_to_reading_order(elements=self.children.regions)
        self.children.lines = [l for r in self.children.regions for l in r.children.lines]
        self.children.words = [w for l in self.children.lines for w in l.children.words]

        if debug_dir:
            _ = draw_textcontainers(self.image.matrix.copy(), self,
                                    os.path.join(debug_dir, self.id + '_raw.png'))

    @lazy_property
    def image(self) -> AjmcImage:
        return AjmcImage(id=self.id, path=Path(self.image_path))

    @lazy_property
    def markup(self) -> bs4.BeautifulSoup:
        return parse_markup_file(self.ocr_path)

    @lazy_property
    def ocr_format(self) -> str:
        return guess_ocr_format(self.ocr_path)

    @lazy_property
    def bbox(self) -> Shape:
        return Shape(get_bbox_from_points([xy for w in self.children.words for xy in w.bbox.bbox]))


class RawSection(TextContainer):

    def __init__(self,
                 commentary,
                 section_type: str,
                 section_title: str,
                 start: int,
                 end: int,
                 **kwargs):

        super().__init__(commentary=commentary,
                         section_type=section_type,
                         section_title=section_title,
                         start=start,
                         end=end,
                         **kwargs)

    @classmethod
    def from_json(cls, json_dict: dict):

        return cls(**json_dict)

    def _get_children(self, children_type) -> List[Optional[Type['TextContainer']]]:
        if children_type == 'pages':
            return [p for p in self.parents.commentary.children.pages
                    if self.start >= p.number <= self.end]

        else:
            return [child for p in self.children.pages for child in getattr(p.children, children_type)]

    def _get_parent(self, parent_type) -> Optional[Type['TextContainer']]:
        if parent_type == 'commentary':
            return self.parents.commentary
        else:
            return None


class OcrTextContainer(TextContainer):

    def __init__(self, **kwargs):
        """Initialize a `OcrTextContainer` with provided kwargs

        Args:
            **kwargs: Use kwargs to pass any desired attribute or to manually set the values of properties.
        """
        super().__init__()
        for k, v in kwargs.items():
            if k == 'page':
                self.parents.page = v
                self.parents.commentary = self.parents.page.parents.commentary
            else:
                setattr(self, k, v)

    @abstractmethod
    def _get_children(self, children_type):
        pass

    def _get_parent(self, parent_type):
        if parent_type not in ['commentary', 'page']:  # else: is provided in init
            raise NotImplementedError(f'`OcrTextContainer.parents` only supports `commentary` and `page`, '
                                      f'not {parent_type}. Build `CanonicalTextContainer.parents` instead.')

    def adjust_bbox(self):
        words_points = [xy for w in self.children.words for xy in w.bbox.bbox]
        self.bbox = Shape(get_bbox_from_points(words_points))

    @lazy_property
    def bbox(self):
        return get_element_bbox(self.markup, self.ocr_format)

    @lazy_property
    def image(self) -> AjmcImage:
        return self.parents.page.image

    @lazy_property
    def ocr_format(self) -> str:
        return self.parents.page.ocr_format

    @lazy_property
    def text(self) -> str:
        return ' '.join([w.text for w in self.children.words])


class OlrRegion(OcrTextContainer):

    @docstring_formatter(**docstrings)
    def __init__(self,
                 region_type: str,
                 bbox: Shape,
                 page: 'OcrPage'):
        """Default constructor.

        Args:
            region_type: {olr_region_type}
            bbox: {coords_single}
            page: {parent_page}
        """
        super().__init__(region_type=region_type, bbox=bbox, page=page)
        self._inclusion_threshold = variables.PARAMETERS['ocr_region_inclusion_threshold']

    @classmethod
    @docstring_formatter(**docstrings)
    def from_via(cls, via_dict: Dict[str, dict], page: 'OcrPage'):
        """Constructs a region directly from its corresponding `via_dict`.

        Args:
            via_dict: {via_dict}
            page: {parent_page}
        """
        return cls(region_type=via_dict['region_attributes']['text'],
                   bbox=Shape.from_xywh(x=via_dict['shape_attributes']['x'],
                                        y=via_dict['shape_attributes']['y'],
                                        w=via_dict['shape_attributes']['width'],
                                        h=via_dict['shape_attributes']['height']),
                   page=page)

    def _get_children(self, children_type):
        return [el for el in getattr(self.parents.page.children, children_type)
                if is_bbox_within_bbox_with_threshold(contained=el.bbox.bbox,
                                                      container=self.bbox.bbox,
                                                      threshold=self._inclusion_threshold)]


class OcrLine(OcrTextContainer):
    """Class for OCR lines."""

    def __init__(self,
                 markup: 'bs4.element.Tag',
                 page: OcrPage,
                 word_ids: Optional[List[Union[str, int]]] = None,
                 **kwargs):
        super().__init__(markup=markup, page=page, word_ids=word_ids, **kwargs)

    def _get_children(self, children_type):
        return [w for w in self.parents.page.children.words if
                w.id in self.word_ids] if children_type == 'words' else []


class OcrWord(OcrTextContainer):
    """Class for ocr words."""

    def __init__(self,
                 id: Union[int, str],
                 markup: 'bs4.element.Tag',
                 page: OcrPage,
                 **kwargs):
        super().__init__(id=id, markup=markup, page=page, **kwargs)

    def _get_children(self, children_type: str):
        return []  # Words have no children

    @lazy_property
    def text(self):
        return get_element_text(element=self.markup, ocr_format=self.ocr_format)

    def adjust_bbox(self):
        self.bbox = adjust_bbox_to_included_contours(self.bbox.bbox, self.parents.page.image.contours)


class RawAnnotation(TextContainer):

    @docstring_formatter(**docstrings)
    def __init__(self,
                 page: 'OcrPage',
                 bboxes: List['Shape'],
                 shifts: Tuple[int, int],
                 text_window: str,
                 warnings: List[str],
                 **kwargs
                 ):
        """Default constructor for annotation.

        Though it can be used directly, it is usually called via `from_cas_annotation` class method instead.
        `kwargs` are used to pass any desired attribute or to manually set the values of properties and to
        pass subclass-specific attributes, such as label for entities or `corrputed` for gold sentences.

        Args:
            page: {parent_page}
            bboxes: A list of `Shape` objects representing the bounding boxes of the annotation.
            shifts: A tuple of two integers representing the shifts of the annotation wrt its word.
            text_window: A string representing the text window of the annotation.
            warnings: A list of strings representing the warnings of the annotation.
        """
        self.parents.page = page
        self.parents.commentary = self.parents.page.parents.commentary
        super().__init__(bboxes=bboxes, shifts=shifts, text_window=text_window, warnings=warnings, **kwargs)

    def _get_parent(self, parent_type):
        return OcrTextContainer._get_parent(self, parent_type)

    def _get_children(self, children_type):
        """Returns the children of the annotation, coping with the fact that annotation have multiple bboxes."""
        return [c for c in getattr(self.parents.page.children, children_type)
                if any([is_bbox_within_bbox_with_threshold(contained=c.bbox.bbox,
                                                           container=bbox.bbox,
                                                           threshold=variables.PARAMETERS['entity_inclusion_threshold'])
                        for bbox in self.bboxes])]


class RawEntity(RawAnnotation):
    """Class for cas imported entities."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                               rebuild=rebuild, verbose=verbose)
        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   transcript=cas_annotation.transcript,
                   label=cas_annotation.value,
                   wikidata_id=cas_annotation.wikidata_id,
                   text_window=text_window,
                   warnings=warnings)


class RawSentence(RawAnnotation):
    """Class for cas imported gold sentences."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                               rebuild=rebuild, verbose=verbose)
        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   text_window=text_window,
                   warnings=warnings,
                   corrupted=cas_annotation.corrupted,
                   incomplete_continuing=cas_annotation.incomplete_continuing,
                   incomplete_truncated=cas_annotation.incomplete_truncated)


class RawHyphenation(RawAnnotation):
    """Class for cas imported hyphenations."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                               rebuild=rebuild, verbose=verbose)
        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   text_window=text_window,
                   warnings=warnings)

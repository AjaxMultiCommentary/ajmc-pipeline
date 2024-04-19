"""
This module handles OCR outputs. Notice that it has to cope with the inconsistencies and vagaries of OCR outputs. Hence,
eventhough the code is not very elegant, I would not recommend to change it without a very good reason and absolute
confidence in your changes.
"""

import json
import re
from abc import abstractmethod
from pathlib import Path
from time import strftime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import bs4.element
import jsonschema
from lazy_objects.lazy_objects import lazy_property, LazyObject
from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.file_management import get_commit_hash
from ajmc.commons.geometry import adjust_bbox_to_included_contours, get_bbox_from_points, is_bbox_within_bbox, \
    is_bbox_within_bbox_with_threshold, Shape, are_bboxes_overlapping_with_threshold
from ajmc.commons.image import AjmcImage
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.olr.utils import sort_to_reading_order
from ajmc.text_processing import cas_utils
from ajmc.text_processing.canonical_classes import CanonicalCommentary, CanonicalEntity, CanonicalHyphenation, \
    CanonicalLine, CanonicalPage, CanonicalRegion, CanonicalSection, CanonicalSentence, CanonicalWord, CanonicalLemma
from ajmc.text_processing.generic_classes import Commentary, Page, TextContainer
from ajmc.text_processing.markup_processing import find_all_elements, get_element_bbox, get_element_text
from ajmc.text_processing.via import ViaProject

logger = get_ajmc_logger(__name__)


class RawTextContainer(TextContainer):

    @abstractmethod
    def _get_children(self, children_type):
        pass

    def _get_parent(self, parent_type):
        raise NotImplementedError('Parents to ``RawTextContainer``s are not implemented. They can only be accessed if provided at __init__.')

    def adjust_bbox(self):
        self.bbox = Shape(get_bbox_from_points([xy for w in self.children.words for xy in w.bbox.bbox]))

    @lazy_property
    def text(self) -> str:
        return ' '.join([w.text for w in self.children.words])


class RawCommentary(Commentary):
    """``RawCommentary`` objects reprensent a single ocr-run of on a commentary, i.e. a collection of page OCRed pages."""

    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: str,
                 ocr_run_id: str = vs.DEFAULT_OCR_RUN_ID,
                 **kwargs):
        """Default constructor..

        Note:
             Useful paths are defined as properties and comply to ajmc's folder structure. Ifyou want to instantiate a
             ``RawCommentary`` without using ``ajmc``'s folder structure, you can overwrite these properties with ``**kwargs``.

        Args:
            ocr_dir: {ocr_dir}
            ocr_run_id: {ocr_run_id}
        """

        super().__init__(id=id, ocr_run_id=vs.get_ocr_run_id_from_pattern(id, ocr_run_id), **kwargs)

    def to_canonical(self) -> CanonicalCommentary:
        """Export the commentary to a ``CanonicalCommentary`` object.

        Note:
            This pipeline must cope with the fact that the OCR may not be perfect. For instance, it may happen that a word
            is empty, or that coordinates are fuzzy. It hence relies on RawPage.optimize() to fix these issues. Though this code
            is far from elegant, I wouldn't recommend touching it unless you are 100% sure of what you are doing.

        Returns:
            A ``CanonicalCommentary`` object.
        """
        self.reset()

        # We start by creating an empty ``CanonicalCommentary``
        can = CanonicalCommentary(id=self.id,
                                  children=None,
                                  images=[],
                                  ocr_run_id=self.ocr_run_id,
                                  ocr_gt_page_ids=self.ocr_gt_page_ids,
                                  olr_gt_page_ids=self.olr_gt_page_ids,
                                  ner_gt_page_ids=self.ner_gt_page_ids,
                                  lemlink_gt_page_ids=self.lemlink_gt_page_ids,
                                  metadata=self.metadata, )

        # We now populate the children and images
        children = {k: [] for k in vs.CHILD_TYPES}
        w_count = 0
        for section in self.children.sections:
            section_start = w_count
            for page in tqdm(section.children.pages, desc=f'Canonizing section {section.section_title}...'):
                page = page.get_ocr_gt_page()
                page.optimise()
                p_start = w_count
                for r in page.children.regions:
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
                                                               region_type=r.region_type,
                                                               is_ocr_gt=r.is_ocr_gt))

                children['pages'].append(CanonicalPage(id=page.id, word_range=(p_start, w_count - 1), commentary=can))

                # Adding images
                can.images.append(AjmcImage(id=page.id, path=Path(page.img_path), word_range=(p_start, w_count - 1)))

                # Adding entities
                for ent in page.children.entities:
                    if ent.children.words:
                        children['entities'].append(
                                CanonicalEntity(word_range=(ent.children.words[0].index, ent.children.words[-1].index),
                                                commentary=can,
                                                shifts=ent.shifts,
                                                transcript=ent.transcript,
                                                label=ent.label,
                                                wikidata_id=ent.wikidata_id))
                    else:
                        print(f'WARNING: NO WORDS. {page.id} ent {ent.transcript}')  # Todo remove
                # Adding sentences
                for s in page.children.sentences:
                    if s.children.words:
                        children['sentences'].append(
                                CanonicalSentence(word_range=(s.children.words[0].index, s.children.words[-1].index),
                                                  commentary=can,
                                                  shifts=s.shifts,
                                                  corrupted=s.corrupted,
                                                  incomplete_continuing=s.incomplete_continuing,
                                                  incomplete_truncated=s.incomplete_truncated))
                    else:
                        print(f'WARNING: NO WORDS. {page.id} sentence')  # Todo remove

                # Adding hyphenations
                for h in page.children.hyphenations:
                    if h.children.words:
                        children['hyphenations'].append(
                                CanonicalHyphenation(word_range=(h.children.words[0].index, h.children.words[-1].index),
                                                     commentary=can,
                                                     shifts=h.shifts))
                    else:
                        print(f'WARNING: NO WORDS. {page.id} hyphen')  # Todo remove

                for l in page.children.lemmas:
                    if l.children.words:
                        children['lemmas'].append(CanonicalLemma(word_range=(l.children.words[0].index, l.children.words[-1].index),
                                                                 commentary=can,
                                                                 shifts=l.shifts,
                                                                 label=l.value,
                                                                 transcript=l.transcript,
                                                                 anchor_target=l.anchor_target))
                    else:
                        print(f'WARNING: NO WORDS. {page.id} lemma {l.transcript}')  # Todo remove

                # We reset the page to free up memory
                page.reset()

            # Adding sections
            children['sections'].append(
                    CanonicalSection(word_range=(section_start, w_count - 1),
                                     commentary=can,
                                     section_types=section.section_types,
                                     section_title=section.section_title))

        # We now populate the children of the commentary
        can.children = LazyObject((lambda x: x), constrained_attrs=vs.CHILD_TYPES, **children)

        # We reset the commentary to free up memory
        self.reset()

        return can

    def _get_children(self, children_type):

        if children_type == 'pages':
            pages = [RawPage(ocr_path=p, commentary=self) for p in self.ocr_dir.glob('*') if p.suffix in vs.OCR_OUTPUTS_EXTENSIONS]
            return sorted(pages, key=lambda x: x.id)

        elif children_type == 'sections':
            sections = json.loads(self.sections_path.read_text(encoding='utf-8'))
            return [RawSection(self, **s) for s in sections]

        else:  # For other children, retrieve them from each page
            return [tc for p in self.children.pages for tc in getattr(p.children, children_type)]

    @lazy_property
    def ocr_dir(self) -> Path:
        """The directory containing the ocr files."""
        return vs.get_comm_ocr_outputs_dir(self.id, self.ocr_run_id)

    @lazy_property
    def via_path(self) -> Path:
        """The path to the commentary's VIA project."""
        return vs.get_comm_via_path(self.id)

    @lazy_property
    def sections_path(self) -> Path:
        """The path to the commentary's sections file."""
        return vs.get_comm_sections_path(self.id)


    @lazy_property
    def ocr_gt_page_ids(self) -> List[str]:
        """The ids of the commentary's pages which have a groundtruth file in ``self.ocr_gt_dir``."""
        return sorted([Path(p['filename']).stem for p in self.via_project['_via_img_metadata'].values()
                       if p['file_attributes']['is_ground_truth'].get('ocr', False)])


    @lazy_property
    def ocr_gt_pages(self) -> Union[List['RawPage'], list]:
        """The commentary's pages which have a ocr groundtruth."""
        return sorted([RawPage(via_dict=p, commentary=self) for p in self.via_project['_via_img_metadata'].values()
                       if Path(p['filename']).stem in self.ocr_gt_page_ids], key=lambda x: x.id)

    @lazy_property
    def ocr_gt_partial_page_ids(self) -> List[str]:
        """The ids of the commentary's pages which have partial ocr groundtruth."""
        ocr_gt_partial_page_ids = []
        for p in self.via_project['_via_img_metadata'].values():
            if (any([vs.OCR_GT_PREFIX in r['region_attributes']['label'] for r in p['regions']])
                    and Path(p['filename']).stem not in self.ocr_gt_page_ids):
                ocr_gt_partial_page_ids.append(Path(p['filename']).stem)

        return sorted(ocr_gt_partial_page_ids)


    @lazy_property
    def ocr_gt_partial_pages(self) -> Union[List['RawPage'], list]:
        """The commentary's pages which have a partial groundtruth file."""
        partial_groundtruth_pages = []
        for p in self.via_project['_via_img_metadata'].values():
            if Path(p['filename']).stem in self.ocr_gt_partial_page_ids:
                partial_groundtruth_pages.append(RawPage(via_dict=p, commentary=self))

        return sorted(partial_groundtruth_pages, key=lambda x: x.id)

    @lazy_property
    def olr_gt_page_ids(self) -> List[str]:
        """A list of page ids containing the groundtruth of the OLR."""
        return sorted([Path(p['filename']).stem for p in self.via_project['_via_img_metadata'].values()
                       if p['file_attributes']['is_ground_truth'].get('olr', False)])


    @lazy_property
    def ner_gt_page_ids(self) -> List[str]:
        """A list of page ids containing the groundtruth of the NER."""
        return sorted([p.stem for p in vs.NE_CORPUS_DIR.rglob(f'**/curated/{self.id}*.xmi')])


    @lazy_property
    def lemlink_gt_page_ids(self) -> List[str]:
        """A list of page ids containing the groundtruth of the lemmatization."""

        return sorted([p.stem for p in vs.LEMLINK_XMI_DIR.rglob(f'{self.id}*.xmi')])


    @lazy_property
    def metadata(self) -> Dict[str, str]:
        """The metadata of the commentary."""
        return {'ne_corpus_commit': get_commit_hash(vs.NE_CORPUS_DIR),
                'ocr_run_id': self.ocr_run_id,
                'lemlink_corpus_commit': get_commit_hash(vs.LEMLINK_XMI_DIR),
                'commentaries_data_commit': get_commit_hash(vs.COMMS_DATA_DIR)}


    @lazy_property
    def via_project(self) -> ViaProject:
        """The VIA project of the commentary.

        Note:
            This returns a ViaProject object, which is a wrapper around the VIA project file and behaves
            like a dictionary with some additional methods.
        """
        return ViaProject.from_json(self.via_path)


    @lazy_property
    def images(self) -> List[AjmcImage]:
        return [p.image for p in self.children.pages]


    def reset(self):
        """Resets the commentary. Use this if your commentary has been modified"""
        delattr(self, 'children')
        delattr(self, 'images')
        delattr(self, 'text')
        delattr(self, 'via_project')
        delattr(self, 'ocr_gt_page_ids')
        delattr(self, 'ocr_gt_pages')
        delattr(self, 'ocr_gt_partial_page_ids')
        delattr(self, 'ocr_gt_partial_pages')
        delattr(self, 'olr_gt_page_ids')


    def include_ocr_gt(self):
        """Includes the available OCR groundtruth to the commentary."""
        logger.info(f'Replacing OCR outputs with available groundtruth in {self.id}')
        self.children.pages = [p.get_ocr_gt_page() for p in self.children.pages]


class RawSection(TextContainer):

    def __init__(self,
                 commentary,
                 section_types: List[str],
                 section_title: str,
                 start: Union[int, str],
                 end: Union[int, str],
                 **kwargs):

        super().__init__(commentary=commentary,
                         section_types=section_types,
                         section_title=section_title,
                         start=int(start),
                         end=int(end),
                         **kwargs)

    def _get_children(self, children_type) -> List[Optional[Type['TextContainer']]]:
        if children_type == 'pages':
            return [p for p in self.parents.commentary.children.pages if self.start <= p.number <= self.end]

        else:
            return [child for p in self.children.pages for child in getattr(p.children, children_type)]

    def _get_parent(self, parent_type) -> Optional[Type['TextContainer']]:
        raise NotImplementedError('Parents to ``RawSection``s are not implemented. They can only be accessed if provided at __init__.')


class RawPage(Page, TextContainer):
    """A class representing a commentary page."""

    def __init__(self,
                 commentary: RawCommentary,
                 ocr_path: Optional[Path] = None,
                 **kwargs):

        super().__init__(ocr_path=ocr_path,
                         commentary=commentary,
                         is_optimised=False,
                         is_from_ocr=ocr_path is not None,
                         **kwargs)


    def _get_children(self, children_type):
        """Returns the children of the page of the given type.

        Note:
            If ``children_type`` is set to one of 'regions', 'lines' or 'words', this method will only be called if the page is not coming from via.
        """
        if children_type == 'regions':

            regions = []
            for r in self.via_dict['regions']:
                if (r['region_attributes']['label'].startswith(vs.OLR_PREFIX)
                        and not any([t in r['region_attributes']['label'] for t in vs.EXCLUDED_REGION_TYPES])):
                    regions.append(RawRegion.from_via(via_dict=r, page=self))
            return regions

        # Lines and words must be retrieved together
        elif children_type in ['lines', 'words']:
            if self.is_from_ocr:
                w_count = 0
                lines = []
                words = []
                for l_markup in find_all_elements(self.markup, 'line', self.ocr_format):
                    line = RawLine(page=self, word_ids=[])
                    for w_markup in find_all_elements(l_markup, 'word', self.ocr_format):
                        line.word_ids.append(w_count)
                        words.append(RawWord(id=w_count,
                                             text=get_element_text(element=w_markup, ocr_format=self.ocr_format),
                                             page=self,
                                             line=line,
                                             bbox=get_element_bbox(w_markup, self.ocr_format)))
                        w_count += 1
                    lines.append(line)

                self.children.words = words
                self.children.lines = lines

                return getattr(self.children, children_type)

            else:  # From VIA
                if children_type == 'lines':
                    return [RawLine(page=self, bbox=Shape.from_via(r)) for r in self.via_dict['regions']
                            if r['region_attributes']['label'].startswith(vs.OLR_PREFIX) and 'line_region' in r['region_attributes']['label']]
                else:
                    return [RawWord.from_via(id=i, via_dict=r, page=self) for i, r in enumerate(self.via_dict['regions'])
                            if not r['region_attributes']['label'].startswith(vs.OLR_PREFIX)]


        elif children_type in ['entities', 'sentences', 'hyphenations']:
            try:
                rebuild = cas_utils.import_page_rebuild(self.id, annotation_type=children_type)
            except:
                logger.debug(f'Looking for {children_type}: No rebuild file found for page {self.id}')
                return []
            cas = cas_utils.import_page_cas(self.id, children_type)
            if cas is not None:
                annotations = cas_utils.safe_import_page_annotations(self.id, cas, rebuild,
                                                                     vs.ANNOTATION_LAYERS[children_type])

                if children_type == 'entities':
                    return [RawEntity.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]
                elif children_type == 'sentences':
                    return [RawSentence.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]
                elif children_type == 'hyphenations':
                    return [RawHyphenation.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]
            else:
                return []

        elif children_type == 'lemmas':
            try:
                rebuild = cas_utils.import_page_rebuild(self.id, annotation_type=children_type)
            except:
                logger.debug(f'Looking for {children_type}: No rebuild file found for page {self.id}')
                return []

            cas = cas_utils.import_page_cas(self.id, children_type)

            if cas is not None:
                annotations = cas_utils.safe_import_page_annotations(page_id=self.id,
                                                                     cas=cas,
                                                                     rebuild=rebuild,
                                                                     annotation_layer=vs.ANNOTATION_LAYERS[children_type])

                return [RawLemma.from_cas_annotation(self, cas_ann, rebuild) for cas_ann in annotations]

            else:
                return []

        else:
            return []

    def _get_parent(self, parent_type):
        raise NotImplementedError('Parents to ``RawPage``s are not implemented. They can only be accessed if provided at __init__.')

    def to_inception_dict(self) -> Dict[str, Any]:
        """Creates canonical data, as used for INCEpTION. """
        data = {'iiif': 'None',
                'id': self.id,
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

    def to_inception_json(self, output_dir: Path, schema_path: Path = vs.SCHEMA_PATH):
        """Validate ``self.to_inception_dict`` and serializes it to json."""
        inception_dict = self.to_inception_dict()
        schema = json.loads(schema_path.read_text('utf-8'))
        jsonschema.validate(instance=inception_dict, schema=schema)
        ((output_dir / self.id).with_suffix('.json')).write_text(
                json.dumps(inception_dict, indent=4, ensure_ascii=False), encoding='utf-8')


    def reset(self):
        """Resets the page to free up memory."""
        delattr(self, 'children')
        delattr(self, 'image')
        delattr(self, 'text')

    def optimise(self, debug_dir: Optional[Path] = None):
        """Optimises coordinates and reading order.

        Warning:
            This function changes the page in place.

        Warning:
            Like ``RawCommentary.to_canonical``, this function must cope with the vagaries of the OCR output. Though its
            code is far from slick, I wouldn't recommend trying to improve it unless you are 100% sure that you know what
            you are doing.

        Args:
            debug_dir: If given, the page will be saved to this directory for debugging purposes.
        """
        if self.is_optimised:
            return

        if debug_dir is not None:
            _ = self.draw_textcontainers(output_path=debug_dir / f'{self.id}_raw.png')

        logger.info("You are optimising a page, bboxes and children are going to be changed")
        self.reset()

        # Process words
        self.children.words = [w for w in self.children.words if re.sub(r'\s+', '', w.text) != '']
        clean_words = []
        clean_word_bboxes = []
        for w in self.children.words:
            w.text = w.text.strip()  # Remove leading and trailing whitespace (happens sometimes)
            w.adjust_bbox()
            if w.bbox.bbox not in clean_word_bboxes:
                clean_words.append(w)
                clean_word_bboxes.append(w.bbox.bbox)
        self.children.words = clean_words

        # Process lines
        clean_lines = []
        clean_line_bboxes = []
        self.children.lines = [l for l in self.children.lines if l.children.words]
        for l in self.children.lines:
            l.adjust_bbox()
            if l.bbox.bbox not in clean_line_bboxes:
                clean_lines.append(l)
                clean_line_bboxes.append(l.bbox.bbox)
        self.children.lines = clean_lines

        # Create fake lines for words without lines
        for word in self.children.words:
            if 'line' not in word.parents.__dict__:
                logger.debug(f'Word without line at page {self.id}: "{word.text}" at {word.bbox.bbox}')
                self.children.lines.append(RawLine(page=self, word_ids=[word.id], bbox=word.bbox))

        # Process regions
        clean_regions = []
        clean_region_bboxes = []
        self.children.regions = [r for r in self.children.regions
                                 if r.region_type not in vs.EXCLUDED_REGION_TYPES
                                 and r.children.words]
        for r in self.children.regions:
            r.adjust_bbox()
            if r.bbox.bbox not in clean_region_bboxes:
                clean_regions.append(r)
                clean_region_bboxes.append(r.bbox.bbox)
        self.children.regions = clean_regions

        # Cut lines according to regions
        for r in self.children.regions:
            r.children.lines = []

            for l in self.children.lines:
                if 'region' in l.parents.__dict__ and l.parents.region != r:
                    continue
                # If the line is entirely in the region, append it
                if is_bbox_within_bbox(contained=l.bbox.bbox,
                                       container=r.bbox.bbox):
                    l.parents.region = r  # Link the line to its region
                    r.children.lines.append(l)

                # If the line is only partially in the region, handle the line splitting problem.
                elif any([is_bbox_within_bbox(w.bbox.bbox, r.bbox.bbox)
                          for w in l.children.words]):

                    # Create the new line and append it both to region and page lines
                    new_line = RawLine(page=self,
                                       word_ids=[w.id for w in l.children.words
                                                 if is_bbox_within_bbox(w.bbox.bbox, r.bbox.bbox)])

                    new_line.adjust_bbox()
                    new_line.parents.region = r
                    r.children.lines.append(new_line)

                    # Actualize the old line
                    l.children.words = [w for w in l.children.words
                                        if w.id not in new_line.word_ids]
                    l.adjust_bbox()

            r.children.lines.sort(key=lambda l: l.bbox.xywh[1])

        # Create fake regions for lines with no regions
        for l in self.children.lines:
            if 'region' not in l.parents.__dict__:
                line_region = RawRegion(region_type='line_region',
                                        bbox=Shape(l.bbox.bbox),
                                        page=self)
                l.parents.region = line_region
                line_region.children.lines = [l]
                self.children.regions.append(line_region)

        # Actualize global page reading order
        self.children.regions = sort_to_reading_order(elements=self.children.regions)

        # Actualize the lines
        self.children.lines = [l for r in self.children.regions for l in r.children.lines]
        self.children.words = [w for l in self.children.lines for w in l.children.words]

        if debug_dir:
            _ = self.draw_textcontainers(output_path=debug_dir / f'{self.id}_optimised.png')

        self.is_optimised = True


    def get_ocr_gt_page(self) -> 'RawPage':
        """Returns the OCR groundtruth of the page if available. If not, returns self."""
        if self.id in self.parents.commentary.ocr_gt_page_ids:
            return self.parents.commentary.ocr_gt_pages[self.parents.commentary.ocr_gt_page_ids.index(self.id)]
        elif self.id in self.parents.commentary.ocr_gt_partial_page_ids:
            self.optimise()
            partial_gt_page = self.parents.commentary.ocr_gt_partial_pages[self.parents.commentary.ocr_gt_partial_page_ids.index(self.id)]
            partial_gt_regions = []
            for partial_gt_region in partial_gt_page.children.regions:
                if hasattr(partial_gt_region, 'is_ocr_gt'):
                    partial_gt_regions.append(partial_gt_region)
                else:
                    for ocr_region in self.children.regions:
                        if are_bboxes_overlapping_with_threshold(partial_gt_region.bbox, ocr_region.bbox, 0.85):
                            partial_gt_regions.append(ocr_region)
                            break
            partial_gt_page.children.regions = partial_gt_regions
            self.reset()
            return partial_gt_page
        else:
            return self

    @lazy_property
    def id(self) -> str:
        return self.ocr_path.stem if self.ocr_path is not None else Path(self.via_dict['filename']).stem

    @lazy_property
    def img_path(self) -> Path:
        return self.parents.commentary.img_dir / f'{self.id}{vs.DEFAULT_IMG_EXTENSION}'

    @lazy_property
    def image(self) -> AjmcImage:
        return AjmcImage(id=self.id, path=self.img_path)

    @lazy_property
    def markup(self) -> bs4.BeautifulSoup:
        if self.ocr_path is not None:
            if self.ocr_format != 'json':
                return bs4.BeautifulSoup(self.ocr_path.read_text('utf-8'), 'xml')
            else:
                return json.loads(self.ocr_path.read_text('utf-8'))

    @lazy_property
    def ocr_format(self) -> str:
        if self.ocr_path.suffix not in vs.OCR_OUTPUTS_EXTENSIONS:
            raise NotImplementedError(
                    f'This OCR output format is not supported. Expecting {vs.OCR_OUTPUTS_EXTENSIONS} but found {self.ocr_path.suffix}.')
        return self.ocr_path.suffix[1:]

    @lazy_property
    def bbox(self) -> Shape:
        return Shape(get_bbox_from_points([xy for w in self.children.words for xy in w.bbox.bbox]))


    @lazy_property
    def via_dict(self) -> dict:
        page_key = [k for k in self.parents.commentary.via_project['_via_img_metadata'].keys() if k.startswith(self.id)][0]
        return self.parents.commentary.via_project['_via_img_metadata'][page_key]


class RawRegion(RawTextContainer):

    @docstring_formatter(**docstrings)
    def __init__(self,
                 region_type: str,
                 bbox: Shape,
                 page: 'RawPage',
                 is_ocr_gt: bool = False,
                 **kwargs):
        """Default constructor.

        Args:
            region_type: {olr_region_type}
            bbox: {coords_single}
            page: {parent_page}
        """

        assert region_type in vs.ORDERED_OLR_REGION_TYPES, f'Unknown region type: {region_type} in page {page.id}'

        super().__init__(region_type=region_type,
                         bbox=bbox,
                         page=page,
                         is_ocr_gt=is_ocr_gt,
                         _inclusion_threshold=vs.PARAMETERS['ocr_region_inclusion_threshold'],
                         **kwargs)


    @classmethod
    @docstring_formatter(**docstrings)
    def from_via(cls, via_dict: Dict[str, dict], page: 'RawPage'):
        """Constructs a region directly from its corresponding ``via_dict``.

        Args:
            via_dict: {via_dict}
            page: {parent_page}
        """

        return cls(region_type=via_dict['region_attributes']['label'].replace(vs.OLR_PREFIX, '').replace(vs.OCR_GT_PREFIX, ''),
                   bbox=Shape.from_via(via_dict),
                   page=page,
                   is_ocr_gt=vs.OCR_GT_PREFIX in via_dict['region_attributes']['label'])

    def _get_children(self, children_type):
        return [el for el in getattr(self.parents.page.children, children_type)
                if is_bbox_within_bbox_with_threshold(contained=el.bbox.bbox,
                                                      container=self.bbox.bbox,
                                                      threshold=self._inclusion_threshold)]


class RawLine(RawTextContainer):
    """Class for OCR lines."""

    def __init__(self, page: RawPage, **kwargs):
        assert 'bbox' in kwargs or 'word_ids' in kwargs, 'Either bbox or word_ids must be provided.'
        super().__init__(page=page, **kwargs)

    def _get_children(self, children_type):
        if children_type == 'words':
            return [w for w in self.parents.page.children.words if w.id in self.word_ids]
        else:
            return [c for c in getattr(self.parents.page.children, children_type)
                    if is_bbox_within_bbox_with_threshold(contained=c.bbox.bbox, container=self.bbox.bbox,
                                                          threshold=vs.PARAMETERS['words_line_inclusion_threshold'])]

    @lazy_property
    def word_ids(self) -> List[Union[str, int]]:
        word_ids = []
        for w in self.parents.page.children.words:
            if 'line' in w.parents.__dict__ and w.parents.line is not self:
                continue
            if is_bbox_within_bbox_with_threshold(contained=w.bbox.bbox, container=self.bbox.bbox,
                                                  threshold=vs.PARAMETERS['words_line_inclusion_threshold']):
                w.parents.line = self
                word_ids.append(w.id)

        return word_ids

    @lazy_property
    def bbox(self) -> Shape:
        return Shape(get_bbox_from_points([xy for w in self.children.words for xy in w.bbox.bbox]))


class RawWord(RawTextContainer):
    """Class for ocr words.


    Args:
        id: The page level unique identifier of the word (its number on the page). Only used in optimise to reforge lines.

    """

    def __init__(self,
                 id: Union[int, str],
                 text: str,
                 page: RawPage,
                 bbox: Shape,
                 **kwargs):
        super().__init__(id=id, text=text, page=page, bbox=bbox, **kwargs)

    @classmethod
    def from_via(cls, id: Union[int, str], via_dict: Dict[str, dict], page: 'RawPage'):
        return cls(id=id, text=via_dict['region_attributes']['label'], page=page, bbox=Shape.from_via(via_dict))


    def _get_children(self, children_type: str):
        return []  # Words have no children


    def adjust_bbox(self):
        self.bbox = adjust_bbox_to_included_contours(self.bbox.bbox, self.parents.page.image.contours)


class RawAnnotation(TextContainer):

    @docstring_formatter(**docstrings)
    def __init__(self,
                 page: 'RawPage',
                 bboxes: List['Shape'],
                 shifts: Tuple[int, int],
                 text_window: str,
                 warnings: List[str],
                 **kwargs):
        """Default constructor for annotation.

        Though it can be used directly, it is usually called via ``from_cas_annotation`` class method instead.
        ``kwargs`` are used to pass any desired attribute or to manually set the values of properties and to
        pass subclass-specific attributes, such as label for entities or ``corrputed`` for gold sentences.

        Args:
            page: {parent_page}
            bboxes: A list of ``Shape`` objects representing the bounding boxes of the annotation.
            shifts: A tuple of two integers representing the shifts of the annotation wrt its word.
            text_window: A string representing the text window of the annotation.
            warnings: A list of strings representing the warnings of the annotation.
        """

        super().__init__(page=page,
                         bboxes=bboxes,
                         shifts=shifts,
                         text_window=text_window,
                         warnings=warnings,
                         **kwargs)

    def _get_parent(self, parent_type):
        raise NotImplementedError('Parents to ``RawAnnotation``s are not implemented. They can only be accessed if provided at __init__.')

    def _get_children(self, children_type):
        """Returns the children of the annotation, coping with the fact that annotation have multiple bboxes."""

        # We first get the children based on overlap with the bboxes
        # Notice that there is no bijection between the set of bboxes and the set of children (i.e. there can be 3 bboxes but 4 words and vice-versa)
        children = [c for c in getattr(self.parents.page.children, children_type)
                    if any([is_bbox_within_bbox_with_threshold(contained=c.bbox.bbox, container=bbox.bbox,
                                                               threshold=vs.PARAMETERS['word_annotation_inclusion_threshold'])
                            for bbox in self.bboxes])]

        if children_type == 'words' and not children:
            # In this case, the annotation has no words. We start by making the bboxes to enlarge bboxes by 20% horizontally
            enlarged_bboxes = [((b.xmin - int(0.2 * b.width), b.ymin), (b.xmax + int(0.2 * b.width), b.ymax)) for b in self.bboxes]
            threshold = vs.PARAMETERS['word_annotation_inclusion_threshold']
            # We now reduce the threshold by 0.1 until we find a word
            while not children and threshold > 0.2:
                children = [c for c in self.parents.page.children.words
                            if any([is_bbox_within_bbox_with_threshold(contained=c.bbox.bbox, container=bbox, threshold=threshold)
                                    for bbox in enlarged_bboxes])]
                threshold -= 0.1

            # If we still have no children, we find the word which stand nearest to the first bbox
            # We weight the distance in order to avoid taking words that are too far vertically (we want words from the same line)
            if not children:
                return [min(self.parents.page.children.words,
                            key=lambda w: (w.bbox.center[0] - self.bboxes[0].center[0]) ** 2 + 10 * (
                                    w.bbox.center[1] - self.bboxes[0].center[1]) ** 2)]
            else:
                return children
        else:
            return children


class RawEntity(RawAnnotation):
    """Class for cas imported entities."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, transcript, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                                           rebuild=rebuild, verbose=verbose)
        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   transcript=transcript,
                   label=cas_annotation.value,
                   wikidata_id=cas_annotation.wikidata_id,
                   text_window=text_window,
                   warnings=warnings)


class RawSentence(RawAnnotation):
    """Class for cas imported gold sentences."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, transcript, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
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
        bboxes, shifts, transcript, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                                           rebuild=rebuild, verbose=verbose)
        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   text_window=text_window,
                   warnings=warnings)


class RawLemma(RawAnnotation):
    """Class for cas imported entities."""

    @classmethod
    def from_cas_annotation(cls, page, cas_annotation, rebuild, verbose: bool = False):
        # Get general text-alignment-related about the annotation
        bboxes, shifts, transcript, text_window, warnings = cas_utils.align_cas_annotation(cas_annotation=cas_annotation,
                                                                                           rebuild=rebuild, verbose=verbose)

        return cls(page,
                   bboxes=[Shape.from_xywh(*bbox) for bbox in bboxes],
                   shifts=shifts,
                   text_window=text_window,
                   warnings=warnings,
                   transcript=transcript,
                   **{attr_: getattr(cas_annotation, attr_, None) for attr_ in ['anchor_target', 'value']})

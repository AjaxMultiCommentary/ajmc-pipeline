import re
import json
import os
from time import strftime
from typing import Dict, Optional, List, Union, Any
import bs4.element

from ajmc.commons.miscellaneous import lazy_property, get_custom_logger, lazy_init
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons import variables
from ajmc.commons.geometry import (
    Shape,
    is_rectangle_within_rectangle_with_threshold,
    is_rectangle_within_rectangle, adjust_to_included_contours, get_bbox_from_points
)
from ajmc.commons.image import Image, draw_page_regions_lines_words
from ajmc.olr.utils import sort_to_reading_order, get_page_region_dicts_from_via
import jsonschema

from ajmc.text_processing.canonical_classes import CanonicalCommentary, CanonicalWord, \
    CanonicalSinglePageTextContainer, CanonicalPage
from ajmc.text_processing.markup_processing import parse_markup_file, get_element_bbox, \
    get_element_text, find_all_elements
from ajmc.commons.file_management.utils import verify_path_integrity, parse_ocr_path, get_path_from_id, guess_ocr_format

logger = get_custom_logger(__name__)


# @lazy_attributer('ocr_format', lambda self: self.page.ocr_format, lazy_property)
class OcrTextContainer:
    pass

    @lazy_property
    def parents(self):
        raise NotImplementedError('`parents` attribute is only available for `CanonicalTextContainer`s')

    @lazy_property
    def bbox(self):
        return get_element_bbox(self.markup, self.ocr_format)

    @lazy_property
    def image(self) -> Image:
        return self.page.Image

    @lazy_property
    def ocr_format(self) -> str:
        return self.page.ocr_format

    @lazy_property
    def text(self) -> str:
        return '\n'.join([l.text for l in self.children['line']])


class OcrCommentary:
    """`OcrCommentary` objects reprensent a single ocr-run of on a commentary, i.e. a collection of page OCRed pages."""

    @lazy_init
    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: Optional[str] = None,
                 ocr_dir: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 via_path: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 groundtruth_dir: Optional[str] = None,
                 ocr_run: Optional[str] = None
                 ):
        """Default constructor, where custom paths can be provided.

        This is usefull when you want to instantiate a `OcrCommentary` without using `ajmc`'s folder structure. Note that
        only the requested paths must be provided. For instance, the object will be created if you do not provided the
        path to the images. But logically, you won't be able to access fonctionnalities that requires it (for instance
        `OcrCommentary.pages[0].image`).

        Args:
            id: The id of the commentary (e.g. sophoclesplaysa05campgoog).
            ocr_dir: {ocr_dir}
            via_path: {via_path}
            image_dir: {image_dir}
            groundtruth_dir: {groundtruth_dir}
        """
        pass

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
                   base_dir=os.path.join(base_dir,id),
                   via_path=os.path.join(base_dir, id, variables.PATHS['via_path']),
                   image_dir=os.path.join(base_dir, id, variables.PATHS['png']),
                   groundtruth_dir=os.path.join(base_dir, id, variables.PATHS['groundtruth']),
                   ocr_run=ocr_run)

    @lazy_property
    def pages(self) -> List['OcrPage']:
        """The pages contained in the commentaries"""
        pages = []
        for file in [f for f in os.listdir(self.ocr_dir) if f[-4:] in ['.xml', 'hocr', 'html']]:
            pages.append(OcrPage(ocr_path=os.path.join(self.ocr_dir, file),
                                 id=file.split('.')[0],
                                 image_path=get_path_from_id(file.split('.')[0], self.image_dir),
                                 commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property  # Todo : This should not be maintained anymore
    def ocr_groundtruth_pages(self) -> Union[List['OcrPage'], list]:
        """The commentary's pages which have a groundtruth file in `self.paths['groundtruth']`."""
        pages = []
        for file in [f for f in os.listdir(self.groundtruth_dir) if f.endswith('.html')]:
            pages.append(OcrPage(ocr_path=os.path.join(self.groundtruth_dir, file),
                                 id=file.split('.')[0],
                                 image_path=get_path_from_id(file.split('.')[0], self.image_dir),
                                 commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def via_project(self) -> dict:
        with open(self.via_path, 'r') as file:
            return json.load(file)

    def to_canonical(self):
        can = CanonicalCommentary(id=self.id,
                                  images=[],
                                  children={k: [] for k in ['page', 'region', 'line', 'word']},
                                  info={'image_dir': self.image_dir,
                                        'base_dir': self.base_dir})

        if hasattr(self, 'ocr_run'):
            can.info['ocr_run'] = self.ocr_run

        w_count, p_count, r_count, l_count = 0, 0, 0, 0

        for i, p in enumerate(self.pages):
            if i % 20 == 0:
                print(f'Processing page {i} of {len(self.pages)}')

            p.optimise()
            p_start = w_count
            for r in p.children['region']:
                r_start = w_count
                for l in r.children['line']:
                    l_start = w_count
                    for w in l.children['word']:
                        can.children['word'].append(CanonicalWord(type='word',
                                                                  index=w_count,
                                                                  bbox=w.bbox.bbox_2,
                                                                  text=w.text,
                                                                  commentary=can))
                        w_count += 1

                    can.children['line'].append(CanonicalSinglePageTextContainer(type='line',
                                                                                 index=l_count,
                                                                                 word_range=(l_start, w_count - 1),
                                                                                 # ->w_count+=1 above
                                                                                 commentary=can))
                    l_count += 1

                can.children['region'].append(CanonicalSinglePageTextContainer(type='region',
                                                                               index=r_count,
                                                                               word_range=(r_start, w_count - 1),
                                                                               commentary=can,
                                                                               info={'region_type': r.region_type}))
                r_count += 1

            can.children['page'].append(CanonicalPage(id=p.id,
                                                      index=p_count,
                                                      word_range=(p_start, w_count - 1),
                                                      commentary=can))
            can.images.append(Image(id=p.id,
                                    path=p.image_path,
                                    word_range=(p_start, w_count - 1)))
            p_count += 1
            p.reset()

        return can


class OcrPage(OcrTextContainer):
    """A class representing a commentary page."""

    @lazy_init
    def __init__(self,
                 ocr_path: str,
                 id: Optional[str] = None,
                 image_path: Optional[str] = None,
                 commentary: Optional[OcrCommentary] = None):
        """Default constructor.

        Args:
            ocr_path: Absolute path to an OCR output file
            commentary: The commentary to which the page belongs.
        """

        self.page = self

    @lazy_property
    def ocr_format(self) -> str:
        return guess_ocr_format(self.ocr_path)

    @lazy_property
    def children(self):
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

        regions = [OlrRegion.from_via(via_dict=r, page=self) for r in
                   get_page_region_dicts_from_via(self.id, self.commentary.via_project)]

        return {'region': regions, 'line': lines, 'word': words}

    @lazy_property
    def image(self) -> Image:
        return Image(id=self.id, path=self.image_path)

    @lazy_property
    def markup(self) -> bs4.BeautifulSoup:
        return parse_markup_file(self.ocr_path)

    @lazy_property
    def canonical_data(self) -> Dict[str, Any]:
        """Creates canonical data, as used for INCEpTION. """
        logger.warning('You are creating a canonical data version 1. For version two, use `commentary.to_canonical`.')
        data = {'id': self.id,
                'iiif': 'None',
                'cdate': strftime('%Y-%m-%d %H:%M:%S'),
                'regions': []}

        for r in self.children['region']:
            r_dict = {'region_type': r.region_type,
                      'bbox': r.bbox.xywh,
                      'lines': [
                          {
                              'bbox': l.bbox.xywh,
                              'words': [
                                  {
                                      'bbox': w.bbox.xywh,
                                      'text': w.text
                                  } for w in l.children['word']
                              ]

                          } for l in r.children['line']
                      ]
                      }
            data['regions'].append(r_dict)

        return data

    def reset(self):
        delattr(self, 'children')
        delattr(self, 'image')
        delattr(self, 'text')
        delattr(self, 'canonical_data')

    def purge_image(self):
        delattr(self, 'image')

    def to_json(self, output_dir: str, schema_path: str = variables.PATHS['schema']):
        """Validate `self.canonical_data` and serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        jsonschema.validate(instance=self.canonical_data, schema=schema)

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(self.canonical_data, f, indent=4, ensure_ascii=False)

    def optimise(self,
                 do_debug: bool = False):

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/{self.id}_raw.png")

        logger.warning("You are optimising a page, bboxes and children are going to be changed")
        self.reset()

        # Process words
        self.children['word'] = [w for w in self.children['word'] if re.sub(r'\s+', '', w.text) != '']
        for w in self.children['word']:
            w.adjust_bbox()

        # Process lines
        self.children['line'] = [l for l in self.children['line'] if l.children['word']]
        for l in self.children['line']:
            l.adjust_bbox()

        # Process regions
        self.children['region'] = [r for r in self.children['region']
                                   if r.region_type not in ['undefined', 'line_number_commentary']
                                   and r.children['word']]
        for r in self.children['region']:
            r.adjust_bbox()

        # Cut lines according to regions
        for r in self.children['region']:
            r.children['line'] = []

            for l in self.children['line']:
                # If the line is entirely in the region, append it
                if is_rectangle_within_rectangle(contained=l.bbox.bbox,
                                                 container=r.bbox.bbox):
                    l.region = r  # Link the line to its region
                    r.children['line'].append(l)

                # If the line is only partially in the region, handle the line splitting problem.
                elif any([is_rectangle_within_rectangle(w.bbox.bbox, r.bbox.bbox)
                          for w in l.children['word']]):

                    # Create the new line and append it both to region and page lines
                    new_line = OcrLine(markup=None,
                                       page=self,
                                       word_ids=[w.id for w in l.children['word']
                                                 if is_rectangle_within_rectangle(w.bbox.bbox, r.bbox.bbox)])
                    new_line.adjust_bbox()
                    new_line.region = r
                    r.children['line'].append(new_line)

                    # Actualize the old line
                    l.children['word'] = [w for w in l.children['word']
                                          if w.id not in new_line.word_ids]
                    l.adjust_bbox()

            r.children['line'].sort(key=lambda l: l.bbox.xywh[1])

        # Actualize global page reading order
        ## Create fake regions for lines with no regions
        for l in self.children['line']:
            if not hasattr(l, 'region'):
                line_region = OlrRegion(region_type='line_region',
                                        bbox=Shape(l.bbox.bbox),
                                        page=self)
                line_region.children['line'] = [l]
                self.children['region'].append(line_region)

        self.children['region'] = sort_to_reading_order(elements=self.children['region'])
        self.children['line'] = [l for r in self.children['region'] for l in r.children['line']]
        self.children['word'] = [w for l in self.children['line'] for w in l.children['word']]

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/{self.id}_optimised.png")


class OlrRegion(OcrTextContainer):
    """A class representing OLR regions.

    `OlrRegion`s can be instantiated from a via-dictionary or manually.

    Attributes:

        region_type (str):
            The type of the region, e.g. 'page_number', 'introduction'...

        bbox (Shape):
            A `Shape` object representing the coordinates of the region as extracted from via.

        page (OcrPage):
            The `OcrPage` object to which the region belongs

        words:
            The words included in the region.
    """

    @lazy_init
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
        self._inclusion_threshold = 0.7

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

    # =================== Parents and children ================
    @lazy_property
    def bbox(self):
        """Just here for the clarity of inheritance, but just wraps value given in __init__"""
        return self.bbox

    @lazy_property
    def children(self):
        return {k: [el for el in self.page.children[k]
                    if is_rectangle_within_rectangle_with_threshold(contained=el.bbox.bbox,
                                                                    container=self.bbox.bbox,
                                                                    threshold=self._inclusion_threshold)]
                for k in ['line', 'word']}

    def adjust_bbox(self):
        words_points = [xy for w in self.children['word'] for xy in w.bbox.bbox]
        self.bbox = Shape(get_bbox_from_points(words_points))


class OcrLine(OcrTextContainer):
    """Class for Ocrlines."""

    @lazy_init
    def __init__(self,
                 markup: 'bs4.element.Tag',
                 page: OcrPage,
                 word_ids: Optional[List[Union[str, int]]] = None):
        pass

    @lazy_property
    def children(self):
        return {'word': [w for w in self.page.children['word'] if w.id in self.word_ids]}

    @lazy_property
    def text(self) -> str:
        return ' '.join([w.text for w in self.children['word']])

    def adjust_bbox(self):
        words_points = [xy for w in self.children['word'] for xy in w.bbox.bbox]
        self.bbox = Shape(get_bbox_from_points(words_points))


class OcrWord(OcrTextContainer):
    """Class for Words."""

    @lazy_init
    def __init__(self,
                 id: Union[int, str],
                 markup: 'bs4.element.Tag',
                 page: OcrPage
                 ):
        pass

    @lazy_property
    def children(self):
        return {}

    @lazy_property
    def text(self):
        return get_element_text(element=self.markup, ocr_format=self.ocr_format)

    def adjust_bbox(self):
        self.bbox = adjust_to_included_contours(self.bbox.bbox, self.page.image.contours)

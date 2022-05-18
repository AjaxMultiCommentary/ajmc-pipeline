import copy
import json
import os
from time import strftime
from typing import Dict, Optional, List, Union, Any
import bs4.element
from ajmc.commons.miscellaneous import lazy_property, get_custom_logger, docstring_formatter
from ajmc.commons import variables
from ajmc.commons.geometry import (
    Shape,
    is_rectangle_within_rectangle_with_threshold,
    get_bounding_rectangle_from_points,
    is_rectangle_within_rectangle, are_rectangles_overlapping,
    shrink_to_included_contours
)
from ajmc.commons.image import Image
from ajmc.olr.utils.region_processing import order_olr_regions, get_page_region_dicts_from_via
import jsonschema
from ajmc.text_importation.markup_processing import parse_markup_file, get_element_coords, \
    get_element_text, find_all_elements
from ajmc.commons.file_management.utils import verify_path_integrity, parse_ocr_path, get_path_from_id, guess_ocr_format

logger = get_custom_logger(__name__)


class Commentary:
    """`Commentary` objects reprensent a single ocr run of a commentary."""

    def __init__(self,
                 commentary_id: str = None,
                 ocr_dir: str = None,
                 via_path: str = None,
                 image_dir: str = None,
                 groundtruth_dir: str = None,
                 _base_dir: str = None,  # Only useful for Commentary.from_structure()
                 ):
        """Default constructor, where custom paths can be provided.

        Args:
            commentary_id: The id of the commentary
            ocr_dir: Absolute path to an ocr output folder.
            via_path:
            image_dir:
            groundtruth_dir:
        """
        self.id = commentary_id
        self.paths = {'ocr_dir': ocr_dir,
                      'via_path': via_path,
                      'image_dir': image_dir,
                      'groundtruth_dir': groundtruth_dir,
                      'base_dir': _base_dir}

    @classmethod
    @docstring_formatter(output_dir_format=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])
    def from_folder_structure(cls, ocr_dir: Optional[str] = None, commentary_id: Optional[str] = None):
        """Use this method to construct a `Commentary`-object using ajmc's folder structure.

        Args:
            ocr_dir: Path to the directory in which ocr-outputs are stored. It should end with
            {output_dir_format}.
            commentary_id: for dev purposes only, do not use :)
        """
        if ocr_dir:
            verify_path_integrity(path=ocr_dir, path_pattern=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])
            base_dir, commentary_id, ocr_run = parse_ocr_path(path=ocr_dir)

        elif commentary_id:
            base_dir = variables.PATHS['base_dir']

        return cls(commentary_id=commentary_id,
                   ocr_dir=ocr_dir,
                   via_path=os.path.join(base_dir, commentary_id, variables.PATHS['via_path']),
                   image_dir=os.path.join(base_dir, commentary_id, variables.PATHS['png']),
                   groundtruth_dir=os.path.join(base_dir, commentary_id, variables.PATHS['groundtruth']),
                   _base_dir=base_dir)

    @lazy_property
    def ocr_format(self) -> str:
        """The format of the commentary's ocr (e.g. 'hocr', 'pagexml'...)."""
        return self.pages[0].ocr_format

    @lazy_property
    def pages(self) -> List['Page']:
        """The pages contained in the commentaries"""
        pages = []
        for file in [f for f in os.listdir(self.paths['ocr_dir']) if f[-4:] in ['.xml', 'hocr', 'html']]:
            pages.append(Page(ocr_path=os.path.join(self.paths['ocr_dir'], file),
                              page_id=file.split('.')[0],
                              groundtruth_path=get_path_from_id(file.split('.')[0], self.paths['groundtruth_dir']),
                              image_path=get_path_from_id(file.split('.')[0], self.paths['image_dir']),
                              via_path=self.paths['via_path'],
                              commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def ocr_groundtruth_pages(self) -> Union[List['Page'], list]:
        """The commentary's pages which have a groundtruth file in `self.paths['groundtruth']`."""
        pages = []
        for file in [f for f in os.listdir(self.paths['groundtruth_dir']) if f.endswith('.html')]:
            pages.append(Page(ocr_path=os.path.join(self.paths['groundtruth_dir'], file),
                              page_id=file.split('.')[0],
                              groundtruth_path=None,
                              image_path=get_path_from_id(file.split('.')[0], self.paths['image_dir']),
                              via_path=self.paths['via_path'],
                              commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def olr_groundtruth_pages(self, ) -> Union[List['Page'], list]:
        """Returns the list of `Page`s which have at least one annotated region which is neither 'commentary' nor
        'undefined'. """

        pages = []
        for k, v in self.via_project['_via_img_metadata'].items():
            if any([r['region_attributes']['text'] not in ['commentary', 'undefined'] for r in v['regions']]):
                p_id = v['filename'].split('.')[0]
                pages.append(Page(ocr_path=get_path_from_id(p_id, self.paths['ocr_dir']),
                                  page_id=p_id,
                                  groundtruth_path=get_path_from_id(p_id, dir_=self.paths['groundtruth_dir']),
                                  image_path=get_path_from_id(p_id, dir_=self.paths['image_dir']),
                                  via_path=self.paths['via_path'],
                                  commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def regions(self):
        return [r for p in self.pages for r in p.regions]

    @lazy_property
    def lines(self):
        return [l for p in self.pages for l in p.lines]

    @lazy_property
    def words(self):
        return [w for p in self.pages for w in p.words]

    @lazy_property
    def via_project(self) -> dict:
        with open(self.paths['via_path'], 'r') as file:
            return json.load(file)

    def _get_page_ids(self) -> List[str]:
        """Gets the ids of the pages contained in Commentary by scanning the png files"""
        return [p[:-4] for p in os.listdir(self.paths['image_dir']) if p.endswith('.png')]


class Page:
    """A class representing a commentary page."""

    def __init__(self,
                 ocr_path: str,
                 page_id: Optional[str] = None,
                 groundtruth_path: Optional[str] = None,
                 image_path: Optional[str] = None,
                 via_path: Optional[str] = None,
                 commentary: Optional[Commentary] = None):
        """Default constructor.

        Args:
            ocr_path: Absolute path to an OCR output file
            commentary: The commentary to which the page belongs.
        """
        self.id = page_id
        self.commentary = commentary
        self.paths = {
            'ocr_path': ocr_path,
            'groundtruth_path': groundtruth_path,
            'image_path': image_path,
            'via_path': via_path
        }

    @classmethod
    def from_structure(cls, ocr_path: str, commentary: Optional[Commentary] = None):
        """Builds Page object from an OCR-path"""

        commentary = commentary if commentary else Commentary.from_folder_structure(
            ocr_dir='/'.join(ocr_path.split('/')[:-1]))
        page_id = ocr_path.split('/')[-1].split('.')[0]

        return cls(ocr_path=ocr_path,
                   page_id=page_id,
                   groundtruth_path=get_path_from_id(page_id=page_id, dir_=commentary.paths['groundtruth_dir']),
                   image_path=get_path_from_id(page_id=page_id, dir_=commentary.paths['image_dir']),
                   via_path=commentary.paths['via_path'],
                   commentary=commentary)

    # ===============================  Properties  ===============================
    @lazy_property
    def groundtruth_page(self) -> Union['Page', None]:
        if os.path.exists(self.paths['groundtruth_path']):
            return Page(ocr_path=self.paths['groundtruth_path'],
                        page_id=self.id,
                        groundtruth_path=None,
                        image_path=self.paths['image_path'],
                        via_path=self.paths['via_path'],
                        commentary=self.commentary)
        else:
            logger.warning(f"""Page {self.id} has no groundtruth page.""")
            return None

    @lazy_property
    def ocr_format(self) -> str:
        return guess_ocr_format(self.paths['ocr_path'])

    @lazy_property
    def regions(self):
        """Gets page regions, removing empty regions and reordering them."""
        return self.get_regions()

    @lazy_property
    def lines(self):
        return [
            TextElement(markup=line,
                        page=self,
                        ocr_format=self.ocr_format)
            for line in find_all_elements(self.markup, 'lines', self.ocr_format)
        ]

    @lazy_property
    def words(self):
        return [w for l in self.lines for w in l.words]

    # ===============================  Other properties  ===============================

    @lazy_property
    def via_project(self) -> dict:
        if self.commentary:
            return self.commentary.via_project
        else:
            with open(self.paths['via_path'], 'r') as file:
                return json.load(file)

    @lazy_property
    def image(self) -> Image:
        return Image(path=self.paths['image_path'])

    @lazy_property
    def text(self) -> str:
        return '\n'.join([line.text for line in self.lines])

    @lazy_property
    def markup(self) -> bs4.BeautifulSoup:
        return parse_markup_file(self.paths['ocr_path'])

    @lazy_property
    def canonical_data(self) -> Dict[str, Any]:
        data = {'id': self.id,
                'iiif': 'None',
                'cdate': strftime('%Y-%m-%d %H:%M:%S'),
                'regions': []}

        for r in self.regions:
            r_dict = {'region_type': r.region_type,
                      'coords': r.coords.xywh,
                      'lines': [
                          {
                              'coords': line.coords.xywh,
                              'words': [
                                  {
                                      'coords': word.coords.xywh,
                                      'text': word.text
                                  } for word in line.words
                              ]

                          } for line in r.lines
                      ]
                      }
            data['regions'].append(r_dict)

        return data

    def to_json(self, output_dir: str, schema_path: str = variables.PATHS['schema']):
        """Validate `self.canonical_data` & serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        jsonschema.validate(instance=self.canonical_data, schema=schema)

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(self.canonical_data, f, indent=4, ensure_ascii=False)

    def get_regions(self, region_types: Union[List[str], str] = 'all') -> List['Region']:
        """Gets the regions belonging to `page` from the `via_project`.

        What this does is :
            1. Extracting page region dicts from via
            2. Instantiating `Region` objects
            3. Removing empty regions
            4. Rebuilding region reading order

        Note:
            This instantiate `Region` objects and may therefore take time. For a more efficient computation,
            preselect the desired region types. But warning, don't create canonical jsons
            with of pre-selected regions only !

        Returns:
            A list region objects
        """

        regions = get_page_region_dicts_from_via(self.id, self.via_project)

        if region_types == 'all':
            regions = [Region(r, self) for r in regions]
        else:
            regions = [Region(r, self) for r in regions if r['region_attributes']['text'] in region_types]

        regions = [r for r in regions if not (r.region_type == 'undefined' and not r.words)]
        regions = order_olr_regions(regions)

        return regions


class Region:
    """A class representing OLR regions extracted from a via project.

    Attributes:
        via_dict (dict):
            The via-dict, as extracted from the commentary's via_project. It should look like :

                $ { 'shape_attributes': {'name': 'rect', 'x': 31, 'y': 54, 'width': 1230, 'height': 453},
                    'region_attributes': {'text': 'preface'} }

        region_type (str):
            The type of the region, e.g. 'page_number', 'introduction'...
        coords (Shape):
            The actualized coordinates of the region, corresponding to the bounding rectangle of included words.
        page:
            The `PageXmlCommentaryPage` object to which the region belongs
        words (List[ElementType]):
            The words included in the region.
    """

    def __init__(self, via_dict: Dict[str, dict], page: 'Page', word_inclusion_threshold: float = 0.7):
        self.region_type = via_dict['region_attributes']['text']
        self.page = page
        self.markup = via_dict
        self._word_inclusion_threshold = word_inclusion_threshold

    # =================== Parents and children ================
    @lazy_property
    def lines(self):
        return self.get_readjusted_lines()

    @lazy_property
    def words(self):
        return [w for line in self.lines for w in line.words]

    @lazy_property
    def coords(self):
        """
        Note:
            On the contrary to other `Element` objects, `Region` has a particular initialization that requires some
            computation. As via regions coords are often fuzzy, `Region.coords` are first reconstructed."""
        return self.get_readjusted_coords(self._word_inclusion_threshold)

    @lazy_property
    def image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    @lazy_property
    def text(self):
        return '\n'.join([l.text for l in self.lines])

    def get_readjusted_coords(self, word_inclusion_threshold: float = 0.7) -> 'Shape':
        """Automatically readjusts the region coordinates, so that exactly fit the words contained in the region.

        This is done by :

                1. Finding the words in the region's page which are contained in the region's initial via_coords.
                This is done using `is_rectangle_partly_within_rectangle` with `word_inclusion_threshold`.
                2. Actualizing the coords to fit the exact bounding rectangle of contained words.

            Hence, `Region.coords` are instanciated immediately and is no `@lazy_property`."""

        # Find the words included
        initial_coords = Shape.from_xywh(x=self.markup['shape_attributes']['x'],
                                         y=self.markup['shape_attributes']['y'],
                                         w=self.markup['shape_attributes']['width'],
                                         h=self.markup['shape_attributes']['height'])

        words = [w for w in self.page.words
                 if is_rectangle_within_rectangle_with_threshold(w.coords.bounding_rectangle,
                                                                 initial_coords.bounding_rectangle,
                                                                 word_inclusion_threshold)]

        # resize region
        words_points = [xy for w in words for xy in w.coords.bounding_rectangle]
        return Shape(get_bounding_rectangle_from_points(words_points)) \
            if words_points else initial_coords

    def get_readjusted_lines(self) -> List['TextElement']:
        """Readjusts region lines to fit only the words of a line that are actually contained within the region.

        Goes through page-level lines to find the lines that overlap with the region. If there is an overlap, makes a
        copy of the lines and finds the words in the line that are contained in the region, then shrinks the lines
        coordinates to fit only the contained words.

        Note:
            This is mostly used to circumvent the double-column lines issue. Please we aware that page.lines will then
            be different from `[region.lines for region in page.regions]`, as this procedure does not changes page
            lines."""

        region_lines = []
        for line in self.page.lines:
            if are_rectangles_overlapping(line.coords.bounding_rectangle, self.coords.bounding_rectangle):
                line_ = copy.copy(line)
                line_._words = []
                for word in line.words:
                    if is_rectangle_within_rectangle(word.coords.bounding_rectangle,
                                                     self.coords.bounding_rectangle):
                        line_._words.append(word)

                if line_._words:
                    line_points = [xy for w in line_._words for xy in w.coords.bounding_rectangle]
                    line_._coords = Shape(get_bounding_rectangle_from_points(line_points)) \
                        if line_points else Shape.from_points([(0, 0)])
                    region_lines.append(line_)

        return region_lines


class TextElement:
    """Class for lines and words."""

    # This could be separated in two classes (`Lines` and `Words`) if needed

    def __init__(self, markup: 'bs4.element.Tag', page: Page, ocr_format: str):
        self.markup = markup
        self.page = page
        self.ocr_format = ocr_format

    @lazy_property
    def id(self):
        return self.markup['id']

    @lazy_property
    def element_type(self):
        raise NotImplementedError

    @lazy_property
    def raw_coords(self):
        return get_element_coords(self.markup, self.ocr_format)

    @lazy_property
    def coords(self) -> Shape:
        if self.words:
            return Shape.from_points([xy for w in self.words for xy in w.coords.bounding_rectangle])
        else:
            raw_shape = get_element_coords(self.markup, self.ocr_format)
            return shrink_to_included_contours(raw_shape.bounding_rectangle, self.page.image.contours)

    @lazy_property
    def image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    @lazy_property
    def text(self):
        # The conditional structure is there to avoid weird text reconstructions ('\n\nWord\n\n...') by bs4.
        if self.words:  # Element is a line
            return ' '.join([w.text for w in self.words])
        else:  # Element is a word
            return get_element_text(self.markup, self.ocr_format)

    @lazy_property
    def words(self):
        return [TextElement(el, self.page, self.ocr_format) for el in
                find_all_elements(self.markup, 'words', self.ocr_format)]

import copy
import json
import os
from time import strftime
from typing import Dict, Optional, List, Union
import bs4.element
from common_utils.general_utils import lazy_property
from common_utils.variables import PATHS
from common_utils.geometry import (
    Shape,
    is_rectangle_partly_within_rectangle,
    get_bounding_rectangle_from_points,
    is_rectangle_within_rectangle, are_rectangles_overlapping,
    shrink_to_included_contours
)
from common_utils.image_processing import Image
from olr.utils.region_processing import order_olr_regions, get_page_region_dicts_from_via
import jsonschema as js
from text_importation.file_management import get_page_ocr_path, get_ocr_dir_from_ocr_run, guess_ocr_format, \
    get_ocr_run_fullname
from text_importation.markup_processing import parse_markup_file, get_element_coords, \
    get_element_text, find_all_elements


class Commentary:
    """Abstract class representing commentary.

    Args:
        commentary_id: The id of the commentary
        ocr_dir: Absolute path to an ocr output folder. Prevails over `ocr_run` if provided.
        ocr_run: Full or partial name of a run in `ocr/runs`. Gets overwriten by `ocr_dir` if the latter is provided.
        """

    def __init__(self, commentary_id: str, ocr_dir: str = None, ocr_run: str = None):
        self.id = commentary_id
        if ocr_dir:
            self.ocr_dir = ocr_dir
            self.ocr_run = None
        elif ocr_run:
            self.ocr_run = get_ocr_run_fullname(self.id, ocr_run)
            self.ocr_dir = get_ocr_dir_from_ocr_run(self.id, self.ocr_run)

    @lazy_property
    def ocr_format(self):
        return self.pages[0].ocr_format

    @lazy_property
    def pages(self):
        return [Page(p_id,
                     commentary=self,
                     ocr_path=get_page_ocr_path(p_id, ocr_dir=self.ocr_dir))
                for p_id in self._get_page_ids()]

    @lazy_property
    def ocr_groundtruth_pages(self):
        gt_dir = os.path.join(PATHS['base_dir'], self.id, PATHS['groundtruth'])
        return [Page(page_id=fname[:-5], commentary=self, ocr_path=os.path.join(gt_dir, fname))
                for fname in os.listdir(gt_dir) if fname.startswith(self.id)]

    @lazy_property
    def olr_groundtruth_pages(self, ):
        """Returns the list of `Page`s which have at least one annotated region."""
        return [
            Page(page_id=item['filename'].split('.')[0],
                 ocr_path=get_page_ocr_path(item['filename'].split('.')[0], ocr_dir=self.ocr_dir),
                 commentary=self) for key, item in self.via_project['_via_img_metadata'].items() if
            any([r['region_attributes']['text'] not in ['commentary','undefined'] for r in item['regions']])
        ]

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
    def via_project(self):
        with open(os.path.join(PATHS['base_dir'], self.id, PATHS['via_project']), 'r') as file:
            return json.load(file)

    def _get_page_ids(self) -> List[str]:
        """Gets the ids of the pages contained in Commentary by scanning the png files"""
        images_dir = os.path.join(PATHS['base_dir'], self.id, PATHS['png'])
        return [p[:-4] for p in os.listdir(images_dir) if p.endswith('.png')]


class Page:
    """A class representing a commentary page."""

    def __init__(self, page_id: str, ocr_path: str = None, ocr_run: str = None,
                 commentary: Optional[Commentary] = None):
        """Default constructor.

        Args:
            page_id: The page identifier (e.g. `'sophoclesplaysa05campgoog_0147'`).
            ocr_path: The full or partial name of the ocr_run.
            commentary: The commentary to which the page belongs.
        """

        self.id = page_id
        if ocr_path:
            self.ocr_path = ocr_path
            self.ocr_run = None
        elif ocr_run:
            self.ocr_run = get_ocr_run_fullname(self.id.split('_')[0], ocr_run)
            self.ocr_path = get_page_ocr_path(self.id, ocr_run=self.ocr_run)

        self.commentary = commentary if commentary else Commentary(self.id.split("_")[0], ocr_run=self.ocr_run)

    # ===============================  Properties  ===============================
    @lazy_property
    def groundtruth_page(self):
        gt_dir = os.path.join(PATHS['base_dir'], self.commentary.id, PATHS['groundtruth'])
        gt_path = get_page_ocr_path(self.id, gt_dir)
        if os.path.exists(gt_path):
            return Page(self.id,
                        ocr_path=gt_path,
                        commentary=self.commentary)
        else:
            print(f"""Warning: Page {self.id} has no groundtruth in {gt_dir}.""")
            return None

    @lazy_property
    def ocr_format(self):
        return guess_ocr_format(self.ocr_path)

    @lazy_property
    def regions(self):
        """Gets page regions, removing empty regions and reordering them."""
        return self.get_regions()

    @lazy_property
    def lines(self):
        return [TextElement(line, self, self.ocr_format) for line in
                find_all_elements(self.markup, 'lines', self.ocr_format)]

    @lazy_property
    def words(self):
        return [w for l in self.lines for w in l.words]

    # ===============================  Other properties  ===============================

    @lazy_property
    def image(self):
        return Image(self.id)

    @lazy_property
    def text(self):
        return '\n'.join([line.text for line in self.lines])

    @lazy_property
    def markup(self):
        return parse_markup_file(self.ocr_path)

    @lazy_property
    def canonical_data(self) -> dict:
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

    def to_json(self, output_dir: str, schema_path: str = PATHS['schema']):
        """Validate `self.canonical_data` & serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        js.validate(instance=self.canonical_data, schema=schema)

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

        regions = get_page_region_dicts_from_via(self.id, self.commentary.via_project)

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
                 if is_rectangle_partly_within_rectangle(w.coords, initial_coords, word_inclusion_threshold)]

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

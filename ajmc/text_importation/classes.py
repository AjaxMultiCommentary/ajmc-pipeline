import copy
import re
import json
import os
from time import strftime
from typing import Dict, Optional, List, Union, Any, Tuple
import bs4.element

from ajmc.commons.arithmetic import are_intervals_within_intervals
from ajmc.commons.miscellaneous import lazy_property, get_custom_logger
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons import variables
from ajmc.commons.geometry import (
    Shape,
    is_rectangle_within_rectangle_with_threshold,
    get_bounding_rectangle_from_points,
    is_rectangle_within_rectangle, are_rectangles_overlapping,
    adjust_to_included_contours
)
from ajmc.commons.image import Image, draw_page_regions_lines_words
from ajmc.olr.utils.region_processing import sort_to_reading_order, get_page_region_dicts_from_via
import jsonschema
from ajmc.text_importation.markup_processing import parse_markup_file, get_element_coords, \
    get_element_text, find_all_elements
from ajmc.commons.file_management.utils import verify_path_integrity, parse_ocr_path, get_path_from_id, guess_ocr_format

logger = get_custom_logger(__name__)


class OcrCommentary:
    """`OcrCommentary` objects reprensent a single ocr-run of on a commentary, i.e. a collection of page OCRed pages."""

    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: str = None,
                 ocr_dir: str = None,
                 via_path: str = None,
                 image_dir: str = None,
                 groundtruth_dir: str = None,
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
        self.id = id
        self.paths = {'ocr_dir': ocr_dir,
                      'via_path': via_path,
                      'image_dir': image_dir,
                      'groundtruth_dir': groundtruth_dir}

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
                   via_path=os.path.join(base_dir, id, variables.PATHS['via_path']),
                   image_dir=os.path.join(base_dir, id, variables.PATHS['png']),
                   groundtruth_dir=os.path.join(base_dir, id, variables.PATHS['groundtruth']))

    @lazy_property
    def pages(self) -> List['OcrPage']:
        """The pages contained in the commentaries"""
        pages = []
        for file in [f for f in os.listdir(self.paths['ocr_dir']) if f[-4:] in ['.xml', 'hocr', 'html']]:
            pages.append(OcrPage(ocr_path=os.path.join(self.paths['ocr_dir'], file),
                                 id=file.split('.')[0],
                                 image_path=get_path_from_id(file.split('.')[0], self.paths['image_dir']),
                                 commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property  # Todo : This should not be maintained anymore
    def ocr_groundtruth_pages(self) -> Union[List['OcrPage'], list]:
        """The commentary's pages which have a groundtruth file in `self.paths['groundtruth']`."""
        pages = []
        for file in [f for f in os.listdir(self.paths['groundtruth_dir']) if f.endswith('.html')]:
            pages.append(OcrPage(ocr_path=os.path.join(self.paths['groundtruth_dir'], file),
                                 id=file.split('.')[0],
                                 image_path=get_path_from_id(file.split('.')[0], self.paths['image_dir']),
                                 commentary=self))

        return sorted(pages, key=lambda x: x.id)

    @lazy_property
    def via_project(self) -> dict:
        with open(self.paths['via_path'], 'r') as file:
            return json.load(file)

    def to_canonical(self):
        can = TextContainer(id=self.id,
                            type='commentary',
                            word_range=None,  # some_list[:] == some_list[slice(None, None)]
                            # image_range=[(0, len(self.pages))],
                            images=[],
                            children={k: [] for k in variables.TEXTCONTAINER_TYPES},
                            parents={k: [] for k in variables.TEXTCONTAINER_TYPES},
                            commentary=None
                            )

        w_count, p_count, r_count, l_count = 0, 0, 0, 0

        for p in self.pages:
            p.readjust_coords(0.9)  # Todo ⚠️:  this is changing the coords of the page. Should be done on a new object.
            p_start = w_count
            for r in p.regions:
                r_start = w_count
                for l in r.lines:
                    l_start = w_count
                    for w in l.words:
                        can.children['word'].append(CanWord(id=w_count,
                                                            coords=[w.coords],
                                                            text=w.text,
                                                            commentary=can))
                        w_count += 1

                    can.children['line'].append(TextContainer(id=l_count,
                                                              type='line',
                                                              word_range=[(l_start, w_count - 1)],
                                                              # as we have already incremented for the following word
                                                              # image_range=[(p_count, p_count)],
                                                              commentary=can))
                    l_count += 1

                can.children['region'].append(TextContainer(id=r_count,
                                                            type='region',
                                                            word_range=[(r_start, w_count - 1)],
                                                            # as we have already incremented for the following word
                                                            # image_range=[(p_count, p_count)],
                                                            commentary=can,
                                                            info={'region_type': r.region_type}))
                r_count += 1

            can.children['page'].append(TextContainer(id=p.id,
                                                      type='page',
                                                      word_range=[(p_start, w_count - 1)],
                                                      # as we have already incremented for the following word
                                                      # image_range=[(p_count, p_count)],
                                                      commentary=can))
            can.images.append(Image(id=p.id,
                                    path=p.paths['image_path'],
                                    word_range=[(p_start, w_count - 1)]))
            p_count += 1

        # Post initiatiation commentary and of word_range (you can't know it before in this special case).
        can.commentary = can
        can.children['commentary'].append(can)  # as commentary included itself
        can.parents['commentary'].append(can)  # as commentary included itself
        can.word_range = [(0, w_count)]

        return can


class OcrPage:
    """A class representing a commentary page."""

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

        self.id = id
        self.commentary = commentary
        self.paths = {
            'ocr_path': ocr_path,
            'image_path': image_path,
        }

    @lazy_property
    def ocr_format(self) -> str:
        return guess_ocr_format(self.paths['ocr_path'])

    @lazy_property
    def regions(self):
        """Gets page regions, removing empty regions and reordering them."""
        return [OlrRegion.from_via(r, self) for r in
                get_page_region_dicts_from_via(self.id, self.commentary.via_project)]

    @lazy_property
    def lines(self) -> List['OcrLine']:
        return [OcrLine(markup=l, page=self, ocr_format=self.ocr_format)
                for l in find_all_elements(self.markup, 'lines', self.ocr_format)]

    @lazy_property
    def words(self) -> List['OcrWord']:
        return [w for l in self.lines for w in l.words]

    @lazy_property
    def image(self) -> Image:
        return Image(id=self.id, path=self.paths['image_path'])

    @lazy_property
    def text(self) -> str:
        return '\n'.join([line.text for line in self.lines])

    @lazy_property
    def markup(self) -> bs4.BeautifulSoup:
        return parse_markup_file(self.paths['ocr_path'])

    @lazy_property
    def canonical_data(self) -> Dict[str, Any]:
        """Creates canonical data, as used for INCEpTION. """
        logger.warning('You are creating a canonical data version 1. For version two, use `commentary.to_canonical`.')
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
        """Validate `self.canonical_data` and serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        jsonschema.validate(instance=self.canonical_data, schema=schema)

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(self.canonical_data, f, indent=4, ensure_ascii=False)

    def readjust_coords(self,
                        do_debug: bool = False):
        # This assumes we are starting from the untouched OCR output
        # This is the central coords and space processing function.

        # READJUST COORDS AND TRIM EMPTY ELEMENTS
        ## Process lines and words simultaneously, because the latter are included in the former in ocr-outputs
        temp_lines = []
        for l in self.lines:
            # Process words
            temp_words = []
            for w in l.words:
                if re.sub(r'\s+', '', w.text) != '':  # Remove empty words
                    w.adjust_coords_to_included_contours()
                    temp_words.append(w)

            l.words = temp_words

            if l.words:
                # adjust line boxes
                l.adjust_coords_to_included_words()
                temp_lines.append(l)

        self.lines = temp_lines

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/1_wl_{self.id}.png")

        ## Process regions coords
        temp_regions = []
        for r in self.regions:
            if r.region_type not in ['undefined', 'line_number_commentary'] and r.words:  ### Remove excluded regions
                ### Readjust region coords with contained words
                r.adjust_coords_to_included_words()
                temp_regions.append(r)

        self.regions = temp_regions

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/2_r_{self.id}.png")

        # CUT LINES ACCORDING TO REGIONS
        for r in self.regions:
            r.lines = []

            for l in self.lines:
                # If the line is entirely in the region, append it
                if is_rectangle_within_rectangle(contained=l.coords.bounding_rectangle,
                                                 container=r.coords.bounding_rectangle):
                    l.region = r  # Link the line to its region # Todo assert we have no overlapping regions.
                    r.lines.append(l)

                # If the line is only partially in the region, handle the line splitting problem.
                elif any([is_rectangle_within_rectangle(w.coords.bounding_rectangle, r.coords.bounding_rectangle)
                          for w in l.words]):

                    # Create the new line and append it both to region and page lines
                    l_ = copy.copy(l)
                    l_.words = [w for w in l.words if is_rectangle_within_rectangle(w.coords.bounding_rectangle,
                                                                                    r.coords.bounding_rectangle)]
                    l_.coords = Shape([xy for w in l_.words for xy in w.coords.bounding_rectangle])
                    l_.region = r
                    r.lines.append(l_)
                    # self.lines.append(l_)

                    # Actualize the old line
                    l.words = [w for w in l.words if w not in l_.words]
                    l.coords = Shape([xy for w in l.words for xy in w.coords.bounding_rectangle])

            r.lines.sort(key=lambda x: x.coords.xywh[1])

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/3_r_{self.id}.png")

        # Actualize global page reading order
        ## Create fake regions for lines with no regions
        for l in self.lines:
            if not hasattr(l, 'region'):
                line_region = OlrRegion(region_type='line_region', coords=l.coords, page=self)
                line_region.lines = [l]  # todo ⚠️
                self.regions.append(line_region)

        self.regions = sort_to_reading_order(elements=self.regions)
        self.lines = [l for r in self.regions for l in r.lines]
        self.words = [w for l in self.lines for w in l.words]

        if do_debug:
            _ = draw_page_regions_lines_words(self.image.matrix.copy(), self,
                                              f"/Users/sven/Desktop/4_r_{self.id}.png")


class OlrRegion:
    """A class representing OLR regions.

    `OlrRegion`s can be instantiated from a via-dictionary or manually.

    Attributes:

        region_type (str):
            The type of the region, e.g. 'page_number', 'introduction'...

        coords (Shape):
            A `Shape` object representing the coordinates of the region as extracted from via.

        page (OcrPage):
            The `OcrPage` object to which the region belongs

        words:
            The words included in the region.
    """

    @docstring_formatter(**docstrings)
    def __init__(self,
                 region_type: str,
                 coords: Shape,
                 page: 'OcrPage'):
        """Default constructor.

        Args:
            region_type: {olr_region_type}
            coords: {coords_single}
            page: {parent_page}
        """

        self.region_type = region_type
        self.coords = coords
        self.page = page
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
                   coords=Shape.from_xywh(x=via_dict['shape_attributes']['x'],
                                          y=via_dict['shape_attributes']['y'],
                                          w=via_dict['shape_attributes']['width'],
                                          h=via_dict['shape_attributes']['height']),
                   page=page)

    # =================== Parents and children ================

    @lazy_property
    def lines(self):
        if not hasattr(self, '_lines'):
            logger.warning("""You are calling `regions.lines` from un-optimised coordinates. Lines may overcross 
            multiple regions.""")
            return [l for l in self.page.lines
                    if is_rectangle_within_rectangle_with_threshold(contained=l.coords.bounding_rectangle,
                                                                    container=self.coords.bounding_rectangle,
                                                                    threshold=self._inclusion_threshold)]
        else:
            return self._lines

    @lazy_property
    def words(self):
        return [w for w in self.page.words
                if is_rectangle_within_rectangle_with_threshold(contained=w.coords.bounding_rectangle,
                                                                container=self.coords.bounding_rectangle,
                                                                threshold=self._inclusion_threshold)]

    @lazy_property
    def image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    @lazy_property
    def text(self):
        return '\n'.join([w.text for w in self.lines])

    def adjust_coords_to_included_words(self):
        # We start with words as lines can be overlapping two regions.
        words_points = [xy for w in self.words for xy in w.coords.bounding_rectangle]
        self.coords = Shape(words_points) if words_points else self.coords


class OcrLine:
    """Class for OCRlines."""

    def __init__(self, markup: 'bs4.element.Tag', page: OcrPage, ocr_format: str):
        self.markup = markup
        self.page = page
        self.ocr_format = ocr_format

    @lazy_property
    def coords(self):
        return get_element_coords(self.markup, self.ocr_format)

    @lazy_property
    def image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    @lazy_property
    def words(self):
        return [OcrWord(el, self.page, self.ocr_format)
                for el in find_all_elements(self.markup, 'words', self.ocr_format)]

    @lazy_property
    def text(self):
        return ' '.join([w.text for w in self.words])

    def adjust_coords_to_included_words(self):
        words_points = [xy for w in self.words for xy in w.coords.bounding_rectangle]
        self.coords = Shape(words_points) if words_points else self.coords
        # todo : here and above, you do not need to keep all the points


class OcrWord:
    """Class for Words."""

    def __init__(self, markup: 'bs4.element.Tag', page: OcrPage, ocr_format: str):
        self.markup = markup
        self.page = page
        self.ocr_format = ocr_format

    @lazy_property
    def coords(self) -> Shape:
        return get_element_coords(self.markup, self.ocr_format)

    @lazy_property
    def image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    @lazy_property
    def text(self):
        return get_element_text(self.markup, self.ocr_format)

    def adjust_coords_to_included_contours(self):
        self.coords = adjust_to_included_contours(self.coords.bounding_rectangle, self.page.image.contours)


class TextContainer:

    def __init__(self,
                 id: Union[str, int],
                 type: str,
                 word_range: List[Tuple[int, int]],
                 commentary: 'TextContainer',
                 info: Optional = None,
                 coords: Optional[object] = None,  # @lazy_property
                 children: Optional[Dict[str, List['TextContainer']]] = None,  # @lazy_property
                 parents: Optional[Dict[str, List['TextContainer']]] = None,  # @lazy_property
                 images: Optional[object] = None,  # @lazy_property
                 ):

        self.id = id
        self.type = type
        self.word_range = word_range
        self.commentary = commentary
        self.info = info

        for arg in ['coords', 'children', 'parents', 'images']:
            if locals()[arg] is not None:
                setattr(self, arg, locals()[arg])

    @lazy_property
    def coords(self) -> List[Shape]:  # todo 👁️ this does not work for multiple page elements.
        if self.type == 'page':
            return [Shape.from_xywh(0, 0, self.images[0].width, self.images[0].height)]
        else:
            return [Shape(get_bounding_rectangle_from_points(
                [xy for w in self.children['word'] for xy in w.coords[0].bounding_rectangle]))]

    @lazy_property
    def word_slices(self) -> List[slice]:
        return [slice(first, last + 1) for first, last in self.word_range]

    # todo 👁️ this computes every children at once
    # todo ⚠️ Children contains self
    @lazy_property
    def children(self) -> Dict[str, List[Union['TextContainer']]]:
        """Gets all the `TextContainer`s entirely included in `self`.

                Note:
                    - This methods works with word ranges, NOT with coordinates.
                    - This methods will NOT retrieve elements which overlap only partially with `self`.
        """
        children = {}
        for type, tcs in self.commentary.children.items():
            if type != 'word':
                children[type] = [tc for tc in tcs if are_intervals_within_intervals(contained=tc.word_range,
                                                                                     container=self.word_range)]
            else:
                children[type] = [w for ws in self.word_slices for w in tcs[ws]]

        return children

    # todo ⚠️ Parents contains self
    @lazy_property
    def parents(self) -> Dict[str, List[Union['TextContainer']]]:
        """Gets all the `TextContainer`s in which `self` is entirely included."""
        parents = {}
        for type, tcs in self.commentary.children.items():
            if type != 'word':
                parents[type] = [tc for tc in tcs if are_intervals_within_intervals(contained=self.word_range,
                                                                                    container=tc.word_range)]

        return parents

    @lazy_property
    def images(self):
        return [img for img in self.commentary.images if are_intervals_within_intervals(contained=self.word_range,
                                                                                        container=img.word_range)]

    def to_json(self,
                output_dir: str) -> dict:

        data = {'words': [w.to_json() for w in self.children['word']],
                'textcontainers': {},
                'images': [img.id for img in self.images]}

        for tc_type, tcs in self.children.items():
            if tc_type != 'region':
                data['textcontainers'][tc_type] = [{'word_range': tc.word_range for tc in tcs}]
            else:
                data['textcontainers'][tc_type] = [{'word_range': tc.word_range,
                                                    'region_type': tc.info['region_type']} for tc in tcs]

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return data


class CanWord:
    # todo doc
    # todo type hints
    def __init__(self,
                 id,
                 coords,
                 text,
                 commentary,
                 parents: Optional[List[object]] = None,
                 trailing_space_char=None
                 ):
        self.id = id
        self.coords = coords
        self.text = text
        self.commentary = commentary

        if parents:
            self._parents = parents

        if trailing_space_char:
            self._trailing_space_char = trailing_space_char

    def to_json(self):
        return {
            'coords': self.coords.bounding_rectangle_2,
            'text': self.text
        }

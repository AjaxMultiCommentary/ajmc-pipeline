import copy
import json
import os
from abc import ABC, abstractmethod
from time import strftime
from typing import Dict, Optional, List, Union
import bs4.element
from commons.utils import lazy_property
from commons.variables import PATHS
from commons.custom_typing_types import PageType, ElementType, CommentaryType, PathType
from oclr.utils.geometry import (
    Shape,
    is_rectangle_partly_within_rectangle,
    get_bounding_rectangle_from_points,
    is_rectangle_within_rectangle, is_point_within_rectangle
)
from oclr.utils.image_processing import Image
from oclr.utils.region_processing import order_olr_regions, get_page_region_dicts_from_via
import jsonschema as js
from text_importer.file_management import get_page_ocr_path


class Commentary(ABC):
    """Abstract class representing commentary."""

    def __init__(self, commentary_id: str):
        self.id = commentary_id
        # self.commentary_data = {}
        # self.rights = None  # Todo implement

    # ================== `Commentary` level properties =====================
    @lazy_property
    def pages(self):
        return self._get_children('pages')

    @lazy_property
    def regions(self):
        return self._get_children('regions')

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def words(self):
        return self._get_children('words')

    @lazy_property
    def via_project(self):
        with open(os.path.join(PATHS['base_dir'], self.id, PATHS['via_project']), 'r') as file:
            return json.load(file)

    # ================== `Commentary` level methods =====================
    def _get_children(self, name: str) -> List[ElementType]:
        if name == 'pages':
            return [self._create_page(id_) for id_ in self._get_page_ids()]
        if name == "region":
            return [r for p in self.pages for r in p.regions]
        elif name == 'lines':
            return [l for p in self.pages for l in p.lines]
        elif name == 'words':
            return [w for p in self.pages for w in p.words]
        else:
            raise NotImplementedError

    def _get_page_ids(self) -> List[PathType]:
        """Gets the ids of the pages contained in Commentary by scanning the png files"""
        images_dir = os.path.join(PATHS['base_dir'], self.id, PATHS['png'])
        return [p[:-4] for p in os.listdir(images_dir) if p.endswith('.png')]

    # ================== `Commentary` abstract methods =====================
    @abstractmethod
    def _create_page(self, page_id: str) -> PageType:
        pass


class Element(ABC):
    """
    The base class for objects such as pages, page regions, lines and words.
    """

    def __init__(self, id_: str):
        self.id = id_

    @lazy_property
    def coords(self) -> Shape:
        return self._get_coords()

    @lazy_property
    def image(self):
        return self._get_image()

    @lazy_property
    def markup(self):
        return self._parse()

    @lazy_property
    def text(self):
        return self._get_text()

    @abstractmethod
    def _get_children(self, name: str) -> List['ElementType']:
        pass

    @abstractmethod
    def _get_coords(self) -> Shape:
        pass

    @abstractmethod
    def _get_image(self) -> Image:
        pass

    @abstractmethod
    def _parse(self):
        pass

    @abstractmethod
    def _get_text(self) -> str:
        pass

    @abstractmethod
    def _find_tags(self, name: str) -> List[ElementType]:
        pass


class Page(Element):
    """A class representing a commentary page."""

    def __init__(self, page_id: str, commentary: Optional[CommentaryType] = None):
        super().__init__(page_id)
        self.commentary = commentary if commentary else self._create_commentary()

    # ===============================  `Page` level properties  ===============================
    @lazy_property
    def regions(self):
        """Gets page regions, removing empty regions and reordering them."""
        return self._get_children('regions')

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def words(self):
        return self._get_children('words')

    @lazy_property
    def ocr_path(self):
        return get_page_ocr_path(page_id=self.id, ocr_format=self.__class__.__name__[:-4].lower())

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

    # ===============================  `Page` level abstratmethods   ===========================
    @abstractmethod
    def _create_commentary(self):
        pass

    @abstractmethod
    def _create_element(self, markup: bs4.element.Tag):
        pass

    # ================================  `Page` level functions  ================================
    def _get_text(self):
        return '\n'.join([line.text for line in self.lines])

    def _get_image(self):
        return Image(self.id)

    def to_json(self, output_dir: str, schema_path: str = PATHS['schema']):
        """Validate `self.canonical_data` & serializes it to json."""

        with open(schema_path, 'r') as file:
            schema = json.loads(file.read())

        js.validate(instance=self.canonical_data, schema=schema)

        with open(os.path.join(output_dir, self.id + '.json'), 'w') as f:
            json.dump(self.canonical_data, f, indent=4, ensure_ascii=False)

    def _get_children(self, name: str, **kwargs) -> List[ElementType]:
        """Retrieves included OLR and OCR elements.

        Note :
            Notice that in the scope of this project, 'region' refers to OLR (via) regions AND NOT TO THE OCR REGIONS.
            To get these, please set `name=TextRegion`.

        Args:
            name: The name of the `bs4.element.Tag` to retrieve. Should be `'words'|'lines'|'regions'`

        Returns:
            A list of Elements objects, except for regions, where it returns a list of `Region` objects
        """

        if name in ['regions', 'olr_regions']:
            return self._get_regions(**kwargs)
        elif name == 'lines':
            return [self._create_element(line) for line in self._find_tags('lines')]
        elif name == 'words':
            return [w for l in self.lines for w in l.words]
        else:
            raise NotImplementedError

    def _get_regions(self, region_types: Union[List[str], str] = 'all') -> List['Region']:
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

        return order_olr_regions([r for r in regions if r.words])

        


class Region(Element):
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

    def __init__(self, via_dict: Dict[str, dict], page: PageType, word_inclusion_threshold: float = 0.7):
        """Default constructor.

        Note:
            On the contrary to other `Element` objects, `Region` has a particular initialization that requires some
            computation. As via regions coords are often fuzzy, `Region.coords` are first reconstructed, which
            is done by :

                1. Finding the words in `page` that are contained in the via_coords. This is done using
                `is_rectangle_partly_within_rectangle` with `word_inclusion_threshold`.
                2. Actualizing the coords to fit the exact bounding rectangle of contained words.

            Hence, `Region.words` and `Region.coords` are instanciated immediately and are no lazy properties.
        """
        super().__init__(id_='')
        self.region_type = via_dict['region_attributes']['text']
        self.page = page
        self._markup = via_dict

        # Get the initial coords 
        self.initial_coords = Shape.from_xywh(x=self.markup['shape_attributes']['x'],
                                              y=self.markup['shape_attributes']['y'],
                                              w=self.markup['shape_attributes']['width'],
                                              h=self.markup['shape_attributes']['height'])
        # Find the words included
        self._words = [w for w in self.page.words
                       if is_rectangle_partly_within_rectangle(w.coords, self.initial_coords, word_inclusion_threshold)]
        # resize region
        words_points = [xy for w in self._words for xy in w.coords.bounding_rectangle]
        self._coords = Shape(get_bounding_rectangle_from_points(words_points)) \
            if words_points else Shape.from_points([(0, 0)])

    # =================== Parents and children ================
    @lazy_property
    def commentary(self):
        return self.page.commentary

    @lazy_property
    def regions(self):
        """Returns OLR sub-regions contained in self."""
        return self._get_children('regions')

    @lazy_property
    def lines(self):
        return self._get_children('lines')

    @lazy_property
    def words(self):
        return self._words

    def _get_children(self, name: Optional[str] = None, *args, **kwargs) -> List[ElementType]:
        """Retrieves elements that are geographically contained within the region"""

        if 'line' in name.lower():  # Reconstruct region lines
            region_lines = []
            for line in self.page.lines:
                if is_rectangle_partly_within_rectangle(line.coords, self.coords, threshold=0.25):
                    line_ = copy.copy(line)
                    line_words = []
                    for word in line_.words:
                        if is_rectangle_within_rectangle(word.coords.bounding_rectangle, self.coords.bounding_rectangle):
                            line_words.append(word)

                    line_._words = line_words
                    line_points = [xy for w in line_._words for xy in w.coords.bounding_rectangle]
                    line_._coords = Shape(get_bounding_rectangle_from_points(line_points)) \
                        if line_points else Shape.from_points([(0, 0)])  # todo, shouldn't this be the default return of shape ?
                    region_lines.append(line_)

            return region_lines

        else:
            return [el for el in getattr(self.page, name) if
                    is_rectangle_within_rectangle(el.coords.bounding_rectangle, self.coords.bounding_rectangle)]

    def _get_coords(self):
        return self._coords

    def _parse(self):
        return self._markup

    def _get_image(self):
        return self.page.image.crop(self.coords.bounding_rectangle)

    def _get_text(self):
        print('Warning, `Region.text` simply joins `Region.words` together and contains no info about lines !')
        return ' '.join([w.text for w in self.words])

    def _find_tags(self, name: str) -> List[ElementType]:
        raise NotImplementedError





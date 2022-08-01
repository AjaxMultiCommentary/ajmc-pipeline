import json
import os
import re
from typing import Optional, Dict, List, Tuple, Union, Any
from ajmc.commons.arithmetic import is_interval_within_interval
from ajmc.commons.geometry import Shape, get_bbox_from_points
from ajmc.commons.image import Image
from ajmc.commons.miscellaneous import lazy_property, lazy_init, lazy_attributer
from jinja2 import Environment, FileSystemLoader
from ajmc.commons import variables


class CanonicalTextContainer:

    @lazy_init
    def __init__(self,
                 commentary: Optional['CanonicalTextContainer'] = None,
                 type: Optional[str] = None,
                 id: Optional[str] = None,
                 index: Optional[int] = None,
                 word_range: Optional[Tuple[int, int]] = None,
                 word_ranges: Optional[List[Tuple[int, int]]] = None,
                 word_slice: Optional[slice] = None,
                 word_slices: Optional[List[slice]] = None,
                 children: Dict[str, List['CanonicalTextContainer']] = None,
                 parents: Dict[str, List['CanonicalTextContainer']] = None,
                 bbox=None,
                 bboxes=None,
                 image: Image = None,
                 images: Optional[List[Image]] = None,
                 text: Optional[str] = None,
                 info: Optional[dict] = None):
        pass

    @lazy_property
    def commentary(self) -> 'CanonicalCommentary':
        """Just here for clarity, but simply wraps the value given in `__init__`."""
        return self.commentary

    @lazy_property
    def type(self) -> str:
        """Generic method to get a `CanonicalTextContainer`'s type."""
        return re.findall(r'[A-Z][a-z]+', self.__class__.__name__)[0].lower()

    @lazy_property
    def id(self) -> str:
        """Generic method to create a `CanonicalTextContainer`'s id."""
        return self.type[0] + '_' + str(self.index)

    @lazy_property
    def index(self) -> int:
        """Generic method to get a `CanonicalTextContainer`'s index in its parent commentary's children list."""
        return self.commentary.children[self.type].index(self)

    @lazy_property
    def word_range(self) -> Tuple[int, int]:
        """Just here for clarity, but simply wraps the value given in `__init__`."""
        return self.word_range

    @lazy_property
    def word_ranges(self) -> List[Tuple[int, int]]:
        return [self.word_range]

    @lazy_property
    def word_slice(self) -> slice:
        """Generic method to get a `CanonicalTextContainer`'s id."""
        return slice(self.word_range[0], self.word_range[1] + 1)

    @lazy_property
    def word_slices(self):
        return [self.word_slice]

    @lazy_property
    def children(self) -> Dict[str, List['CanonicalTextContainer']]:
        """Gets all the `CanonicalTextContainer`s entirely included in `self`, except self.

                Note:
                    - This methods works with word ranges, NOT with coordinates.
                    - This methods will NOT retrieve elements which overlap only partially with `self`.
        """
        children = {}
        for type_, tcs in self.commentary.children.items():
            children[type_] = [tc for tc in tcs
                               if is_interval_within_interval(contained=tc.word_range, container=self.word_range)
                               and self.id != tc.id]

        return children

    @lazy_property
    def parents(self) -> Dict[str, List[Union['CanonicalTextContainer']]]:
        """Generic method to get a `CanonicalTextContainer`'s parents, i.e. all the `CanonicalTextContainer`s in which
        `self` is entirely included."""
        parents = {}

        for type, tcs in self.commentary.children.items():
            parents[type] = [tc for tc in tcs
                             if is_interval_within_interval(contained=self.word_range, container=tc.word_range)
                             and tc.id != self.id]

        return parents

    @lazy_property
    def bbox(self) -> Shape:
        """Generic method to get a `CanonicalTextContainer`'s bbox."""
        return Shape(get_bbox_from_points([xy for w in self.children['word'] for xy in w.bbox.bbox]))

    @lazy_property
    def bboxes(self) -> List[Shape]:
        return [self.bbox]

    @lazy_property
    def image(self) -> Image:
        """Generic method to create a `CanonicalTextContainer`'s image."""
        candidates_images = [img for img in self.commentary.images
                             if is_interval_within_interval(contained=self.word_range, container=img.word_range)]
        if len(candidates_images) == 1:
            return candidates_images[0]
        else:
            raise NotImplementedError("""Object possesses multiple images""")

    @lazy_property
    def images(self) -> List[Image]:
        return [self.image]

    # todo 👁️ there should be a possibility to add various space chars at the end of words
    # todo 👁️ there should be a possibility to de-hyphenate
    @lazy_property
    def text(self: 'CanonicalTextContainer') -> str:
        """Generic method to get a `CanonicalTextContainer`'s text."""
        return ' '.join([w.text for w in self.children['word']])

    def to_json(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        """Generic method to generate a `CanonicalTextContainer`'s canonical representation."""
        data = {'id': self.id, 'word_range': self.word_range}
        if hasattr(self, 'info'):
            data['info'] = self.info
        return data

class CanonicalCommentary(CanonicalTextContainer):

    def __init__(self,
                 id: str,
                 children: dict,
                 images: List[Image],
                 **kwargs):
        super().__init__(id=id, children=children, images=images, **kwargs)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as file:
            can_json = json.loads(file.read())

        comm = cls(id=can_json['metadata']['id'],
                   images=[Image(id=img['id'], path=img['path'], word_range=img['word_range']) for img in
                           can_json['images']],
                   children={})

        comm.children = {type_: [types_to_classes[type_](type=type_, commentary=comm, **tc) for tc in tcs]
                         for type_, tcs in can_json['textcontainers'].items()}

        return comm

    @lazy_property
    def commentary(self):
        return self

    @lazy_property
    def index(self) -> int:
        return 0

    @lazy_property
    def word_range(self):
        return 0, len(self.children['word'])

    @lazy_property
    def parents(self):
        return {}

    # Todo ⚠️ create json schema
    def to_json(self, output_path: Optional[str] = None) -> dict:
        data = {'metadata': {'id': self.id, 'ocr_run': self.info['ocr_run']},
                'images': [{'id': img.id, 'path': img.path, 'word_range': img.word_range} for img in self.images],
                'textcontainers': {type_: [tc.to_json() for tc in tcs] for type_, tcs in self.children.items()}}

        if output_path is None:
            output_path = os.path.join(variables.PATHS['base_dir'], self.id, 'canonical/v2/',
                                       self.info['ocr_run'] + '.json')

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return data

    def to_alto(self,
                children_types: List[str],
                output_dir: str):
        """Exports self.children['page'] to alto."""

        for p in self.children['page']:
            p.to_alto(children_types=children_types, output_path=os.path.join(output_dir, p.id + '.xml'))


class CanonicalPage(CanonicalTextContainer):

    def __init__(self, id: str,
                 word_range: Tuple[int, int],
                 commentary: CanonicalTextContainer,
                 **kwargs):
        super().__init__(id=id, word_range=word_range, commentary=commentary, **kwargs)

    @lazy_property
    def bbox(self):
        return Shape.from_xywh(0, 0, self.image.width, self.image.height)

    def to_alto(self,
                children_types: List[str],
                output_path: str):
        """Exports a page to ALTO-xml.

        Args:
            children_types: The list of sub-page element-types you want to includ, e.g. `['region', 'line']`.
            output_path: self-explanatory.
        """
        file_loader = FileSystemLoader('data/templates')
        env = Environment(loader=file_loader)
        template = env.get_template('alto.xml.jinja2')

        with open(output_path, 'w') as f:
            f.write(template.render(page=self, elements=children_types))


class CanonicalSinglePageTextContainer(CanonicalTextContainer):

    def __init__(self,
                 type: str,
                 word_range: Tuple[int, int],
                 commentary: CanonicalTextContainer,
                 info: Optional[None] = None,
                 **kwargs):
        super().__init__(type=type, word_range=word_range, commentary=commentary, info=info, **kwargs)


class CanonicalWord(CanonicalTextContainer):

    def __init__(self,
                 text: str,
                 bbox: List[Tuple[int, int]],
                 commentary: CanonicalTextContainer,
                 **kwargs):
        super().__init__(text=text, commentary=commentary, **kwargs)
        self.bbox = Shape(bbox)

    @lazy_property
    def word_range(self):
        return self.index, self.index

    def to_json(self):
        return {'id': self.id, 'bbox': self.bbox.bbox_2, 'text': self.text}


types_to_classes = {'commentary': CanonicalCommentary,
                    'page': CanonicalPage,
                    'region': CanonicalSinglePageTextContainer,
                    'line': CanonicalSinglePageTextContainer,
                    'word': CanonicalWord}

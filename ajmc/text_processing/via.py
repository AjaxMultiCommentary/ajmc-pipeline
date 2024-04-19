import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ajmc.commons import variables as vs, geometry as geom
from ajmc.commons.miscellaneous import get_ajmc_logger

logger = get_ajmc_logger(__name__)


class ViaProject:
    """Class for creating a VIA project file.

    Note:
        Handles only rectangle regions (not polygon regions)

    """

    def __init__(self,
                 name: str = '',
                 default_path: Union[Path, str] = '',
                 attributes: Optional[List[Dict[str, str]]] = None):
        """Create a ViaProject object.

        Args:
            name (str): The name of the project.
            default_path (Path or str): The default path for the project.
            attributes (list[dict]): The region attributes for the project. Each dict must have the keys 'name', 'level', 'type', 'description', and
            'default_value'. See ``ViaProject.add_attribute()`` for more details.
        """

        # Load the template project file
        self.project_dict = json.loads((vs.PACKAGE_DIR / 'ajmc/data/templates/via_project.json').read_text())
        # Set the project name
        self.project_dict['_via_settings']['project']['name'] = name
        # Set the default path
        if default_path:
            self.project_dict['_via_settings']['core']['default_filepath'] = str(Path(default_path)) + '/'

        # Add the region attributes
        if attributes:
            for attr_dict in attributes:
                self.add_attribute(**attr_dict)

    @classmethod
    def from_json(cls, json_path: Path):
        """Create a ViaProject from an existing via project in a JSON file.

        Args:
            json_path (Path): The path to the JSON file.

        Returns:
            ViaProject: The ViaProject object.

        """
        vp = cls()
        vp.project_dict = json.loads(json_path.read_text())
        return vp

    @classmethod
    def from_canonical_commentary(cls, can_com: 'CanonicalCommentary'):
        """Create a ViaProject from a CanonicalCommentary.

        Args:
            can_com (CanonicalCommentary): The commentary.

        Returns:
            ViaProject: The ViaProject object.
        """

        base_path = vs.get_comm_img_dir(can_com.id)
        via_project = ViaProject(can_com.id,
                                 base_path,
                                 [
                                     {'name': 'label', 'level': 'region', 'type': 'text', 'default_value': ''},
                                     {'name': 'is_ground_truth', 'level': 'file', 'type': 'checkbox',
                                      'options': {'ocr': '', 'olr': ''}},
                                 ])
        for page in can_com.children.pages:
            region_shapes = []
            region_attributes = []
            file_attributes = {}

            for w in page.children.words:
                word_region_attributes = {'label': w.text}
                region_shapes.append(w.bbox)
                region_attributes.append(word_region_attributes)

            for l in page.children.lines:
                line_region_attributes = {'label': vs.OLR_PREFIX + 'line_region'}
                region_shapes.append(l.bbox)
                region_attributes.append(line_region_attributes)

            for r in page.children.regions:
                if r.region_type not in vs.EXCLUDED_REGION_TYPES:
                    region_shapes.append(r.bbox)

                    if r.is_ocr_gt:
                        region_attributes.append({'label': vs.OLR_PREFIX + vs.OCR_GT_PREFIX + r.region_type})
                    else:
                        region_attributes.append({'label': vs.OLR_PREFIX + r.region_type})

            file_attributes['is_ground_truth'] = {'ocr': page.id in can_com.ocr_gt_page_ids,
                                                  'olr': page.id in can_com.olr_gt_page_ids}

            via_project.add_image(page.image.path.relative_to(base_path),
                                  region_shapes=region_shapes,
                                  region_attributes=region_attributes,
                                  file_attributes=file_attributes)

        return via_project


    def __getitem__(self, item):
        return self.project_dict[item]

    def __setitem__(self, key, value):
        self.project_dict[key] = value


    def add_attribute(self, name: str,
                      level: str,
                      type: str,
                      description: str = '',
                      default_value: str = '',
                      options: Optional[List[str]] = None):
        """Add a region attribute to the project.

        Args:
            name (str): The name of the attribute.
            level (str): The level of the attribute. Must be one of 'region', 'file'.
            type (str): The type of the attribute. Must be one of 'text', 'dropdown', 'radio', 'image', 'checkbox'.
            description (str): The description of the attribute.
            default_value (str): The default value of the attribute.
            options (list[str]): The options for the attribute. Only required if type is 'dropdown', 'radio', or 'checkbox'.

        """
        self.project_dict['_via_attributes'][level][name] = {'type': type,
                                                             'description': description,
                                                             'default_value': default_value}
        if options:
            self.project_dict['_via_attributes'][level][name]['options'] = options

    def add_image(self, image_path: Path,
                  region_shapes: List,
                  region_attributes: List[Dict[str, str]],
                  file_attributes: Optional[Dict[str, Dict]] = None):
        """Add an image and its regions to the project.

        Args:
            image_path (Path): The absolute or relative path to the image. If relative, it must be relative to the project's default path.
            region_shapes (list[Shape]): The regions of the image.
            region_attributes (list[dict]): The attributes of the regions, in the form {'attr_name': 'attr_value'}.
            file_attributes (dict): The attributes of the file, in the form {'attr_name': 'attr_value'}.

        """

        # Get the image's size
        image_abs_path = Path(self.project_dict['_via_settings']['core']['default_filepath']) / image_path
        image_size = Path(image_abs_path).stat().st_size

        # Get the image's name
        image_name = str(image_path) + str(image_size)

        # Create the regions list
        assert len(region_shapes) == len(region_attributes), """The number of region shapes must be equal to the number of region attributes."""
        regions = []
        for shape, attributes in zip(region_shapes, region_attributes):
            region = {'shape_attributes': {'name': 'rect',
                                           'x': shape.xywh[0],
                                           'y': shape.xywh[1],
                                           'width': shape.width,
                                           'height': shape.height},
                      'region_attributes': attributes}
            regions.append(region)

        # Add the image to the project
        self.project_dict['_via_img_metadata'][image_name] = {'filename': str(image_path),
                                                              'size': image_size,
                                                              'regions': regions,
                                                              'file_attributes': file_attributes if file_attributes else {}}
        self.project_dict['_via_image_id_list'].append(image_name)


    def _is_not_duplicate(self, region, texts, bboxes):
        region_bbox = geom.Shape.from_via(region)
        for bbox, text in zip(bboxes, texts):
            if (bbox == region_bbox.bbox or (geom.are_bboxes_overlapping_with_threshold(region_bbox.bbox, bbox.bbox, 0.7)
                                             and region['region_attributes']['label'] == text)):
                return False

        return True


    def check_page_duplicates(self, page_dict: Dict[str, Any]):

        pruned_words = []
        words_boxes = []
        words_texts = []

        pruned_lines = []
        lines_boxes = []
        lines_texts = []

        pruned_regions = []
        regions_boxes = []
        regions_texts = []

        total_words = 0
        total_lines = 0
        total_regions = 0

        for region in page_dict['regions']:
            if region['region_attributes']['label'].startswith(vs.OLR_PREFIX):
                if 'line_region' in region['region_attributes']['label']:
                    total_lines += 1
                    if self._is_not_duplicate(region, lines_texts, lines_boxes):
                        pruned_lines.append(region)
                        lines_boxes.append(geom.Shape.from_via(region))
                        lines_texts.append(region['region_attributes']['label'])
                else:
                    total_regions += 1
                    if self._is_not_duplicate(region, regions_texts, regions_boxes):
                        pruned_regions.append(region)
                        regions_boxes.append(geom.Shape.from_via(region))
                        regions_texts.append(region['region_attributes']['label'])
            else:
                total_words += 1
                if self._is_not_duplicate(region, words_texts, words_boxes):
                    pruned_words.append(region)
                    words_boxes.append(geom.Shape.from_via(region))
                    words_texts.append(region['region_attributes']['label'])

        diffs = {'words': total_words - len(pruned_words),
                 'lines': total_lines - len(pruned_lines),
                 'regions': total_regions - len(pruned_regions)
                 }

        if any(diffs.values()):
            print(f'{page_dict["filename"]} - {diffs["words"]} words, {diffs["lines"]} lines, {diffs["regions"]} regions.')

        return pruned_words + pruned_lines + pruned_regions

    def prune_duplicates(self):
        for page_dict in self.project_dict['_via_img_metadata'].values():
            page_dict['regions'] = self.check_page_duplicates(page_dict)

    def clean(self):
        self.prune_duplicates()

        # Automatically clean obvious typos
        for page_dict in self.project_dict['_via_img_metadata'].values():
            for region in page_dict['regions']:
                region['region_attributes']['label'] = re.sub(r'\s+', '', region['region_attributes']['label'])

    def safe_check(self, check_duplicates: bool = True,
                   check_overlapping_regions: bool = True,
                   check_typos: bool = True):
        print('Safe-checking the via project, this may take a while....')
        is_overlap_free = True
        is_typo_free = True

        if check_duplicates:
            print('********************** Checking for duplicates **********************')
            for page_dict in self.project_dict['_via_img_metadata'].values():
                self.check_page_duplicates(page_dict)

        if check_overlapping_regions:
            print('********************** Checking for overlapping regions **********************')

            for page_dict in self.project_dict['_via_img_metadata'].values():
                regions = [r for r in page_dict['regions']
                           if r['region_attributes']['label'].startswith(vs.OLR_PREFIX)
                           and not any([r_type in r['region_attributes']['label'] for r_type in vs.EXCLUDED_REGION_TYPES])]

                region_bboxes = []

                for region in regions:
                    region_bbox = geom.Shape.from_via(region)
                    if any(geom.compute_bbox_overlap_area(region_bbox.bbox, bbox.bbox) > min(region_bbox.area, bbox.area) * 0.4
                           for bbox in region_bboxes):
                        print(f'{page_dict["filename"]} - {region["region_attributes"]["label"]} overlaps with another region.')
                        is_overlap_free = False
                    region_bboxes.append(region_bbox)

        if check_typos:
            print('********************** Checking for typos **********************')

            for p in self.project_dict['_via_img_metadata'].values():
                for r in p['regions']:
                    region_label = r['region_attributes']['label']
                    if region_label.startswith(vs.OLR_PREFIX):
                        region_label = region_label.replace(vs.OLR_PREFIX, '').replace(vs.OCR_GT_PREFIX, '')
                        if region_label not in vs.ORDERED_OLR_REGION_TYPES:
                            is_typo_free = False
                            print(Path(p['filename']).stem, '-', 'Unknown region', r['region_attributes']['label'])

        return is_overlap_free and is_typo_free

    def save(self, output_path: Path):
        """Save the project to a JSON file.

        Args:
            output_path (Path): The path to save the project to.

        """
        output_path.write_text(json.dumps(self.project_dict, ensure_ascii=False, indent=2))
        print(f'Saved via file to {output_path}')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Create, clean and safe-check via projects.')

    parser.add_argument('--comm_ids', nargs='+', help='The commentary ids to process.', default=vs.ALL_COMM_IDS)
    parser.add_argument('--clean', action='store_true', help='Clean the via project, removing duplicates and typos')
    parser.add_argument('--safe_check', action='store_true', help='Safe-check the via project for duplicates, typos and overlapping regions.')
    parser.add_argument('--save', action='store_true', help='Save the via project.')

    args = parser.parse_args()


    ##############################
    # Debugging
    # args = argparse.Namespace()
    # args.comm_ids = ['sophokle1v3soph']
    # args.clean = True
    # args.safe_check = True
    # args.save = True
    ##############################

    for comm_id in args.comm_ids:
        via_path = vs.get_comm_via_path(comm_id)
        print(f'Using via file {via_path}')

        print(f'Processing {comm_id}...')
        via_project = ViaProject.from_json(via_path)

        if args.clean:
            via_project.clean()

        if args.safe_check:
            is_safe = via_project.safe_check(check_duplicates=not args.clean)

        if args.save:
            via_project.save(via_path)

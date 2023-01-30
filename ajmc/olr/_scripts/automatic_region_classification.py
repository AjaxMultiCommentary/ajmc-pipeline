"""A proto-script to automatically classify regions"""

MISSING_XMIS_COMMS = [
    'annalsoftacitusp00taci',
    'bsb10234118',
    'Colonna1975',
    'DeRomilly1976',
    'Ferrari1974',
    'Garvie1998',
    'Kamerbeek1953',
    'Paduano1982',
    'pvergiliusmaroa00virggoog',
    'thukydides02thuc',
    'Untersteiner1934',
]

COMM_ID = 'Stanford1963'
OVERLAP_THRESHOLD = 0.6
REGIONS_PER_PAGE = 1
DESIRED_REGION = 'commentary'
DESIRED_SECTION = 'commentary'
MINIMAL_AREA_FACTOR = 0.5
MAXIMAL_AREA_FACTOR = 1.2


# Functions to get regions
def overlap_criterion(gt_region, candidate_region) -> bool:
    return are_bboxes_overlapping_with_threshold(gt_region['shape'].bbox, candidate_region['shape'].bbox,
                                                 threshold=OVERLAP_THRESHOLD)


def area_criterion(gt_region, candidate_region) -> bool:
    gt_area = gt_region['shape'].area
    candidate_area = candidate_region['shape'].area
    return MINIMAL_AREA_FACTOR * gt_area < candidate_area < MAXIMAL_AREA_FACTOR * gt_area


def horizontal_position_criterion(gt_region, candidate_region) -> bool:
    top_limit = gt_region['shape'].bbox[0][1] - 100
    bottom_limit = gt_region['shape'].bbox[1][1] + 100
    return (top_limit < candidate_region['shape'].bbox[0][1] < bottom_limit) and (
                top_limit < candidate_region['shape'].bbox[1][1] < bottom_limit)


CRITERIA = [
    overlap_criterion,
    area_criterion,
    # horizontal_position_criterion,
]

import json
# read section
from pathlib import Path

from ajmc.commons import variables

comm_dir = Path(variables.COMMS_DATA_DIR) / COMM_ID
sections_path = comm_dir / 'sections.json'
sections = json.loads(sections_path.read_text(encoding='utf-8'))
section = [s for s in sections if DESIRED_SECTION in s['section_type']][0]
section['start'] = int(section['start'])
section['end'] = int(section['end'])

# read via project
via_project_path = comm_dir / 'olr/via_project.json'
via_project = json.loads(via_project_path.read_text(encoding='utf-8'))

# get the layout template
# We always take the second page as the first might have a title or layout variation
layout_template_number = section['start'] + 1
layout_template = [p_dict for p_dict in via_project['_via_img_metadata'].values()
                   if layout_template_number == int(p_dict['filename'].split('_')[-1].split('.')[0])][0]

template_regions = [r_dict for r_dict in layout_template['regions']
                    if r_dict['region_attributes']['text'] != 'undefined']

# get the regions' shapes
from ajmc.commons.geometry import Shape, are_bboxes_overlapping_with_threshold


def shape_from_via_region(via_region: dict) -> Shape:
    return Shape.from_xywh(x=via_region['shape_attributes']['x'],
                           y=via_region['shape_attributes']['y'],
                           w=via_region['shape_attributes']['width'],
                           h=via_region['shape_attributes']['height'])


for temp_region in template_regions:
    temp_region['shape'] = shape_from_via_region(temp_region)

# get the commentary section page from the via_dict
warning_count = 0
for page_dict in via_project['_via_img_metadata'].values():
    page_number = int(page_dict['filename'].split('_')[-1].split('.')[0])
    if (not section['start'] <= page_number <= section['end']) or page_number == layout_template_number:
        continue
    else:
        # main script to compare the overlap of regions

        # give regions a shape
        for r in page_dict['regions']:
            r['shape'] = shape_from_via_region(r)

        for gt_r in template_regions:
            if gt_r['region_attributes']['text'] not in [DESIRED_REGION]:
                continue
            for r in page_dict['regions']:
                if r['region_attributes']['text'] == 'undefined':
                    # print('Warning: undefined region in page {}'.format(page_number))
                    if all([criterion(gt_r, r) for criterion in CRITERIA]):
                            r['region_attributes']['text'] = gt_r['region_attributes']['text']

        for region in page_dict['regions']:
            if 'shape' in region.keys():
                del region['shape']

        comm_regions = len([r for r in page_dict['regions'] if r['region_attributes']['text'] == DESIRED_REGION])
        if comm_regions != REGIONS_PER_PAGE:
            print(f'WARNING: page {page_dict["filename"]}: {comm_regions} {DESIRED_REGION} region(s) detected')
            warning_count += 1

for temp_region in template_regions:
    del temp_region['shape']

print(f'warning_count: {warning_count}')
Path('/Users/sven/Desktop/via_project_test.json').write_text(json.dumps(via_project, ensure_ascii=False),
                                                             encoding='utf-8')

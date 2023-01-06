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

COMM_ID = 'annalsoftacitusp00taci'
OVERLAP_THRESHOLD = 0.6
COMM_REGIONS_PER_PAGE = 2
DESIRED_REGION = 'commentary'
MINIMAL_AREA_FACTOR=0.4
MAXIMAL_AREA_FACTOR=1.5

# read section
from pathlib import Path
import json
from ajmc.commons import variables

comm_dir = Path(variables.PATHS['base_dir']) / COMM_ID
sections_path = comm_dir / 'sections.json'
sections = json.loads(sections_path.read_text(encoding='utf-8'))
comm_section = [s for s in sections if DESIRED_REGION in s['section_type']][0]
comm_section['start'] = int(comm_section['start'])
comm_section['end'] = int(comm_section['end'])


# read via project
via_project_path = comm_dir / 'olr/via_project.json'
via_project = json.loads(via_project_path.read_text(encoding='utf-8'))

# get the layout template
# We always take the second page as the first might have a title or layout variation
layout_template_number = comm_section['start']  #+ 1
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
    if (not comm_section['start'] <= page_number <= comm_section['end']) or page_number == layout_template_number:
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
                    if are_bboxes_overlapping_with_threshold(gt_r['shape'].bbox,
                                                             r['shape'].bbox,
                                                             threshold=OVERLAP_THRESHOLD):
                        if MINIMAL_AREA_FACTOR * gt_r['shape'].area <= r['shape'].area <= MAXIMAL_AREA_FACTOR * gt_r['shape'].area:
                            r['region_attributes']['text'] = gt_r['region_attributes']['text']

        for region in page_dict['regions']:
            if 'shape' in region.keys():
                del region['shape']

        comm_regions = len([r for r in page_dict['regions'] if r['region_attributes']['text'] == DESIRED_REGION])
        if comm_regions != COMM_REGIONS_PER_PAGE:
            print(f'WARNING: page {page_dict["filename"]}: {comm_regions} {DESIRED_REGION} region(s) detected')
            warning_count += 1

for temp_region in template_regions:
    del temp_region['shape']

print(f'warning_count: {warning_count}')
Path('/Users/sven/Desktop/via_project_test.json').write_text(json.dumps(via_project, ensure_ascii=False), encoding='utf-8')

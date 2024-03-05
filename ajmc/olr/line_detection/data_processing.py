from pathlib import Path
from typing import List, Dict

from ajmc.commons.file_management import get_olr_gt_spreadsheet
from ajmc.commons.geometry import Shape


def get_page_ids_from_via(via_dict: dict):
    return sorted(Path(k).stem for k in via_dict['_via_img_metadata'].keys())


def get_split_page_ids(split: str) -> List[str]:
    olr_gt = get_olr_gt_spreadsheet()
    return olr_gt['page_id'][olr_gt['split'] == split].tolist()


def get_pages_lines(split_page_ids: List[str], via_dict: dict) -> Dict[str, List[Shape]]:
    pages_lines = {}
    for page_dict in via_dict['_via_img_metadata'].values():
        page_id = Path(page_dict['filename']).stem
        if page_id in split_page_ids:
            pages_lines[page_id] = [Shape.from_xywh(x=line['shape_attributes']['x'],
                                                    y=line['shape_attributes']['y'],
                                                    w=line['shape_attributes']['width'],
                                                    h=line['shape_attributes']['height'])
                                    for line in page_dict['regions']]

    return pages_lines

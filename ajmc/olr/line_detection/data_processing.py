import json
from pathlib import Path
from typing import List, Dict, Optional

from ajmc.commons import geometry as geom, variables as vs
from ajmc.commons.file_management import get_olr_gt_spreadsheet
from ajmc.commons.geometry import Shape


def get_page_ids_from_via(via_dict: dict):
    return sorted(Path(k).stem for k in via_dict['_via_img_metadata'].keys())


def get_split_page_ids(split: str) -> List[str]:
    olr_gt = get_olr_gt_spreadsheet()
    return olr_gt['page_id'][olr_gt['split'] == split].tolist()


def get_pages_lines(via_dict: dict, split_page_ids: Optional[List[str]] = None) -> Dict[str, List[Shape]]:
    pages_lines = {}
    for page_dict in via_dict['_via_img_metadata'].values():
        page_id = Path(page_dict['filename']).stem
        if split_page_ids is None or page_id in split_page_ids:
            pages_lines[page_id] = [Shape.from_via(line) for line in page_dict['regions']]

    return pages_lines


def balance_dataset(lines_via_path: Path, output_path: Path, max_pages: int = 40, minimum_pages: int = 35) -> Dict[str, List[geom.Shape]]:
    via_dict = json.loads(lines_via_path.read_text(encoding='utf-8'))
    split_page_ids = get_split_page_ids('test') + get_split_page_ids('train')

    pages_lines = {}
    for page_dict in via_dict['_via_img_metadata'].values():
        page_id = Path(page_dict['filename']).stem
        if page_id in split_page_ids:
            pages_lines[page_id] = page_dict

    # Balance commentaries
    import random

    random.seed(0)

    balanced_groundtruth = {}
    for comm_id in vs.ALL_COMM_IDS:
        comm_gt_keys = [k for k in pages_lines.keys() if k.startswith(comm_id)]
        sampled_keys = random.sample(comm_gt_keys, k=min(max_pages, len(comm_gt_keys)))
        if len(sampled_keys) < minimum_pages:
            print(f'Comm {comm_id} has less than {minimum_pages} pages')
            continue
        balanced_groundtruth.update({k: pages_lines[k] for k in sampled_keys})

    output_path.write_text(json.dumps({'_via_img_metadata': balanced_groundtruth}, indent=2))

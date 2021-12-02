"""Functions and helpers to manage the files """

import os
from commons.types import PathType
from commons.variables import PATHS


def get_page_ocr_path(page_id: str, ocr_format: str) -> PathType:
    """Gets the path to a page's ocr"""

    ocr_dir = os.path.join(PATHS['base_dir'], page_id.split('_')[0], PATHS[ocr_format])
    files = [p for p in os.listdir(ocr_dir) if page_id in p]
    assert len(files) == 1, f'There are several filenames containing {page_id} in {ocr_dir}. Please verify.'

    return os.path.join(ocr_dir, files[0])

"""Functions and helpers to manage the files """

import os
from commons.variables import PATHS
from commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)


def get_page_ocr_path(page_id: str, ocr_dir: str = None, ocr_run: str = None) -> str:
    """Gets the path to a page's ocr"""

    if not ocr_dir:
        ocr_dir = os.path.join(PATHS['base_dir'], page_id.split('_')[0], 'ocr/runs', ocr_run, 'outputs')
    files = [f for f in os.listdir(ocr_dir) if page_id in f]

    assert len(files) <= 1, f"""There are {len(files)} files matching the name {page_id} in {ocr_dir}. Please check."""

    if len(files) == 0:
        logger.warning(f"""Page_id {page_id} matches no file in {ocr_dir}, skipping...""")
        return ""

    else:
        return os.path.join(ocr_dir, files[0])


def guess_ocr_format(ocr_path: str) -> str:
    """Guesses the ocr-format of a file.

    Args:
        ocr_path: Absolute path to an ocr output file"""

    if ocr_path[-3:] == 'xml':
        return 'pagexml'
    elif ocr_path[-4:] == 'html':
        return 'krakenhocr'
    elif ocr_path[-4:] == 'hocr':
        return 'tesshocr'
    else:
        raise NotImplementedError("""The format could not be identified. It looks like the format is neither 
        `tesshocr`, nor `krakenhocr` nor `pagexml`, which are the only formats this module deals with.""")

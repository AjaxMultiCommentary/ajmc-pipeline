"""Contains all the necessary utils for the management of spreadsheets."""
import json
import logging
import os
from typing import List, Tuple, Set
from ajmc.commons.variables import SPREADSHEETS_IDS
from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.commons.miscellaneous import get_custom_logger, read_google_sheet
from ajmc.commons import docstrings, variables

logger = get_custom_logger(__name__)


@docstrings.docstring_formatter(**docstrings.docstrings)
def check_via_spreadsheet_conformity(via_project: dict,
                                     sheet_page_ids: List[str],
                                     check_comm_only: bool = False) -> Tuple[Set[str], Set[str]]:
    """Verifies that `via_project` actually contains the page ids listed in `sheet_page_ids` and vice-versa.

    This function is used to make sure that the pages marked as groundtruth in spreadsheets are actually present in the
    respective via_project and vice-versa. If there are differences between the sets of via and spreadsheet pages,
    prints a logging.error. In any case, return the two sets of difference.

    Args:
        via_project: {via_project}
        sheet_page_ids: The list of page ids present in the spreadsheet, i.e. only the page ids of a the commentary
        corresponding to the `via_project`.
        check_comm_only: Whether to check only the pages where only commentary sections are annotated.

    Returns:
         A tuple containing two sets of str:
             1. The difference $sheet_pages - via_pages$.
             2. The difference $via_pages - sheet_pages$.
    """

    via_full_gt_pages = []  # This contains the pages which are entirely annotated
    via_comm_gt_pages = []  # This contains the pages where only commentary sections are annotated

    for v in via_project['_via_img_metadata'].values():

        if any([r['region_attributes']['text'] not in ['commentary', 'undefined'] for r in v['regions']]):
            via_full_gt_pages.append(v['filename'].split('.')[0])

        elif all([r['region_attributes']['text'] in ['commentary', 'undefined'] for r in v['regions']]) and \
                any([r['region_attributes']['text'] in ['commentary'] for r in v['regions']]):
            via_comm_gt_pages.append(v['filename'].split('.')[0])

    via_pages = set(via_comm_gt_pages) if check_comm_only else set(via_full_gt_pages)
    sheet_pages = set(sheet_page_ids)

    diff_sheet_via = sheet_pages.difference(via_pages)
    diff_via_sheet = via_pages.difference(sheet_pages)

    if diff_sheet_via:
        logger.error(f"""The following pages are in annotated in sheet 
        but not in via : \n{sheet_pages.difference(via_pages)}\n""")

    if diff_via_sheet:
        logger.error(f"""The following pages are in annotated in via 
        but not in sheet : \n{via_pages.difference(sheet_pages)}\n""")

    if not diff_sheet_via and not diff_via_sheet:
        logger.info("""Checking passed : The set of via pages equates the set of sheet pages""")

    return diff_sheet_via, diff_via_sheet


def check_entire_via_spreadsheets_conformity(sheet_id:str = SPREADSHEETS_IDS['olr_gt'],
                                             sheet_name:str = 'olr_gt'):
    """Applies `check_via_spreadsheet_conformity` to an entire spreadsheet with multiple commentaries."""

    df = read_google_sheet(sheet_id=sheet_id, sheet_name=sheet_name)
    differences = {}

    for comm_id, comm_df in df.groupby(df['commentary_id']):
        logger.info(f"""Checking commentary {comm_id}""")
        with open(os.path.join(variables.PATHS['base_dir'], comm_id, variables.PATHS['via_path']), 'r') as file:
            via_project = json.load(file)

        differences[comm_id] = check_via_spreadsheet_conformity(via_project=via_project,
                                                                sheet_page_ids=comm_df['page_id'].tolist())

    if not any([s for t in differences.values() for s in t]):
        logger.info("""Checking passed, sheet is conform with vias.""")
    return differences


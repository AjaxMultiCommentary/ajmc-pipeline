"""Use this script to bulk convert ocr runs to canonical commentary"""
import argparse

from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.miscellaneous import ROOT_LOGGER
from ajmc.text_processing.ocr_classes import OcrCommentary


parser = argparse.ArgumentParser()
parser.add_argument('--commentary_ids', nargs='+', help='Commentaries to process', default=vs.ALL_COMM_IDS)
parser.add_argument('--ocr_run_pattern', type=str, help='OCR run pattern to process, eg. *_tess_base',
                    default='*_tess_retrained')
parser.add_argument('--ocr_gt_comms_only', action='store_true', help='Process only commentaries with OCR ground truth')
parser.add_argument('--non_ocr_gt_comms_only', action='store_true',
                    help='Process only commentaries with OCR ground truth')
parser.add_argument('--stream_handler_level', type=str, help='Stream handler level', default='ERROR')
args = parser.parse_args()

ROOT_LOGGER.setLevel(args.stream_handler_level)

for comm_id in tqdm(args.commentary_ids, desc='Processing commentaries'):

    comm = OcrCommentary.from_ajmc_data(id=comm_id, ocr_run_id=args.ocr_run_pattern)

    comm_can_path = vs.get_comm_canonical_default_path(comm_id, ocr_run_id=comm.ocr_run_id)

    # if comm_can_path.exists():
    #     if str(datetime.fromtimestamp(comm_can_path.stat().st_mtime)).startswith('2023-01-31'):
    #         print(f'******* ALREADY CONVERTED {comm_id}')
    #         continue

    if args.ocr_gt_comms_only and not comm.ocr_gt_page_ids:
        continue

    if args.non_ocr_gt_comms_only and comm.ocr_gt_page_ids:
        continue
    comm.to_canonical().to_json()

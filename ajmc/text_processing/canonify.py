"""Use this script to bulk convert ocr runs to canonical commentary"""
import argparse
import json

from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.file_management import get_commit_hash, check_change_in_lemlink, check_change_in_ne
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.text_processing.ocr_classes import OcrCommentary

ROOT_LOGGER.setLevel('DEBUG')

if __name__ == '__main__':
    logger = get_ajmc_logger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--commentary_ids',
                        nargs='+',
                        help='Commentaries to process',
                        default=vs.ALL_COMM_IDS,
                        required=False)
    parser.add_argument('--ocr_run_pattern',
                        type=str,
                        help='OCR run pattern to process, eg. *_tess_base',
                        default='*_tess_retrained',
                        required=False)
    parser.add_argument('--stream_handler_level',
                        type=str,
                        help='Stream handler level',
                        default='ERROR',
                        required=False)
    parser.add_argument('--force',
                        action='store_true',
                        help='Force recanonification',
                        required=False)

    args = parser.parse_args()

    # args.commentary_ids = ['sophoclesplaysa05campgoog']
    # args.force = True
    # args.stream_handler_level = 'DEBUG'

    ROOT_LOGGER.setLevel(args.stream_handler_level)

    for comm_id in tqdm(args.commentary_ids, desc='Processing commentaries'):

        can_path = vs.get_comm_canonical_path_from_pattern(comm_id, ocr_run_pattern=args.ocr_run_pattern)
        if can_path.exists():
            existing_can_json = json.loads(can_path.read_text('utf-8'))

            # see if the via has changed
            if not any([
                # check_change_in_commentary_data(comm_id, existing_can_json['metadata']['commentaries_data_commit'],
                #                                 get_commit_hash(vs.COMMS_DATA_DIR)),
                check_change_in_ne(comm_id, existing_can_json['metadata']['ne_corpus_commit'],
                                   get_commit_hash(vs.NE_CORPUS_DIR)),
                check_change_in_lemlink(comm_id, existing_can_json['metadata']['lemlink_corpus_commit'],
                                        get_commit_hash(vs.LEMLINK_CORPUS_DIR)),
                args.force]):
                continue

        OcrCommentary.from_ajmc_data(id=comm_id, ocr_run_id=args.ocr_run_pattern).to_canonical().to_json()

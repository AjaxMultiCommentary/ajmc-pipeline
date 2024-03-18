"""Use this script to bulk convert ocr runs to canonical commentaries.

Warning:
    This script will first check the via project of each commentary and will skip the commentary if the via project is not safe (i.e. if there are
    overlapping regions, if there are duplicate regions or if there are mispelled regions). Please see the traceback for more information.

"""
import argparse
import json

from ajmc.commons import variables as vs
from ajmc.commons.file_management import get_commit_hash, check_change_in_lemlink, check_change_in_ne, check_change_in_commentary_data
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.text_processing.raw_classes import RawCommentary

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
                        default=vs.DEFAULT_OCR_RUN_ID,
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

    if input("""WARNING: You are about to recanonify commentaries. Did you safe-check VIA files ? [y/n]
         Otherwise, please run `via.py --comm_ids [your comm ids] --clean --safe_check --save` and commit your changes.""") != 'y':
        raise SystemExit

    ################### DEBUG ###################
    # args.commentary_ids = ['sophoclesplaysa05campgoog']
    # args.force = True
    # args.stream_handler_level = 'DEBUG'
    ################### DEBUG ###################

    ROOT_LOGGER.setLevel(args.stream_handler_level)

    unsafe_canonicals = []

    for comm_id in args.commentary_ids:
        print(f'Processing commentary {comm_id}...')

        can_path = vs.get_comm_canonical_path_from_ocr_run_id(comm_id, ocr_run_pattern=args.ocr_run_pattern)
        if can_path.exists():
            existing_can_json = json.loads(can_path.read_text('utf-8'))

            # see if the via has changed
            if not any([
                check_change_in_commentary_data(comm_id, existing_can_json['metadata']['commentaries_data_commit'],
                                                get_commit_hash(vs.COMMS_DATA_DIR)),
                check_change_in_ne(comm_id, existing_can_json['metadata']['ne_corpus_commit'],
                                   get_commit_hash(vs.NE_CORPUS_DIR)),
                check_change_in_lemlink(comm_id, existing_can_json['metadata']['lemlink_corpus_commit'],
                                        get_commit_hash(vs.LEMLINK_CORPUS_DIR)),
                args.force]):
                continue

        comm = RawCommentary(id=comm_id, ocr_run_id=args.ocr_run_pattern).to_canonical()

        if comm.is_safe():
            comm.to_json()

        else:
            print('WARNING: The created canonical commentary contains duplicates, skipping...')
            unsafe_canonicals.append(comm_id)

    if unsafe_canonicals:
        print('WARNING: THE FOLLOWING COMMENTARIES WERE SKIPPED DUE TO DUPLICATES:')
        print(unsafe_canonicals)

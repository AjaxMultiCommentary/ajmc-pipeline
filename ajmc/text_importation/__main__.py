import argparse
import json
import os
import sys
from ajmc.text_importation.classes import Commentary
from ajmc.text_importation.rebuild import basic_rebuild, rebuilt_to_xmi
from ajmc.commons.variables import PATHS
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)

def create_default_pipeline_parser() -> argparse.ArgumentParser:
    """Adds `__main__.py` relative arguments to the parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--commentary_ids',
                        nargs='+',
                        type=list,
                        required=False,
                        help='The ids of commentary to be processed.')

    parser.add_argument('--commentary_formats',
                        nargs='+',
                        type=list,
                        required=False,
                        help='The respective ocr format to process commentary in')

    parser.add_argument('--json_dir',
                        type=str,
                        required=False,
                        help='Absolute path to the directory in which to write the json files')

    parser.add_argument('--make_jsons',
                        action='store_true',
                        help='Whether to create canonical jsons')

    parser.add_argument('--xmi_dir',
                        type=str,
                        required=False,
                        help='Absolute path to the directory in which to write the xmi files')

    parser.add_argument('--make_xmis',
                        action='store_true',
                        help='Whether to create xmis')

    parser.add_argument('--region_types',
                        nargs='+',
                        type=list,
                        help="""The desired regions to convert to xmis, 
                        eg `introduction, preface, commentary, footnote`""")
    return parser


def initialize_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parses args from command-line if available, else locally."""

    # If `__main__.py` is called from cli
    if sys.argv[0].endswith('__main__.py') and len(sys.argv) > 1:
        return parser.parse_args()

    else:  # For testing, without cli
        args = parser.parse_args([])
        # args.commentary_ids = ['sophokle1v3soph', 'cu31924087948174', 'Wecklein1894']
        # args.commentary_formats = ['pagexml'] * 3
        args.commentary_ids = ['lestragdiesdeso00tourgoog']
        args.commentary_formats = ['tesshocr']
        args.region_types = ['introduction', 'preface', 'commentary', 'footnote']
        args.make_xmis = True
        args.xmi_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/lestragdiesdeso00tourgoog/ner/annotation/tesshocr'
        args.json_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/lestragdiesdeso00tourgoog/canonical/tesshocr'
        args.make_jsons = True

        return args


def main():

    args = initialize_args(create_default_pipeline_parser())

    for commentary_id, commentary_format in zip(args.commentary_ids, args.commentary_formats):

        commentary = Commentary(commentary_id)

        args.json_dir = os.path.join(PATHS['base_dir'], commentary_id, 'canonical', commentary_format)
        os.makedirs(args.json_dir, exist_ok=True)

        args.xmi_dir = os.path.join(PATHS['base_dir'], commentary_id, 'ner/annotation', commentary_format)
        os.makedirs(args.xmi_dir, exist_ok=True)

        if args.make_jsons and args.make_xmis:
            for page in commentary.pages:
                logger.info('Processing page  ' + page.id)
                page.to_json(output_dir=args.json_dir)
                rebuild = basic_rebuild(page.canonical_data, args.region_types)
                if len(rebuild['fulltext']) > 0:  # h andles the empty-page case
                    rebuilt_to_xmi(rebuild, args.xmi_dir)

        elif args.make_jsons:
            for page in commentary.pages:
                logger.info('Canonizing page  ' + page.id)
                page.to_json(output_dir=args.json_dir)


        elif args.make_xmis:

            for filename in os.listdir(args.json_dir):
                logger.info('Xmi-ing page  ' + page.id)
                if filename.endswith('.json'):
                    with open(os.path.join(args.json_dir, filename), 'r') as f:
                        page = json.loads(f.read())  # Why can't this be done directly from commentary ?

                    rebuild = basic_rebuild(page, args.region_types)
                    if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                        rebuilt_to_xmi(rebuild, args.xmi_dir, typesystem_path=PATHS['typesystem'])


if __name__ == '__main__':
    main()




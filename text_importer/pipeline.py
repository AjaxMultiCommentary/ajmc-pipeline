import argparse
import json
import os
import sys
from text_importer.pagexml import PagexmlCommentary
from text_importer.rebuild import basic_rebuild, rebuilt_to_xmi
from text_importer.tesshocr import TessHocrCommentary
from text_importer.krakenhocr import KrakenHocrCommentary
from commons.variables import PATHS


def create_pagexml_pipeline_parser() -> argparse.ArgumentParser:
    """Adds `pipeline.py` relative arguments to the parser"""

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
                        help='The respective format to process commentary in')

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

    # If `pipeline.py` is called from cli
    if sys.argv[0].endswith('pipeline.py') and len(sys.argv) > 1:
        return parser.parse_args()

    else:  # For testing, without cli
        args = parser.parse_args([])
        args.commentary_ids = ['sophokle1v3soph', 'cu31924087948174', 'Wecklein1894']
        args.commentary_formats = ['pagexml'] * 3
        args.region_types = ['introduction', 'preface', 'commentary', 'footnote']
        args.make_xmis = True
        args.make_jsons = True
        args.region_types = ['introduction', 'commentary', 'footnotes', 'preface']

        return args


def main():
    commentary_classes = {'pagexml': PagexmlCommentary,
                          'krakenhocr': KrakenHocrCommentary,
                          'tesshocr': TessHocrCommentary}

    args = initialize_args(create_pagexml_pipeline_parser())

    for commentary_id, commentary_format in zip(args.commentary_ids, args.commentary_formats):

        commentary = commentary_classes[commentary_format](commentary_id)


        args.json_dir = os.path.join(PATHS['base_dir'], commentary_id, 'canonical', commentary_format)
        os.makedirs(args.json_dir, exist_ok=True)

        args.xmi_dir = os.path.join(PATHS['base_dir'], commentary_id, 'ner/annotation', commentary_format)
        os.makedirs(args.xmi_dir, exist_ok=True)

        if args.make_jsons and args.make_xmis:
            for page in commentary.pages:
                print('Processing page  ' + page.id)
                page.to_json(output_dir=args.json_dir)
                rebuild = basic_rebuild(page.canonical_data, args.region_types)
                if len(rebuild['fulltext']) > 0:  # h andles the empty-page case
                    rebuilt_to_xmi(rebuild, args.xmi_dir)

        elif args.make_jsons:
            for page in commentary.pages:
                print('Canonizing page  ' + page.id)
                page.to_json(output_dir=args.json_dir)


        elif args.make_xmis:

            for filename in os.listdir(args.json_dir):
                print('Xmi-ing page  ' + page.id)
                if filename.endswith('.json'):
                    with open(os.path.join(args.json_dir, filename), 'r') as f:
                        page = json.loads(f.read())

                    rebuild = basic_rebuild(page, args.region_types)
                    if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                        rebuilt_to_xmi(rebuild, args.xmi_dir, typesystem_path=PATHS['typesystem'])



if __name__ == '__main__':
    main()

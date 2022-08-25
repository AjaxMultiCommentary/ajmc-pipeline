import argparse
import json
import os
import sys
from typing import Union, List, Dict
from cassis import load_typesystem, Cas
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.commons.variables import PATHS
from ajmc.text_processing.ocr_classes import OcrCommentary

logger = get_custom_logger(__name__)


def basic_rebuild(page: dict,
                  region_types: Union[List[str], str] = "all",
                  string: str = "") -> dict:
    # todo 👁️ a light version of this function computing only what you actually need
    """Basic rebuild function"""

    coordinates = {"regions": [], "lines": [], "words": []}
    offsets = {"regions": [], "lines": [], "words": []}
    texts = {"regions": [], "lines": [], "words": []}

    for region in page["regions"]:

        if region_types == "all" or region["region_type"] in region_types:

            region_text = ""
            region_offsets = [len(string)]

            for line in region["lines"]:
                line_text = ""
                line_offsets = [len(string)]

                for n, word in enumerate(line['words']):
                    word_offsets = [len(string)]

                    region_text += word["text"] + " "
                    line_text += word["text"] + " "
                    string += word["text"] + " "

                    word_offsets.append(len(string) - 1)

                    texts["words"].append(word["text"] + " ")
                    offsets['words'].append(word_offsets)
                    coordinates['words'].append(word['bbox'])

                line_offsets.append(len(string) - 1)
                offsets["lines"].append(line_offsets)
                coordinates['lines'].append(line['bbox'])
                texts["lines"].append(line_text)

            region_offsets.append(len(string) - 1)
            coordinates['regions'].append(region['bbox'])
            offsets["regions"].append(region_offsets)
            texts["regions"].append(region_text)

    return {"id": page["id"], "fulltext": string, "bbox": coordinates, "offsets": offsets, "texts": texts}


def get_iiif_url(page_id: str,
                 box: List[int],
                 base: str = "http://lorem_ipsum.com/ajax",
                 iiif_manifest_uri: str = None,
                 pct: bool = False,
                 ) -> str:
    """ Returns impresso iiif url given a page id and a box

    :param page_id: impresso page id, e.g. EXP-1930-06-10-a-p0001
    :param box: iiif box (x, y, w, h)
    :return: iiif url of the box
    """
    prefix = "pct:" if pct else ""
    suffix = "full/0/default.jpg"

    box = ",".join(str(x) for x in box)

    if iiif_manifest_uri is None:
        return os.path.join(base, page_id, prefix + box, suffix)
    else:
        return os.path.join(iiif_manifest_uri.replace('/info.json', ''), prefix + box, suffix)


def compute_image_links(page: dict,
                        padding: int = 20,
                        iiif_endpoint: str = None,
                        iiif_links: Dict[str, str] = None,
                        pct: bool = False,):

    image_links = []

    for line_coords, line_offsets in zip(page["bbox"]["lines"], page["offsets"]["lines"]):

        if iiif_links is None:
            iiif_link = get_iiif_url(page["id"], box=line_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page["id"], box=line_coords, iiif_manifest_uri=iiif_links[page["id"]], pct=pct)
        image_links.append((iiif_link, line_offsets[0], line_offsets[1]))

    for word_coords, word_offsets in zip(page["bbox"]["words"], page["offsets"]["words"]):

        if iiif_links is None:
            iiif_link = get_iiif_url(page["id"], box=word_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page["id"], box=word_coords, iiif_manifest_uri=iiif_links[page["id"]], pct=pct)
        image_links.append((iiif_link, word_offsets[0], word_offsets[1]))

    return image_links


def rebuilt_to_xmi(page: dict,  # todo 👁️ should this accept CommentaryPage objects ?
                   output_dir: str,
                   typesystem_path: str = PATHS['typesystem'],
                   iiif_mappings=None,
                   pct_coordinates=False):
    """
    Converts a rebuilt page into Apache UIMA/XMI format.

    The resulting file will be named after the page ID, adding
    the `.xmi` extension.

    :param page: the page to be converted
    :param output_dir: the path to the output directory
    :param typesystem_path: TypeSystem file containing defitions of annotation layers.
    """

    with open(typesystem_path, "rb") as f:
        typesystem = load_typesystem(f)  # object for the type system

    cas = Cas(typesystem=typesystem)
    cas.sofa_string = page["fulltext"]  # str # `ft` field in the rebuild CI
    cas.sofa_mime = 'text/plain'

    sentence = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
    word = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token')

    img_link_type = 'webanno.custom.AjMCImages'
    image_link = typesystem.get_type(img_link_type)

    # create sentence-level annotations
    for offsets in page["offsets"]["lines"]:
        cas.add_annotation(sentence(begin=offsets[0], end=offsets[1]))

    for offsets in page["offsets"]["words"]:
        cas.add_annotation(word(begin=offsets[0], end=offsets[1]))

    iiif_links = compute_image_links(page, iiif_links=iiif_mappings, pct=pct_coordinates)

    # inject the IIIF links into
    for iiif_link, start, end in iiif_links:
        cas.add_annotation(image_link(begin=start, end=end, link=iiif_link))

    outfile_path = os.path.join(output_dir, f'{page["id"]}.xmi')
    cas.to_xmi(outfile_path, pretty_print=True)


def create_default_pipeline_parser() -> argparse.ArgumentParser:
    """Adds relative arguments to the parser"""

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

    # If called from cli
    if sys.argv[0].endswith('cas_export.py') and len(sys.argv) > 1:
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

        commentary = OcrCommentary(commentary_id)

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

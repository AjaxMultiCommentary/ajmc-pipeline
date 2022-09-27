"""
This is a legacy but functionnal code.
"""
import glob
import json
import os
from typing import Union, List, Dict, Optional
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
                        pct: bool = False, ):
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


# todo 👁️ This should rely on ocr outputs dirs
def main(commentaries: List[Dict[str, str]],
         make_jsons: bool,
         make_xmis: bool,
         json_dir: Optional[str] = None,
         xmi_dir: Optional[str] = None,
         region_types: Union[List[str], str] = 'all'):
    """
    Main function for the pipeline.
    
    Args:
        commentaries: A list of dicts `{'commentary_id': 'ocr_run'}` linking to the commentaries to be processed.
        json_dir: Absolute path to the directory in which to write the json files or take them from.
        xmi_dir: Absolute path to the directory in which to write the xmi files.
        make_jsons: Whether to create canonical jsons. If false, jsons are grepped from json_dir.
        make_xmis: Whether to create xmis.
        region_types: The desired regions to convert to xmis, eg `introduction, preface, commentary, footnote`.   
    """

    for commentary_id, ocr_run in commentaries.items():

        # Create paths
        ocr_dir = os.path.join(PATHS['base_dir'], commentary_id, PATHS['ocr'], ocr_run, 'outputs')
        json_dir = json_dir if json_dir else os.path.join(PATHS['base_dir'], commentary_id, 'canonical', ocr_run)
        xmi_dir = xmi_dir if xmi_dir else os.path.join(PATHS['base_dir'], commentary_id, 'ner/annotation', ocr_run)

        # Get the commentary
        commentary = OcrCommentary.from_ajmc_structure(ocr_dir=ocr_dir)

        if make_jsons and make_xmis:
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(xmi_dir, exist_ok=True)

            for page in commentary.children.pages:
                logger.info('Processing page  ' + page.id)
                page.to_json(output_dir=json_dir)
                rebuild = basic_rebuild(page.to_canonical_v1(), region_types)
                if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                    rebuilt_to_xmi(rebuild, xmi_dir)

        elif make_jsons:
            os.makedirs(json_dir, exist_ok=True)
            for page in commentary.children.pages:
                logger.info('Canonizing page  ' + page.id)
                page.to_json(output_dir=json_dir)

        elif make_xmis:
            os.makedirs(xmi_dir, exist_ok=True)

            for filename in glob.glob(os.path.join(json_dir, '*.json')):
                with open(os.path.join(json_dir, filename), 'r') as f:
                    logger.info('Xmi-ing page  ' + page['id'])
                    page = json.loads(f.read())  # Why can't this be done directly from commentary ?

                rebuild = basic_rebuild(page, region_types)
                if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                    rebuilt_to_xmi(rebuild, xmi_dir, typesystem_path=PATHS['typesystem'])




main({'sophoclesplaysa05campgoog': '248095_greek-english_porson_sophoclesplaysa05campgoog'},
     make_jsons=True,
     make_xmis=True,
     region_types='all')
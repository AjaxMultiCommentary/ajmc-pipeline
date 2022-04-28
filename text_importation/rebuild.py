# %%
import os
from typing import Union, List, Dict
from cassis import load_cas_from_xmi, load_typesystem, Cas

from common_utils.variables import PATHS


def basic_rebuild(page: dict,
                  region_types: Union[List[str], str] = "all",
                  string: str = "") -> dict:
    # todo a light version of this function computing only what you actually need
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
                    coordinates['words'].append(word['coords'])

                line_offsets.append(len(string) - 1)
                offsets["lines"].append(line_offsets)
                coordinates['lines'].append(line['coords'])
                texts["lines"].append(line_text)

            region_offsets.append(len(string) - 1)
            coordinates['regions'].append(region['coords'])
            offsets["regions"].append(region_offsets)
            texts["regions"].append(region_text)

    return {"id": page["id"], "fulltext": string, "coords": coordinates, "offsets": offsets, "texts": texts}


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
                        pct: bool = False,
                        ):
    """From : `impresso_commons/utils/uima.py`

    :param type page: Description of parameter `page`.
    :param type padding: Description of parameter `padding`.
    :return: Description of returned object.
    :rtype: type

    """

    image_links = []

    for line_coords, line_offsets in zip(page["coords"]["lines"], page["offsets"]["lines"]):

        if iiif_links is None:
            iiif_link = get_iiif_url(page["id"], box=line_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page["id"], box=line_coords, iiif_manifest_uri=iiif_links[page["id"]], pct=pct)
        image_links.append((iiif_link, line_offsets[0], line_offsets[1]))

    for word_coords, word_offsets in zip(page["coords"]["words"], page["offsets"]["words"]):

        if iiif_links is None:
            iiif_link = get_iiif_url(page["id"], box=word_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page["id"], box=word_coords, iiif_manifest_uri=iiif_links[page["id"]], pct=pct)
        image_links.append((iiif_link, word_offsets[0], word_offsets[1]))

    return image_links


def rebuilt_to_xmi(page: dict,  # todo : should this accept CommentaryPage objects ?
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

# %%
# import json
#
# with open("/Users/sven/ajmc/data/json/bsb10234118_0019.json", "r") as f:
#     page = json.loads(f.read())
#
# page = basic_rebuild(page)
#
# if len(page["fulltext"]) > 0:  # handles the empty-page case
#     rebuilt_to_xmi(page, output_dir="/Users/sven/ajmc/data/json/",
#                    typesystem_path="/Users/sven/ajmc/data/xmi_annotation/TypeSystem.xml")

# %%


# create a dict similar to `article` (List[dict]) (see contentitem schema)
# For each page, you apply the basic rebuild we get the fulltext, and information
# about coordinates for regions and words and text offsets for regions, lines, words.
# we do not concat the full tewxt for multiple page. Our CI (corresponding to an inception document) is a page
# so you have to keep the

# we just use a mock link for iiif (this is to be implemented later on)
# # For image links
#     sentType = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' ## This can be cpy
#     imgLinkType = 'webanno.custom.ImpressoImages'   # wil be given mby matteo
#     Sentence = typesystem.get_type(sentType)  # use the same
#     ImageLink = typesystem.get_type(imgLinkType)  # same

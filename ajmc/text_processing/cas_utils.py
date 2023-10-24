"""
``basic_rebuild``, ``get_iiif_url``, ``compute_image_links``, ``get_cas``, ``rebuild_to_xmi``, ``export_commentaries_to_xmi`` are
legacy but functional.
"""
import json
import os
from pathlib import Path
from time import strftime
from typing import Dict, List, Type, Optional

from cassis import Cas, load_cas_from_xmi, load_typesystem
from cassis.typesystem import FeatureStructure
from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.miscellaneous import aligned_print, get_ajmc_logger

logger = get_ajmc_logger(__name__)


def basic_rebuild(page: dict,
                  region_types: List[str],
                  string: str = '') -> dict:
    # todo 👁️ a light version of this function computing only what you actually need
    """Basic rebuild function"""

    coordinates = {'regions': [], 'lines': [], 'words': []}
    offsets = {'regions': [], 'lines': [], 'words': []}
    texts = {'regions': [], 'lines': [], 'words': []}

    for region in page['regions']:

        if region['region_type'] in region_types:

            region_text = ''
            region_offsets = [len(string)]

            for line in region['lines']:
                line_text = ''
                line_offsets = [len(string)]

                for n, word in enumerate(line['words']):
                    word_offsets = [len(string)]

                    region_text += word['text'] + ' '
                    line_text += word['text'] + ' '
                    string += word['text'] + ' '

                    word_offsets.append(len(string) - 1)

                    texts['words'].append(word['text'] + ' ')
                    offsets['words'].append(word_offsets)
                    try:
                        coordinates['words'].append(word['bbox'])
                    except KeyError:
                        coordinates['words'].append(word['coords'])  # for old rebuilds

                line_offsets.append(len(string) - 1)
                offsets['lines'].append(line_offsets)
                try:
                    coordinates['lines'].append(line['bbox'])
                except KeyError:
                    coordinates['lines'].append(line['coords'])  # for old rebuilds
                texts['lines'].append(line_text)

            region_offsets.append(len(string) - 1)
            try:
                coordinates['regions'].append(region['bbox'])
            except KeyError:
                coordinates['regions'].append(region['coords'])  # for old rebuilds
            offsets['regions'].append(region_offsets)
            texts['regions'].append(region_text)

    return {'id': page['id'], 'fulltext': string, 'bbox': coordinates, 'offsets': offsets, 'texts': texts}


def get_iiif_url(page_id: str,
                 box: List[int],
                 base: str = 'http://lorem_ipsum.com/ajax',
                 iiif_manifest_uri: str = None,
                 pct: bool = False,
                 ) -> str:
    """ Returns impresso iiif url given a page id and a box

    Args:
        page_id (str): page id
        box (List[int]): iiif box coordinates (x, y, w, h)
        base (str): base url
        iiif_manifest_uri (str): iiif manifest uri
        pct (bool): if True, returns pct coordinates
    """
    prefix = 'pct:' if pct else ''
    suffix = 'full/0/default.jpg'

    box = ','.join(str(x) for x in box)

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

    for line_coords, line_offsets in zip(page['bbox']['lines'], page['offsets']['lines']):

        if iiif_links is None:
            iiif_link = get_iiif_url(page['id'], box=line_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page['id'], box=line_coords, iiif_manifest_uri=iiif_links[page['id']], pct=pct)
        image_links.append((iiif_link, line_offsets[0], line_offsets[1]))

    for word_coords, word_offsets in zip(page['bbox']['words'], page['offsets']['words']):

        if iiif_links is None:
            iiif_link = get_iiif_url(page['id'], box=word_coords, pct=pct)
        else:
            iiif_link = get_iiif_url(page['id'], box=word_coords, iiif_manifest_uri=iiif_links[page['id']], pct=pct)
        image_links.append((iiif_link, word_offsets[0], word_offsets[1]))

    return image_links


def rebuild_to_xmi(page: dict,
                   output_dir: Path,
                   ocr_run_id: str,
                   region_types: List[str],
                   typesystem_path: Path = vs.TYPESYSTEM_PATH,
                   iiif_mappings=None,
                   pct_coordinates=False):
    """
    Converts a rebuild-dict into Apache UIMA/XMI format.

    The resulting file will be named after the page ID, adding
    the ``.xmi`` extension.

    Args:
        page (dict): a page dict as returned by ``basic_rebuild``
        output_dir (Path): path to the output directory
        ocr_run_id (str): OCR run ID
        region_types: regions considered
        typesystem_path (Path): path to the typesystem file
        iiif_mappings (dict): a dict mapping page IDs to IIIF manifest URIs
        pct_coordinates (bool): if True, coordinates are expressed in percentage
    """

    with open(str(typesystem_path), 'rb') as f:
        typesystem = load_typesystem(f)  # object for the type system

    cas = Cas(typesystem=typesystem)
    cas.sofa_string = page['fulltext']  # str # ``ft`` field in the rebuild CI
    cas.sofa_mime = 'text/plain'

    sentence = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
    word = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token')

    img_link_type = 'webanno.custom.AjMCImages'
    ajmc_metadata_type = 'webanno.custom.AjMCDocumentmetadata'
    image_link = typesystem.get_type(img_link_type)
    ajmc_metadata = typesystem.get_type(ajmc_metadata_type)

    # create metadata annotations
    metadata = ajmc_metadata(ocr_run_id=ocr_run_id,
                             region_types=','.join(region_types),
                             xmi_creation_date=strftime('%Y-%m-%d %H:%M:%S'))
    cas.add(metadata)

    # create sentence-level annotations
    for offsets in page['offsets']['lines']:
        cas.add(sentence(begin=offsets[0], end=offsets[1]))

    for offsets in page['offsets']['words']:
        cas.add(word(begin=offsets[0], end=offsets[1]))

    iiif_links = compute_image_links(page, iiif_links=iiif_mappings, pct=pct_coordinates)

    # inject the IIIF links into
    for iiif_link, start, end in iiif_links:
        cas.add(image_link(begin=start, end=end, link=iiif_link))

    cas.to_xmi((output_dir / f'{page["id"]}.xmi'), pretty_print=True)


def export_commentary_to_xmis(commentary: Type['OcrCommentary'],
                              make_jsons: bool,
                              make_xmis: bool,
                              jsons_dir: Path,
                              xmis_dir: Path,
                              region_types: List[str],
                              overwrite: bool = False):
    """
    Main function for the pipeline.
    
    Args:
        commentary: The commentary to convert to xmis, should be an OcrCommentary object (not a canonical commentary).
        jsons_dir: Absolute path to the directory in which to write the json files or take them from.
        xmis_dir: Absolute path to the directory in which to write the xmi files.
        make_jsons: Whether to create canonical jsons. If false, jsons are grepped from json_dir.
        make_xmis: Whether to create xmis.
        region_types: The desired regions to convert to xmis, eg ``introduction, preface, commentary, footnote``.
        overwrite: Whether to overwrite existing files.
    """

    if make_jsons and make_xmis:
        jsons_dir.mkdir(parents=True, exist_ok=overwrite)
        xmis_dir.mkdir(parents=True, exist_ok=overwrite)

        for page in tqdm(commentary.children.pages, desc=f'Building jsons and xmis for {commentary.id}'):
            page.to_inception_json(output_dir=jsons_dir)
            rebuild = basic_rebuild(page.to_inception_dict(), region_types)
            if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                rebuild_to_xmi(rebuild, xmis_dir, commentary.ocr_run_id, region_types)

    elif make_jsons:
        jsons_dir.mkdir(parents=True, exist_ok=overwrite)
        for page in tqdm(commentary.children.pages, desc=f'Creating jsons for {commentary.id}'):
            logger.debug('Canonizing page  ' + page.id)
            page.to_inception_json(output_dir=jsons_dir)

    elif make_xmis:
        xmis_dir.mkdir(parents=True, exist_ok=overwrite)

        for json_path in tqdm(jsons_dir.glob('*.json'), desc=f'Building xmis for {commentary.id}'):
            page = json.loads(json_path.read_text(encoding='utf-8'))
            logger.debug('Xmi-ing page  ' + page['id'])
            rebuild = basic_rebuild(page, region_types)
            if len(rebuild['fulltext']) > 0:  # handles the empty-page case
                rebuild_to_xmi(rebuild, xmis_dir, commentary.ocr_run_id, region_types)


def get_cas(xmi_path: Path, xml_path: Path) -> Cas:
    # ⚠️ for some reason, passing to ``load_cas_from_xmi()`` the file object
    # works just fine, while passing to it the path (``str``) raises an
    # exception of empty XMI files (very strange!) 
    with open(xmi_path, 'rb') as inputfile:
        cas = load_cas_from_xmi(inputfile, typesystem=load_typesystem(xml_path))
    return cas


def align_cas_annotation(cas_annotation, rebuild, verbose: bool = False):
    # We define an empty dict to store the annotation.
    bboxes, shifts, warnings = [], [], []

    # We get the annotation's surrounding text
    text_window = f"""{cas_annotation.sofa.sofaString[cas_annotation.begin - 10:cas_annotation.begin]}|\
    {cas_annotation.sofa.sofaString[cas_annotation.begin:cas_annotation.end]}|\
    {cas_annotation.sofa.sofaString[cas_annotation.end:cas_annotation.end + 10]}"""

    # We then find the words included in the annotation and retrieve their bboxes
    ann_words = [{'bbox': bbox, 'offsets': offsets}
                 for bbox, offsets in zip(rebuild['bbox']['words'], rebuild['offsets']['words'])
                 if compute_interval_overlap((cas_annotation.begin, cas_annotation.end), offsets) > 0]

    # If the annotation words are not found in the page dictionary. This is a problem, should not happen
    if not ann_words:
        logger.error(f"""Annotation has no words: {text_window}""")
        warnings = ['no words']

    else:
        bboxes = [ann_word['bbox'] for ann_word in ann_words]
        shifts = [cas_annotation.begin - ann_words[0]['offsets'][0],
                  cas_annotation.end - ann_words[-1]['offsets'][1]]

        if verbose:
            aligned_print(rebuild['fulltext'][ann_words[0]['offsets'][0]:ann_words[-1]['offsets'][1]],
                          cas_annotation.sofa.sofaString[cas_annotation.begin:cas_annotation.end])

    return bboxes, shifts, text_window, warnings


def import_page_rebuild(page_id: str, annotation_type: str = 'ner'):
    """Finds and rebuild the inception json of the fgiven ``page_id``.

    Args:
        page_id: The id of the page to rebuild.
        annotation_type: The type of annotation to rebuild, either ``ner`` or ``lemlink``.
    """
    comm_id = page_id.split('_')[0]

    if annotation_type in ['entities', 'sentences', 'hyphenations']:

        rebuild_path = vs.get_comm_ner_jsons_dir(comm_id, vs.IDS_TO_NER_RUNS[comm_id]) / (page_id + '.json')
        if comm_id == 'sophoclesplaysa05campgoog' and page_id in vs.MINIREF_PAGES:
            rebuild_path = vs.get_comm_ner_jsons_dir(comm_id, '1bm0b4_tess_final') / (page_id + '.json')
        return basic_rebuild(page=json.loads(rebuild_path.read_text('utf-8')),
                             region_types=vs.IDS_TO_REGIONS[comm_id])

    elif annotation_type == 'lemmas':
        run_dir = [dir_ for dir_ in (vs.get_comm_base_dir(comm_id) / vs.COMM_LEMLINK_ANN_REL_DIR).glob('*') if dir_.is_dir()][0]
        rebuild_path = run_dir / 'jsons' / (page_id + '.json')
        metadata = json.loads((run_dir / 'xmis' / 'metadata.json').read_text('utf-8'))

        return basic_rebuild(page=json.loads(rebuild_path.read_text('utf-8')),
                             region_types=metadata['region_types'])


def import_page_cas(page_id: str,
                    annotation_type: str) -> Optional[Cas]:
    """Finds and rebuild the inception ``.xmi`` of the fgiven ``page_id``, returning ``None`` if not found."""

    if annotation_type in ['entities', 'sentences', 'hyphenations']:
        xml_path = vs.NE_CORPUS_DIR / 'data/preparation/TypeSystem.xml'
        candidate_xmi_paths = list(vs.NE_CORPUS_DIR.glob(f'data/preparation/corpus/*/curated/{page_id}.xmi'))
        if candidate_xmi_paths:
            xmi_path = candidate_xmi_paths[0]
            return get_cas(xmi_path, xml_path)
        else:
            return

    elif annotation_type == 'lemmas':
        xml_path = vs.LEMLINK_XMI_DIR / 'TypeSystem.xml'
        xmi_path = vs.LEMLINK_XMI_DIR / (f'{page_id}.xmi')

        return get_cas(xmi_path, xml_path)


def safe_import_page_annotations(page_id,
                                 cas,
                                 rebuild,
                                 annotation_layer: str,
                                 manual_safe_check: bool = False) -> List[FeatureStructure]:
    if manual_safe_check and cas.sofa_string != rebuild['fulltext']:
        print(f'Alignment error, skipping: {page_id}')
        print('REBUILD**************************')
        print(rebuild['fulltext'])
        print('CAS**************************')
        print(cas.sofa_string)
        input('Press enter to continue')

    return cas.select(annotation_layer)

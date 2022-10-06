import glob
import os
import json
from cassis import Cas, load_cas_from_xmi, load_typesystem
from typing import List, Dict, Any

from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.variables import PATHS, MINIREF_PAGES
from ajmc.text_processing.cas_export import basic_rebuild
from pathlib import Path
from ajmc.commons.miscellaneous import aligned_print
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)
logger.handlers[0].setLevel('ERROR')

IDS_TO_RUNS = {
    'cu31924087948174': '1bm0b3_tess_final',
    'lestragdiesdeso00tourgoog': '21i0dA_tess_hocr',
    'sophokle1v3soph': '1bm0b5_tess_final',
    'sophoclesplaysa05campgoog': '15o0dN_lace_retrained_sophoclesplaysa05campgoog-2021-05-24-08-15-12-porson-with-sophoclesplaysa05campgoog-2021-05-23-22-17-38',
    'Wecklein1894': '1bm0b6_tess_final'
}

IDS_TO_REGIONS = {
    'cu31924087948174': ['commentary', 'introduction', 'preface'],
    'Wecklein1894': ['commentary', 'introduction', 'preface'],
    'sophokle1v3soph': ['commentary', 'introduction', 'preface'],
    'sophoclesplaysa05campgoog': ['commentary', 'introduction', 'preface', 'footnote'],
    'lestragdiesdeso00tourgoog': ['commentary', 'introduction', 'preface', 'footnote']
}

NE_TYPE = 'webanno.custom.AjMCNamedEntity'
SEGMENT_TYPE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
SENTENCE_TYPE = 'webanno.custom.GoldSentences'
HYPHENATION_TYPE = 'webanno.custom.GoldHyphenation'
TOKEN_TYPE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'

xml_path = Path('/Users/sven/packages/AjMC-NE-corpus/data/preparation/TypeSystem.xml')


def get_cas(xmi_path: 'pathlib.Path', xml_path: 'pathlib.Path') -> 'cassis.Cas':
    typesystem = load_typesystem(xml_path)
    cas = load_cas_from_xmi(xmi_path, typesystem=typesystem)
    return cas


def retrieve_entities_bboxes(cas,
                             rebuild,
                             layer,
                             print_text: bool = False) -> List[Dict[str, Any]]:
    """Retrieves entities values and bounding boxes from a CAS and a page dictionary."""

    entities = []

    for ent in cas.select(layer):
        # We start by retrieving the text around the entity
        entity_text_window = f"""{cas.sofa_string[ent.begin - 10:ent.begin]}|{cas.sofa_string[ent.begin:ent.end]}|{cas.sofa_string[ent.end:ent.end + 10]}"""

        # We create the entity dict
        entity = {'value': ent.value,
                  'text_window': entity_text_window,
                  'warning': ''}

        # We then find the words included in the entity and retrieve their bboxes
        ent_words = [{'bbox': bbox, 'offsets': offsets}
                     for bbox, offsets in zip(rebuild['bbox']['words'], rebuild['offsets']['words'])
                     if compute_interval_overlap((ent.begin, ent.end), offsets) > 0]

        # CASE 1: The entity words are not found in the page dictionary. This is a problem, should not happen
        if not ent_words:
            logger.error(f'Entity has no words: {entity_text_window}')
            entity['bbox'] = None
            entity['shift_left'] = None
            entity['shift_right'] = None
            entity['warning'] += 'no words - '

        else:
            entity['bboxes'] = [ent_word['bbox'] for ent_word in ent_words]
            entity['shift_left'] = [ent_words[0]['offsets'][0] - ent.begin]
            entity['shift_right'] = [ent_words[-1]['offsets'][1] - ent.end]

            # CASE 2A: The entity words are found in the page dictionary, but entity and words begin are not aligned
            if ent_words[0]['offsets'][0] != ent.begin:  # if the entity begin is shifted but less critical
                logger.warning(f"""Entity begin not aligned: {entity_text_window}""")
                entity['warning'] += 'begin not aligned - '

            # CASE 2B: The entity words are found in the page dictionary, but entity and words end are not aligned
            if ent_words[-1]['offsets'][1] != ent.end:  # if the entity end is shifted
                logger.warning(f'Entity end not aligned:  {entity_text_window}')
                entity['warning'] += 'end not aligned - '

            if print_text:
                aligned_print(rebuild["fulltext"][ent_words[0]['offsets'][0]:ent_words[-1]['offsets'][1]],
                              cas.sofa_string[ent.begin:ent.end],
                              ent.value)

        entities.append(entity)

    return entities


entities = {id: {} for id in IDS_TO_RUNS.keys()}

for xmi_path in sorted(list(glob.glob('/Users/sven/packages/AjMC-NE-corpus/data/preparation/corpus/*/curated/*.xmi'))):
    xmi_path = Path(xmi_path)
    comm_id = xmi_path.stem.split('_')[0]
    json_path = Path(os.path.join(PATHS['base_dir'], comm_id, 'canonical',
                                  IDS_TO_RUNS[comm_id], xmi_path.stem + '.json'))

    # Special case of Jebb
    if comm_id == 'sophoclesplaysa05campgoog' and xmi_path.stem in MINIREF_PAGES:
        json_path = Path(os.path.join(PATHS['base_dir'], comm_id, 'canonical',
                                      '1bm0b4_tess_final', xmi_path.stem + '.json'))

    cas = get_cas(xmi_path, xml_path)

    with open(json_path, "r") as file:
        page_dict = json.loads(file.read())

    rebuild = basic_rebuild(page=page_dict, region_types=IDS_TO_REGIONS[comm_id])

    if not cas.sofa_string == rebuild['fulltext']:
        logger.error(f'Alignment error, skipping: {xmi_path.stem}')
        continue

    entities[comm_id][xmi_path.stem] = retrieve_entities_bboxes(cas, rebuild, NE_TYPE, print_text=False)


# we now export our entities to a json file
for comm_id in entities.keys():
    os.makedirs(os.path.join(PATHS['base_dir'], comm_id, 'ner', 'entities'), exist_ok=True)
    with open(os.path.join(PATHS['base_dir'], comm_id, 'ner', 'entities', 'entities.json'), 'w') as file:
        json.dump(entities[comm_id], file, indent=4)



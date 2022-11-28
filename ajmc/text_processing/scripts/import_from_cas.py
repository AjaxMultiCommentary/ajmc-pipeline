from typing import List, Dict, Any

from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.variables import PATHS, ANNOTATION_LAYERS
from ajmc.text_processing.cas_utils import import_page_annotations_from_xmi
from ajmc.commons.miscellaneous import aligned_print
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)
logger.handlers[0].setLevel('ERROR')

# VARIABLES

# PARAMETERS


def retrieve_annotations_bboxes(cas,
                                rebuild,
                                annotation_layer,
                                print_text: bool = False) -> List[Dict[str, Any]]:
    """Retrieves annotations values and bounding boxes from a CAS and a page dictionary."""

    annotations = []

    for ann in cas.select(annotation_layer):
        # We start by retrieving the text around the annotation
        annotation_text_window = f"""{cas.sofa_string[ann.begin - 10:ann.begin]}|{cas.sofa_string[ann.begin:ann.end]}|{cas.sofa_string[ann.end:ann.end + 10]}"""

        # We create the annotation dict
        annotation = {'bboxes': [],
                      'shifts': [],
                      'is_nil': ann.is_NIL,
                      'is_noisy_ocr': ann.noisy_ocr,
                      'transcript': ann.transcript,
                      'layer_type': ann.type.name,
                      'annotation_type': ann.value,
                      'wikidata_id': ann.wikidata_id,
                      'text_window': annotation_text_window,
                      'warnings': []}

        # We then find the words included in the annotation and retrieve their bboxes
        ann_words = [{'bbox': bbox, 'offsets': offsets}
                     for bbox, offsets in zip(rebuild['bbox']['words'], rebuild['offsets']['words'])
                     if compute_interval_overlap((ann.begin, ann.end), offsets) > 0]

        # If the annotation words are not found in the page dictionary. This is a problem, should not happen
        if not ann_words:
            logger.error(f'Annotation has no words: {annotation_text_window}')
            annotation['warnings'].append('no words')

        else:
            annotation['bboxes'] = [ann_word['bbox'] for ann_word in ann_words]
            annotation['shifts'] = [ann.begin - ann_words[0]['offsets'][0], ann.end - ann_words[-1]['offsets'][1]]

            if print_text:
                aligned_print(rebuild["fulltext"][ann_words[0]['offsets'][0]:ann_words[-1]['offsets'][1]],
                              cas.sofa_string[ann.begin:ann.end],
                              ann.value)

        annotations.append(annotation)

    return annotations


ann = import_page_annotations_from_xmi('sophoclesplaysa05campgoog_0146', PATHS['ne_corpus'], ANNOTATION_LAYERS['entity'])

#%%
# cas = get_cas(Path('/Users/sven/data/AjMC-NE-corpus/data/preparation/corpus/de/curated/sophokle1v3soph_0017.xmi'),
#                Path('/Users/sven/data/AjMC-NE-corpus/data/preparation/TypeSystem.xml'))
#
# anns = {}
# for layer in [NE_TYPE, SEGMENT_TYPE, SENTENCE_TYPE, HYPHENATION_TYPE, TOKEN_TYPE]:
#     anns[layer]  = cas.select(layer)
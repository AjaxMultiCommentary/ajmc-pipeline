import os
from typing import Optional
from ajmc.nlp.token_classification.config import parse_config_from_json
from ajmc.commons.variables import ORDERED_OLR_REGION_TYPES

excluded_region_types = ['line_number_commentary', 'handwritten_marginalia', 'undefined', 'line_region'] # todo in configs

rois = [rt for rt in ORDERED_OLR_REGION_TYPES if rt not in excluded_region_types]

regions_to_coarse_labels = {
    # Commentary
    'commentary': 'commentary',
    # Primary text
    'primary_text': 'primary_text',
    # Footnotes
    'footnote': 'footnote',
    # Running header
    'running_header': 'running_header',
    # Paratext
    'preface': 'paratext',
    'introduction': 'paratext',
    'appendix': 'paratext',
    # Numbers
    'line_number_text': 'numbers',
    'line_number_commentary': 'numbers',
    'page_number': 'numbers',
    # App Crit
    'app_crit': 'app_crit',
    # Others
    'translation': 'others',
    'bibliography': 'others',
    'index_siglorum': 'others',
    'table_of_contents': 'others',
    'title': 'others',
    'printed_marginalia': 'others',
    'handwritten_marginalia': 'others',
    'other': 'others',
    'undefined': 'others',
    'line_region': 'others'
}

coarse_labels_to_ids = {l: i for i, l in enumerate(sorted(list(regions_to_coarse_labels.values())))}

ids_to_coarse_labels = {l: i for i, l in coarse_labels_to_ids.items()}

ner_labels_to_ids = {
    'B-commentary': 0,
    'I-commentary': 1,
    'B-primary_text': 2,
    'I-primary_text': 3,
    'B-paratext': 4,
    'I-paratext': 5,
    'B-numbers': 6,
    'I-numbers': 7,
    'B-app_crit': 8,
    'I-app_crit': 9,
    'B-others': 10,
    'I-others': 11,
}
ids_to_ner_labels = {v: k for k, v in ner_labels_to_ids.items()}


def create_olr_config(json_path: Optional[str] = None,
                      prefix=None
                      # '/content/drive/MyDrive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/'
                      ):
    config = parse_config_from_json(json_path=json_path)
    if prefix:
        new_data_dirs = {}
        for set_ in config['data_dirs_and_sets']:
            new_data_dirs[set_] = {}
            for path, it in config['data_dirs_and_sets'][set_].items():
                new_data_dirs[set_][os.path.join(prefix, path)] = it
        config['data_dirs_and_sets'] = new_data_dirs
    config['regions_to_coarse_labels'] = regions_to_coarse_labels
    config['labels_to_ids'] = coarse_labels_to_ids
    config['ids_to_labels'] = ids_to_coarse_labels
    config['model_inputs'] = ['input_ids', 'bbox', 'token_type_ids', 'attention_mask', 'image', 'labels']
    config['splits'] = ['train', 'dev']
    config['rois'] = rois
    config['num_labels'] = len(list(coarse_labels_to_ids.keys()))

    return config

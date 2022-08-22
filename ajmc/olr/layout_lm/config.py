import os
from typing import Optional
from ajmc.nlp.token_classification.config import parse_config_from_json
from ajmc.commons.variables import ORDERED_OLR_REGION_TYPES, PATHS

region_types_to_labels = {
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


# todo centralize config handling
def create_olr_config(json_path: Optional[str] = None,
                      prefix=None  # todo add this were this is called
                      # '/content/drive/MyDrive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/'
                      ):
    config = parse_config_from_json(json_path=json_path)

    # handles data paths
    if prefix is None:
        prefix = PATHS['base_dir']

    for set_, data_list in config['data '].items:
        for dict_ in data_list:
            dict_['path'] = os.path.join(prefix, dict_['id'], PATHS['canonical'], dict_['run']+'.json')

    config['rois'] = [rt for rt in ORDERED_OLR_REGION_TYPES if rt not in config['excluded_region_types']]
    config['region_types_to_labels'] = region_types_to_labels
    config['labels_to_ids'] = {l: i for i, l in enumerate(sorted(list(config['region_types_to_labels'].values())))}
    config['ids_to_labels'] = {l: i for i, l in config['labels_to_ids'].items()}
    config['num_labels'] = len(list(config['labels_to_ids'].keys()))

    return config
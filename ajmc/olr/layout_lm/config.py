import os
from typing import Optional
from ajmc.nlp.token_classification.config import parse_config_from_json
from ajmc.commons.variables import ORDERED_OLR_REGION_TYPES, PATHS


# todo 👁️ centralize config handling
def create_olr_config(json_path: str,
                      prefix: str):

    config = parse_config_from_json(json_path=json_path)

    for set_, data_list in config['data'].items():
        for dict_ in data_list:
            dict_['path'] = os.path.join(prefix, dict_['id'], PATHS['canonical'], dict_['run']+'.json')

    config['rois'] = [rt for rt in ORDERED_OLR_REGION_TYPES if rt not in config['excluded_region_types']]
    config['labels_to_ids'] = {l: i for i, l in enumerate(sorted(set(config['region_types_to_labels'].values())))}
    config['ids_to_labels'] = {l: i for i, l in config['labels_to_ids'].items()}
    config['num_labels'] = len(list(config['labels_to_ids'].keys()))
    if 'sampling' not in config.keys():
        config['sampling'] = None

    return config



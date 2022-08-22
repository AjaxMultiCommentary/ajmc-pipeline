import os
from typing import Optional
from ajmc.nlp.token_classification.config import parse_config_from_json
from ajmc.commons.variables import ORDERED_OLR_REGION_TYPES, PATHS


# todo centralize config handling
def create_olr_config(json_path: Optional[str] = None,
                      prefix=None):

    config = parse_config_from_json(json_path=json_path)

    # handles data paths
    if prefix is None:
        prefix = PATHS['base_dir']

    for set_, data_list in config['data'].items():
        for dict_ in data_list:
            dict_['path'] = os.path.join(prefix, dict_['id'], PATHS['canonical'], dict_['run']+'.json')

    config['rois'] = [rt for rt in ORDERED_OLR_REGION_TYPES if rt not in config['excluded_region_types']]
    config['labels_to_ids'] = {l: i for i, l in enumerate(sorted(list(config['region_types_to_labels'].values())))}
    config['ids_to_labels'] = {l: i for i, l in config['labels_to_ids'].items()}
    config['num_labels'] = len(list(config['labels_to_ids'].keys()))

    return config


config = create_olr_config(json_path='/data/layoutlm/configs/4A_omnibus_base.json',
                           prefix= PATHS['base_dir'])
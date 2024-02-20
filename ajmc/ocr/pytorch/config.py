"""Utilities for reading and processing config files."""

import json
import unicodedata
from pathlib import Path

from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.commons.unicode_utils import SUPERSCRIPT_MAPPING, SUBSCRIPT_MAPPING


logger = get_ajmc_logger(__name__)


def get_config(config_path: Path) -> dict:
    """Reads a config file from json, processes it and returns a dictionary of the config."""
    logger.info(f'Loading config from {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))

    # Handle classes
    config['classes'] = unicodedata.normalize('NFD', config['classes'])  # Makes sure we have no combining chars issues with json
    special_classes = ''.join([c[1] for c in config.get('special_classes', [])])  # Get the special classes
    if not config['classes'].startswith(special_classes):
        config['classes'] = special_classes + config['classes']  # Add special classes to the beginning
    config['num_classes'] = len(config['classes'])

    # Create mappings from chars to special classes and vice versa
    config['special_classes_dict'] = {c[0]: c[1] for c in config.get('special_classes', [])}  # Create a dictionary of special classes for easy access
    # Get the mappings for superscript and subscript
    config['chars_to_special_classes'] = {k: config['special_classes_dict']['superscript'] + v for k, v in SUPERSCRIPT_MAPPING.items()}
    config['chars_to_special_classes'].update({k: config['special_classes_dict']['subscript'] + v for k, v in SUBSCRIPT_MAPPING.items()})
    # Add eventual custom special mappings
    config['chars_to_special_classes'].update(config.get('special_mapping', {}))
    config['special_classes_to_chars'] = {v: k for k, v in config['chars_to_special_classes'].items()}

    # Create mappings from chars to indices and vice versa
    # We do not want the blank and unknown characters to be in the mapping
    config['classes_to_indices'] = {class_: i for i, class_ in enumerate(config['classes'][2:])}
    config['indices_to_classes'] = {i: class_ for i, class_ in enumerate(config['classes'][2:])}

    # Handle paths
    for k, v in config.items():
        if k.endswith('_dir') or k.endswith('_path'):
            config[k] = Path(v) if v is not None else None

    # Others
    config['decoder'] = {'in_features': config['encoder']['TransformerEncoderLayer']['d_model'], 'out_features': config['num_classes']}
    config['config_name'] = config_path.name

    return config


def write_config_to_output_dir(config: dict):
    # Write the config back to the file and to the output directory
    excluded_keys = ['classes_to_indices', 'indices_to_classes', 'special_classes_dict',
                     'special_classes_to_chars']
    write_config = {k: v for k, v in config.items() if k not in excluded_keys}
    config_json = json.dumps(write_config, ensure_ascii=False, indent=2, default=lambda x: str(x), sort_keys=True)
    (config['output_dir'] / config['config_name']).write_text(config_json, encoding='utf-8')

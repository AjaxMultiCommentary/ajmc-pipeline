"""Utilities for reading and processing config files."""

import json
import os
import unicodedata
from pathlib import Path

from ajmc.commons.miscellaneous import get_ajmc_logger

logger = get_ajmc_logger(__name__)


def get_config(config_path: Path) -> dict:
    """Reads a config file from json, processes it and returns a dictionary of the config."""

    logger.info(f'Loading config from {config_path}')

    config = json.loads(config_path.read_text(encoding='utf-8'))

    # Handle classes
    config['classes'] = unicodedata.normalize('NFD', config['classes'])
    config['num_classes'] = len(config['classes'])

    # Handle paths
    for k, v in config.items():
        if k.endswith('_dir') or k.endswith('_path'):
            config[k] = Path(v) if v is not None else None

    # Others
    config['decoder'] = {'in_features': config['encoder']['TransformerEncoderLayer']['d_model'], 'out_features': config['num_classes']}
    config['num_workers'] = int(os.environ.get('WORLD_SIZE', 1))

    # Write the config back to the file and to the output directory
    config_json = json.dumps(config, ensure_ascii=False, indent=2, default=lambda x: str(x), sort_keys=True)
    # config_path.write_text(config_json, encoding='utf-8')
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    (config['output_dir'] / config_path.name).write_text(config_json, encoding='utf-8')

    config['classes_to_indices'] = {class_: i for i, class_ in enumerate(config['classes'][2:], start=2)}
    # We do not want the blank and unknown characters to be in the mapping
    config['indices_to_classes'] = {i: class_ for i, class_ in enumerate(config['classes'])}

    return config

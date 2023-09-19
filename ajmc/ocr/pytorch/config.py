"""Utilities for reading and processing config files."""

import json
import os
from pathlib import Path

import unicodedata


def get_config(config_path: Path, rewrite: bool = True) -> dict:
    """Reads a config file from json, processes it and returns a dictionary of the config."""
    config = json.loads(config_path.read_text(encoding='utf-8'))

    # Handle classes
    config['classes'] = unicodedata.normalize('NFD', config['classes'])
    if not config['classes'].startswith(config['blank_class']):
        config['classes'] = config['blank_class'] + config['classes']
    config['num_classes'] = len(config['classes'])

    # Handle paths
    for k, v in config.items():
        if k.endswith('_dir') or k.endswith('_path'):
            config[k] = Path(v)

    # Others
    config['decoder'] = {'in_features': config['encoder']['TransformerEncoderLayer']['d_model'], 'out_features': config['num_classes']}

    config['classes_to_indices'] = {class_: i for i, class_ in enumerate(config['classes'])}

    config['indices_to_classes'] = {i: class_ for class_, i in config['classes_to_indices'].items()}

    config['num_workers'] = int(os.environ.get('WORLD_SIZE', 1))

    if rewrite:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=lambda x: str(x), sort_keys=True), encoding='utf-8')

    return config

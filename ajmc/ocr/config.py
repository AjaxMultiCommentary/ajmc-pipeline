"""This module handles reading of configs from tsv-files.

Note:
    Configs are stored in central tsv files rather than json for easier comparability.
"""
import pandas as pd
from pathlib import Path
from typing import Dict
from ajmc.ocr import variables as ocr_vars


def row_to_dataset_config(row: 'pd.Series'):
    # Extract dict
    raw_config = row.to_dict()

    config = {'id': raw_config['id'],  # id and source are str and list[str]
              'source': raw_config['source'].split('-'),
              'sampling': {k: v.split('-') if type(v) == str else v
                           for k, v in raw_config.items() if k in ocr_vars.SAMPLING_TYPES},
              'transform': {k: v for k, v in raw_config.items() if k in ocr_vars.TRANSFORM_OPERATIONS}}

    return config


def row_to_model_config(row: 'pd.Series'):
    return {k: v.split(',') if type(v) == str and k in ['script', 'language'] else v
            for k, v in row.to_dict().items()}


def row_to_experiment_config(row: pd.Series) -> dict:
    return {k: v.split('+') if k == 'models' else v
            for k, v in row.to_dict().items()}


def get_config_reader(config_type: str):
    return row_to_dataset_config if config_type == 'datasets' \
        else row_to_model_config if config_type == 'models' \
        else row_to_experiment_config


def config_to_tesstrain_config(config, cores: int = 12):
    """Transforms a config to a tesstrain config which will be added to the training command.

    Note:
        See https://github.com/tesseract-ocr/tesstrain for all possible parameters.
    """
    tesstrain_config = {}
    tesstrain_config['MODEL_NAME'] = config["id"]
    tesstrain_config['START_MODEL'] = config['source']
    tesstrain_config['GROUND_TRUTH_DIR'] = str(ocr_vars.get_dataset_dir(config['train_dataset']))
    tesstrain_config['LANGDATA_DIR'] = str(ocr_vars.LANGDATA_DIR)
    tesstrain_config['TESSDATA'] = ocr_vars.get_traineddata_dir(config['source'])
    tesstrain_config['CORES'] = cores
    tesstrain_config['EPOCHS'] = config['epochs']
    tesstrain_config['LEARNING_RATE'] = config['learning_rate']
    tesstrain_config['PSM'] = 7
    tesstrain_config['RATIO_TRAIN'] = config['train_ratio']

    return tesstrain_config


def get_all_configs(xl_path: Path = ocr_vars.CONFIGS_PATH) -> Dict[str, Dict[str, dict]]:
    configs = {}

    for config_type in ['experiments', 'datasets', 'models']:
        df = pd.read_excel(xl_path, sheet_name=config_type, keep_default_na=False)
        configs[config_type] = {r['id']: get_config_reader(config_type)(r.apply(lambda x: None if x == '' else x)) for
                                _, r in df.iterrows()}

    return configs

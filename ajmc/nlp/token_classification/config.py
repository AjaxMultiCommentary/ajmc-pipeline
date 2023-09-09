"""This module handles the configs"""

import json
from typing import Dict, Any

import torch
from transformers import TrainingArguments

from ajmc.commons.miscellaneous import get_ajmc_logger

logger = get_ajmc_logger(__name__)

def create_default_config() -> Dict[str, Any]:
    """Creates a default token-classification config."""

    config = dict()

    # ================ PATHS AND DIRS ==================================================================================
    config['train_path']: str = None  # Absolute path to the tsv data file to train on # Required: False
    config['train_url']: str = None  # url to the tsv data file to train on # Required: False
    config['eval_path']: str = None  # Absolute path to the tsv data file to evaluate on # Required: False
    config['eval_url']: str = None  # url to the tsv data file to evaluate on # Required: False
    config[
        'output_dir']: str = None  # Absolute path to the directory in which outputs are to be stored # Required: False
    config[
        'hipe_script_path']: str = None  # The path the CLEF-HIPE-evaluation script. This parameter is required if ``do_hipe_eval`` is True # Required: False
    config[
        'config_path']: str = None  # The path to a config json file from which to extract config. Overwrites other specified config # Required: False
    config['predict_paths']: list = []  # A list of tsv files to predict # Required: False
    config['predict_urls']: list = []  # A list of tsv files-urls to predict # Required: False

    # ================ DATA RELATED ====================================================================================
    config['labels_column']: str = None  # Name of the tsv col to extract labels from # Required: False
    config['unknownify_tokens']: bool = False  # Sets all tokens to '[UNK]'. Useful for ablation experiments. # Required: False
    # config['sampling'] # todo 👁️ add ?

    # ================ MODEL INFO ======================================================================================
    config[
        'model_name_or_path']: str = None  # Absolute path to model directory  or HF model name (e.g. 'bert-base-cased') # Required: False

    # =================== ACTIONS ======================================================================================
    config['do_train']: bool = False  # whether to train. Leave to false if you just want to evaluate
    config['do_eval']: bool = False  # Performs CLEF-HIPE evaluation, alone or at the end of training if ``do_train``.
    config['do_predict']: bool = False  # Predicts on ``predict_urls`` or/and ``predict_paths``
    config['evaluate_during_training']: bool = False  # Whether to evaluate during training.
    config['do_debug']: bool = False  # Breaks all loops after a single iteration for debugging
    config['overwrite_output_dir']: bool = False  # Whether to overwrite the output dir
    config[
        'do_early_stopping']: bool = False  # Breaks stops training after ``early_stopping_patience`` epochs without improvement.

    # =============================== TRAINING PARAMETERS ==============================================================
    config['device_name']: str = "cuda:0"  # Device in the format 'cuda:1', 'cpu'
    config['epochs']: int = 3  # Total number of training epochs to perform.
    config['early_stopping_patience']: int = 3  # Number of epochs to wait for early stopping
    config['seed']: int = 42  # Random seed
    config['batch_size']: int = 8  # Batch size per device.
    config['gradient_accumulation_steps']: int = 1  # Number of steps to accumulate before performing backpropagation.

    # ===================================== HF PARAMETERS ==============================================================
    default_hf_args: dict = vars(TrainingArguments(''))
    config.update(**{arg: default_hf_args[arg] for arg in ['local_rank', 'weight_decay', 'max_grad_norm',
                                                           'adam_epsilon', 'learning_rate', 'warmup_steps']})
    return config


def parse_config_from_json(json_path: str) -> Dict[str, Any]:
    """Parses config from a json file.

    Also transforms ``config['device_name']`` to ``torch.device(config['device_name'])``, raising an error if ``cuda[:#]``
    is set but not available.
    """

    with open(json_path, "r") as file:
        json_args = json.loads(file.read())

    config = create_default_config()
    config.update(**{arg: json_args[arg] for arg in json_args.keys()})

    if config['device_name'].startswith("cuda") and not torch.cuda.is_available():
        logger.error("You set ``device_name`` to {} but cuda is not available, setting device to cpu.".format(config['device_name']))
        config['device'] = torch.device('cpu')

    else:
        config['device'] = torch.device(config['device_name'])


    return config



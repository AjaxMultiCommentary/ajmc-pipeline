"""This module handles the CLI config with `argparse`."""

import json
import argparse
import sys
import torch
from transformers import TrainingArguments
from typing import Optional


def create_pipeline_parser() -> argparse.ArgumentParser:
    """Adds `pipeline.py` relative arguments to the parser"""

    parser = argparse.ArgumentParser()

    # ================ PATHS AND DIRS ==================================================================================

    parser.add_argument("--train_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Absolute path to the tsv data file to train on")

    parser.add_argument("--train_url",
                        type=str,
                        required=False,
                        default=None,
                        help="url to the tsv data file to train on")

    parser.add_argument("--eval_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Absolute path to the tsv data file to evaluate on")

    parser.add_argument("--eval_url",
                        type=str,
                        required=False,
                        default=None,
                        help="url to the tsv data file to evaluate on")

    parser.add_argument("--output_dir",
                        type=str,
                        required=False,
                        help="Absolute path to the directory in which outputs are to be stored")

    parser.add_argument("--hipe_script_path",
                        type=str,
                        required=False,
                        help="The path the CLEF-HIPE-evaluation script. This parameter is required if `do_hipe_eval`"
                             "is True")

    parser.add_argument("--config_path",
                        type=str,
                        required=False,
                        help="The path to a config json file from which to extract config. "
                             "Overwrites other specified config")

    # ================ DATA RELATED ====================================================================================
    parser.add_argument("--labels_column",
                        type=str,
                        required=False,
                        help="Name of the tsv col to extract labels from")

    # ================ MODEL INFO ======================================================================================
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=False,
                        help="Absolute path to model directory  or HF model name (e.g. 'bert-base-cased')")

    # =================== ACTIONS ======================================================================================

    parser.add_argument("--do_train",
                        action="store_true",
                        default=False,
                        help="whether to train. Leave to false if you just want to evaluate")

    parser.add_argument("--do_eval",
                        action="store_true",
                        default=False,
                        help="Performs CLEF-HIPE evaluation, alone or at the end of training if `do_train`.")

    parser.add_argument("--do_debug",
                        action="store_true",
                        default=False,
                        help="Breaks all loops after a single iteration for debugging")

    parser.add_argument("--overwrite_output_dir",
                        action="store_true",
                        default=False,
                        help="Whether to overwrite the output dir")

    parser.add_argument("--do_early_stopping",
                        action="store_true",
                        default=False,
                        help="Breaks stops training after `early_stopping_patience` epochs without improvement.")

    # =============================== TRAINING PARAMETERS ==============================================================

    parser.add_argument("--device_name",
                        type=str,
                        default="cuda:0",
                        help="Device in the format 'cuda:1', 'cpu'")

    parser.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--early_stopping_patience",
                        type=int,
                        default=3,
                        help="Number of epochs to wait for early stopping")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed")

    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size per device.")

    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate before performing backpropagation.")

    return parser


def parse_config_from_json(json_path: str) -> argparse.Namespace:
    """Parses config from a json file. If `config` is provided, it only gets updated. """

    config = create_pipeline_parser().parse_args([])
    with open(json_path, "r") as file:
        json_args = json.loads(file.read())
    config.__dict__.update(**{arg: json_args[arg] for arg in json_args.keys()})
    return config


def post_initialize_config(config: argparse.Namespace) -> argparse.Namespace:
    """Adds some attributes to `config` after initialization, both manually and from `transformers.TrainingArguments`.

    Also transforms `config.device_name` to `torch.device(config.device_name)`, raising an error if `cuda[:#]`
    is set but not available.
    """
    # Todo first path on data to get the labels should be done here 
    # Adds some nitty-gritty training arguments
    default_hf_args: dict = vars(TrainingArguments(''))

    config.__dict__.update(**{arg: default_hf_args[arg] for arg in ['local_rank', 'weight_decay',
                                                                    'max_grad_norm', 'adam_epsilon', 
                                                                    'learning_rate', 'warmup_steps']})


    if config.device_name.startswith("cuda"):
        assert torch.cuda.is_available(), "You set `device_name` to {} but cuda is not available.".format(
            config.device_name)
    config.device = torch.device(config.device_name)

    return config


def initialize_config(json_path: str = None) -> argparse.Namespace:
    """The general idea here is to always look for a `json_path` as parameter or as attribute to `config`, and to parse
    config from the corresponding json.

    So the list of priorities would be:
        1) If `json_path`, parse the corresponding json
        2) elif something is passed to CLI and contain `json_path`, parse json. If not, parse from CLI.
        3) else parse manually (then again, from json if manual config contain `json_path`"""

    if json_path:
        return post_initialize_config(parse_config_from_json(json_path=json_path))

    # If `pipeline.py` is called from cli
    elif sys.argv[0].endswith("pipeline.py") and len(sys.argv) > 1:
        config = create_pipeline_parser().parse_args()
        if config.config_path:
            config = parse_config_from_json(config.config_path)
        return post_initialize_config(config)




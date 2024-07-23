"""This module handles the configs"""

import json
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Optional, List

import torch
from transformers import TrainingArguments

from ajmc.commons.miscellaneous import get_ajmc_logger

logger = get_ajmc_logger(__name__)


@dataclass
class AjmcNlpConfig:
    """This class holds the configuration for the NLP pipeline"""

    # ================ PATHS AND DIRS ==================================================================================
    train_path: Optional[Path] = None  # Absolute path to the tsv data file to train on # Required: False
    train_url: Optional[str] = None  # url to the tsv data file to train on # Required: False
    eval_path: Optional[Path] = None  # Absolute path to the tsv data file to evaluate on # Required: False
    eval_url: Optional[str] = None  # url to the tsv data file to evaluate on # Required: False
    output_dir: Optional[Path] = None  # Absolute path to the directory in which outputs are to be stored # Required: False
    hipe_script_path: Optional[Path] = None  # The path the CLEF-HIPE-evaluation script. This parameter is required if ``do_hipe_eval``
    config_path: Path = None  # The path to a config json file from which to extract config. Overwrites other specified config # Required: False
    predict_paths: List[Path] = field(default_factory=list)  # A list of tsv files to predict # Required: False
    predict_urls: List[str] = field(default_factory=list)  # A list of tsv files-urls to predict # Required: False

    # ================ DATA RELATED ====================================================================================
    labels_column: Optional[str] = None  # Name of the tsv col to extract labels from # Required: False
    unknownify_tokens: bool = False  # Sets all tokens to '[UNK]'. Useful for ablation experiments. # Required: False
    data_format: str = 'ner'  # The format of the data. 'ner' or 'lemlink' # Required: False

    # ================ MODEL INFO ======================================================================================
    model_name_or_path: Optional[Path] = None  # Absolute path to model directory  or HF model name (e.g. 'bert-base-cased') # Required: False
    model_max_length: Optional[int] = None  # Maximum length of the input sequence # Required: False # Leave to None to default to model's

    # =================== ACTIONS ======================================================================================
    do_train: bool = False  # whether to train. Leave to false if you just want to evaluate
    do_hipe_eval: bool = False  # Performs CLEF-HIPE evaluation, alone or at the end of training if ``do_train``.
    do_seqeval: bool = False  # Performs seqeval evaluation, alone or at the end of training if ``do_train``.
    do_predict: bool = False  # Predicts on ``predict_urls`` or/and ``predict_paths``
    do_save: bool = True  # Saves the model after training
    evaluate_during_training: bool = False  # Whether to evaluate during training.
    overwrite_outputs: bool = False  # Whether to overwrite the output in the output directory
    do_early_stopping: bool = False  # Breaks stops training after ``early_stopping_patience`` epochs without improvement.
    do_debug: bool = False  # Breaks all loops after a single iteration for debugging purposes

    # =============================== TRAINING PARAMETERS ==============================================================
    device_name: str = 'cuda:0'  # Device in the format 'cuda:1', 'cpu'
    epochs: int = 3  # Total number of training epochs to perform.
    early_stopping_patience: int = 3  # Number of epochs to wait for early stopping
    seed: int = 42  # Random seed
    batch_size: int = 8  # Batch size per device.
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate before performing backpropagation.


    def __post_init__(self):
        # Set the device as a torch device
        if self.device_name.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError(f'You set ``device_name`` to {self.device_name} but cuda is not available')
        self.device: torch.device = torch.device(self.device_name)

        # Create the output subdirectories
        self.model_save_dir: Path = self.output_dir / 'model'
        self.predictions_dir: Path = self.output_dir / 'predictions'
        self.seqeval_output_dir: Path = self.output_dir / 'results/seqeval'
        self.hipe_output_dir: Path = self.output_dir / 'results/hipe_eval'

        # Add HF training arguments
        default_hf_args: dict = vars(TrainingArguments(''))
        for arg in ['local_rank', 'weight_decay', 'max_grad_norm', 'adam_epsilon', 'learning_rate', 'warmup_steps']:
            setattr(self, arg, default_hf_args[arg])


    @classmethod
    def from_dict(cls, config: dict) -> 'AjmcNlpConfig':
        """Creates a config from a dictionary"""

        # Convert paths to Path objects
        for k, v in config.items():
            if (k.endswith('_path') or k.endswith('_dir')) and v is not None and k != 'model_name_or_path':  # Todo 👁️ fix this
                config[k] = Path(v)

        if config.get('predict_paths', False):
            config['predict_paths'] = [Path(p) for p in config['predict_paths']]

        return cls(**config)

    @classmethod
    def from_json(cls, path: Path) -> 'AjmcNlpConfig':
        """Loads a config from a json file"""
        config = json.loads(path.read_text(encoding='utf-8'))
        return cls.from_dict(**config)

    def to_json(self, path: Path):
        """Saves the config to a json file"""
        path.write_text(json.dumps({f.name: getattr(self, f.name) for f in fields(self)},
                                   skipkeys=True, indent=2, sort_keys=True,
                                   ensure_ascii=False, default=lambda x: str(x)),
                        encoding='utf-8')

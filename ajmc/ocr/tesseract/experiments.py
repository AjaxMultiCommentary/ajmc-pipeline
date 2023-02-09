"""Contains the code for running experiments."""
import json
from typing import List, Optional

import ajmc.ocr.evaluation as ocr_eval
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import CONFIGS
from ajmc.ocr.preprocessing import data_preparation
from ajmc.ocr.tesseract.models import make_model, run

logger = get_custom_logger(__name__)


def make_experiment_dir(experiment_id: str):
    """Creates an empty experiment directory with its subdirectories"""
    ocr_vs.get_experiment_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_experiment_model_outputs_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_experiment_models_dir(experiment_id).mkdir(parents=True, exist_ok=True)


def make_experiment(xp_config: dict,
                    overwrite: bool = False):
    """Creates the experiment repo"""

    logger.info(f"Making experiment {xp_config['id']}")

    # Get the experiment's paths
    xp_models_dir = ocr_vs.get_experiment_models_dir(xp_config['id'])
    xp_model_outputs_dir = ocr_vs.get_experiment_model_outputs_dir(xp_config['id'])
    xp_config_path = ocr_vs.get_experiment_config_path(xp_config['id'])

    # Check if the experiment already exists
    if xp_config_path.is_file() and not overwrite:  # if the config file exists
        existing_xp_config = json.loads(xp_config_path.read_text(encoding='utf-8'))
        assert xp_config == existing_xp_config, f"""An experiment with id {xp_config['id']} already exists but its model_config is different. Please check manually."""

    # If the experiment does not already exist
    make_experiment_dir(xp_config['id'])  # Create the experiment's repository

    # Get the required test datasets exist, else create it
    test_dataset_config = CONFIGS['datasets'][xp_config['test_dataset']]
    test_dataset_dir = ocr_vs.get_dataset_dir(test_dataset_config['id'])
    data_preparation.make_dataset(test_dataset_config, overwrite=overwrite)

    # Check if the required models exists, build if not
    for model_id in xp_config['models']:
        model_config = CONFIGS['models'][model_id]
        model_path = ocr_vs.get_trainneddata_path(model_config['id'])
        make_model(model_config, overwrite=overwrite)
        # copy the traineddata file to the experiment's models directory
        (xp_models_dir / model_path.name).write_bytes(model_path.read_bytes())

    # Run the xp's traineddatas on the test datasets
    run(img_dir=test_dataset_dir,
        output_dir=xp_model_outputs_dir,
        langs='+'.join(xp_config['models']),
        psm=7,
        tessdata_prefix=xp_models_dir)

    # Evaluate the outputs
    ocr_eval.line_by_line_evaluation(gt_dir=test_dataset_dir,
                                     ocr_dir=xp_model_outputs_dir,
                                     output_dir=xp_model_outputs_dir.parent, )

    # Save the config file
    xp_config_path.write_text(json.dumps(xp_config, indent=4), encoding='utf-8')


def make_experiments(experiment_ids: Optional[List[str]] = None, overwrite: bool = False):
    """Makes the experiments"""
    for xp_id, xp_config in CONFIGS['experiments'].items():
        if experiment_ids is None or xp_id in experiment_ids:
            make_experiment(xp_config, overwrite=overwrite)

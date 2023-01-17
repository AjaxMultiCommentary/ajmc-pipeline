"""
Contains the code for running experiments.
"""
import json
from pathlib import Path
from ajmc.ocr.config import get_all_configs
from ajmc.ocr.preprocessing import data_preparation
from ajmc.ocr.tesseract.models import get_or_make_traineddata_path, run
from ajmc.ocr import variables as ocr_vars


def make_experiment_dir(experiment_id: str):
    """Creates an empty experiment directory with its subdirectories"""
    ocr_vars.get_experiment_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_experiment_model_outputs_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_experiment_models_dir(experiment_id).mkdir(parents=True, exist_ok=True)

def get_or_make_experiment_dir(xp_config: dict,
                               overwrite: bool = False) -> Path:
    """Creates the experiment repo"""

    # Get the experiment's paths
    xp_dir = ocr_vars.get_experiment_dir(xp_config['id'])
    xp_models_dir = ocr_vars.get_experiment_models_dir(xp_config['id'])
    xp_model_outputs_dir = ocr_vars.get_experiment_model_outputs_dir(xp_config['id'])
    xp_config_path = ocr_vars.get_experiment_config_path(xp_config['id'])

    # Check if the experiment already exists
    if xp_dir.is_dir() and not overwrite:  # if the experiment already exists
        if xp_config_path.is_file():  # if the config file exists
            existing_xp_config = json.loads(xp_config_path.read_text(encoding='utf-8'))
            assert xp_config == existing_xp_config, f"""An experiment with id {xp_config['id']} already exists but its model_config is different. Please check manually."""
            return xp_dir

    # If the experiment does not already exist
    make_experiment_dir(xp_config['id'])  # Create the experiment's repository
    configs = get_all_configs()

    # Get the required test datasets exist, else create it
    test_dataset_config = configs['datasets'][xp_config['test_dataset']]
    test_dataset_dir = data_preparation.get_or_make_dataset_dir(test_dataset_config, overwrite=overwrite)

    # Check if the required models exists, build if not
    for model_id in xp_config['models']:
        model_config = configs['models'][model_id]
        traineddata_path = get_or_make_traineddata_path(model_config, overwrite=overwrite)
        # copy the traineddata file to the experiment's models directory
        (xp_models_dir / traineddata_path.name).write_bytes(traineddata_path.read_bytes())

    # Run the xp's traineddatas on the test datasets
    run(img_dir=test_dataset_dir,
        output_dir=xp_model_outputs_dir,
        langs='+'.join(xp_config['models']),
        psm=7,
        tessdata_prefix=xp_models_dir)



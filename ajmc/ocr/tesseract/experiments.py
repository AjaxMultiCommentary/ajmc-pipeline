"""Contains the code for running experiments."""
import json
from typing import List, Optional

import pandas as pd

import ajmc.ocr.evaluation as ocr_eval
from ajmc.commons.file_management import walk_dirs
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import CONFIGS
from ajmc.ocr.preprocessing import data_preparation
from ajmc.ocr.tesseract.models import make_model, run

logger = get_ajmc_logger(__name__)


def make_experiment_dir(experiment_id: str):
    """Creates an empty experiment directory with its subdirectories"""
    ocr_vs.get_experiment_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_experiment_model_outputs_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_experiment_models_dir(experiment_id).mkdir(parents=True, exist_ok=True)


def make_experiment(xp_config: dict, overwrite_xps: bool = False, overwrite_models: bool = False,
                    overwrite_datasets: bool = False) -> None:
    """Creates the experiment repo"""

    logger.info(f"Making experiment {xp_config['id']}")

    # Get the experiment's paths
    xp_models_dir = ocr_vs.get_experiment_models_dir(xp_config['id'])
    xp_model_outputs_dir = ocr_vs.get_experiment_model_outputs_dir(xp_config['id'])
    xp_config_path = ocr_vs.get_experiment_config_path(xp_config['id'])

    # Check if the experiment already exists
    if xp_config_path.is_file() and not overwrite_xps:  # if the config file exists
        existing_xp_config = json.loads(xp_config_path.read_text(encoding='utf-8'))
        assert xp_config == existing_xp_config, f"""An experiment with id {xp_config['id']} already exists but its model_config is different. Please check manually."""
        return None

    # If the experiment does not already exist
    make_experiment_dir(xp_config['id'])  # Create the experiment's repository

    # Get the required test datasets exist, else create it
    test_dataset_config = CONFIGS['datasets'][xp_config['test_dataset']]
    test_dataset_dir = ocr_vs.get_dataset_dir(test_dataset_config['id'])
    data_preparation.make_dataset(test_dataset_config, overwrite=overwrite_datasets)

    # Check if the required models exists, build if not
    for model_id in xp_config['models']:
        model_config = CONFIGS['models'][model_id]
        model_path = ocr_vs.get_trainneddata_path(model_config['id'])
        make_model(model_config, overwrite_models=overwrite_models, overwrite_datasets=overwrite_datasets)
        # copy the traineddata file to the experiment's models directory
        (xp_models_dir / model_path.name).write_bytes(model_path.read_bytes())

    # Run the xp's traineddatas on the test datasets
    run(img_dir=test_dataset_dir,
        output_dir=xp_model_outputs_dir,
        langs='+'.join(xp_config['models']),
        psm=7,
        tessdata_prefix=xp_models_dir)

    # Evaluate the outputs
    ocr_eval.directory_evaluation(gt_dir=test_dataset_dir,
                                  ocr_dir=xp_model_outputs_dir,
                                  output_dir=xp_model_outputs_dir.parent, )

    # Save the config file
    xp_config_path.write_text(json.dumps(xp_config, indent=4), encoding='utf-8')


def make_general_results_table():
    """Makes a table with the general results of the experiments"""
    xps_results = pd.DataFrame()
    for xp_dir in walk_dirs(ocr_vs.EXPERIMENTS_DIR):
        # Get the experiment's config
        xp_config = {k: [v] if k != 'models' else ['+'.join(v)] for k, v in CONFIGS['experiments'][xp_dir.name].items()}
        xp_config = pd.DataFrame.from_dict(xp_config)

        # Get the test set's config
        test_set_config = {}
        for k, v in CONFIGS['datasets'][xp_config['test_dataset'][0]].items():
            if k == 'source':
                test_set_config[f'test_set_{k}'] = [','.join(v)]
            elif k in ['sampling', 'transform']:
                for k2, v2 in v.items():
                    test_set_config[f'test_set_{k2}'] = [','.join(v2)] if type(v2) == list else [v2]

        test_set_config = pd.DataFrame.from_dict(test_set_config)

        model_config = {k: [v] if type(v) != list else [','.join(v)] for k, v in CONFIGS['models'][xp_config['models'][0].split('+')[0]].items()}
        model_config = pd.DataFrame.from_dict(model_config)

        xp_results = pd.read_csv((xp_dir / 'results.tsv'), sep='\t')
        xp_results = pd.concat([xp_config, test_set_config, model_config, xp_results], axis=1)
        xps_results = pd.concat([xps_results, xp_results], axis=0)

    xps_results.to_csv(ocr_vs.EXPERIMENTS_DIR / 'general_results.tsv', sep='\t', index=False)


def make_experiments(experiment_ids: Optional[List[str]] = None,
                     overwrite_xps: bool = False,
                     overwrite_datasets: bool = False,
                     overwrite_models: bool = False):
    """Makes the experiments"""
    for xp_id, xp_config in CONFIGS['experiments'].items():
        if experiment_ids is None or xp_id in experiment_ids:
            make_experiment(xp_config, overwrite_xps=overwrite_xps, overwrite_models=overwrite_models,
                            overwrite_datasets=overwrite_datasets)

    make_general_results_table()

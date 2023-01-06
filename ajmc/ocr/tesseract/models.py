import json
import os
import shutil
from pathlib import Path

from ajmc.commons.miscellaneous import log_to_file
from ajmc.ocr import variables as ocr_vars
from ajmc.ocr.tesseract.dictionaries import change_traineddata_wordlist, write_unpacked_traineddata
from ajmc.ocr.config import get_all_configs, config_to_tesstrain_config
from ajmc.ocr.tesseract.tesseract_utils import run_tess_command


def make_model_repo(model_name):
    ocr_vars.get_model_dir(model_name).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_traineddata_dir(model_name).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_model_train_dir(model_name).mkdir(parents=True, exist_ok=True)


def get_or_create_model_path(config: dict, overwrite: bool = False):
    """Creates the model repo"""

    model_dir = ocr_vars.get_model_dir(config['id'])
    model_path = ocr_vars.get_trainneddata_path(config['id'])
    config_path = ocr_vars.get_model_config_path(config['id'])

    if model_dir.is_dir() and not overwrite:  # if the model already exists
        if model_path.is_file() and config_path.is_file():
            existing_model_config = json.loads(config_path.read_text(encoding='utf-8'))
            assert config == existing_model_config, f"""A model with id {config['id']} already exists but its config is different. Please check manually."""
            return ocr_vars.get_trainneddata_path(config['id'])
        else:
            pass

    # Create the model's repository
    make_model_repo(config['id'])

    # if the desired model is a tess native model
    if config['id'] in [p.stem for p in ocr_vars.TESSDATA_DIR.glob('*.traineddata')]:
        # Copy the `.traineddata` file
        model_path.write_bytes((ocr_vars.TESSDATA_DIR / (config['id'] + '.traineddata')).read_bytes())
        # Give the model a config
        ocr_vars.get_model_config_path(config['id']).write_text(json.dumps(config, indent=2), encoding='utf-8')


    else:  # Build the model from its config
        source_model_config = get_all_configs()['models'][config['source']]
        source_model_path = get_or_create_model_path(source_model_config)
        model_path.write_bytes(source_model_path.read_bytes())

        # Change the wordlist if necessary
        change_traineddata_wordlist(config['id'], wordlist_name=config['wordlist'])

        # Give the model a config
        ocr_vars.get_model_config_path(config['id']).write_text(json.dumps(config, indent=2), encoding='utf-8')

        # train the model ?
        if config['train_dataset'] is not None:
            train(config)

    return model_path


def get_model(config: dict, models_dir: Path = ocr_vars.MODELS_DIR) -> Path:
    if config['id'] in list(models_dir.glob('*')):
        existing_model_config = json.loads(ocr_vars.get_model_config_path(config['id']).read_text(encoding='utf-8'))
        assert config == existing_model_config, f"""A model with id {config['id']} already exists in 
        {models_dir} but its config is different. Please check manually."""

    else:
        get_or_create_model_path(config)

    return ocr_vars.get_trainneddata_path(config['id'])


def train(config: dict):

    # Get the model's dir
    model_dir = ocr_vars.get_model_dir(config['id'])

    # Get training dir
    train_dir = ocr_vars.get_model_train_dir(config['id'])
    command_path = train_dir / 'train_command.sh'

    # Creates the logging files
    log_path = train_dir / 'train_log.txt'
    print(f"See {log_path} for the training output.")

    command = f"\
    cd {ocr_vars.TESSTRAIN_DIR}\n \
    make training "

    tesstrain_config = config_to_tesstrain_config(config=config)

    for k, v in tesstrain_config.items():
        command += f'{k}={v} '

    command_path.write_text(command, encoding='utf-8')
    log_to_file(command, log_path)
    output = run_tess_command(command)

    #Write the log
    log_to_file(output.stdout.decode('ascii'), log_path)


def create_all_models(overwrite: bool = False):
    configs = get_all_configs()

    for model_config in configs['models'].values():
        get_or_create_model_path(model_config, overwrite=overwrite)




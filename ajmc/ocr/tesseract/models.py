import json
import subprocess
from pathlib import Path
from ajmc.commons.miscellaneous import log_to_file, get_custom_logger
from ajmc.ocr import variables as ocr_vars
from ajmc.ocr.preprocessing.data_preparation import get_or_create_dataset_dir
from ajmc.ocr.tesseract.dictionaries import change_traineddata_wordlist, write_unpacked_traineddata
from ajmc.ocr.config import get_all_configs, config_to_tesstrain_config

logger = get_custom_logger(__name__)

def make_model_dirs(model_id: str):
    """Creates an empty model with its subdir"""
    ocr_vars.get_model_dir(model_id).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_traineddata_dir(model_id).mkdir(parents=True, exist_ok=True)
    ocr_vars.get_model_train_dir(model_id).mkdir(parents=True, exist_ok=True)


def get_or_create_traineddata_path(model_config: dict, overwrite: bool = False) -> Path:
    """Creates the model repo"""

    # Get the model's paths
    model_dir = ocr_vars.get_model_dir(model_config['id'])
    model_path = ocr_vars.get_trainneddata_path(model_config['id'])
    config_path = ocr_vars.get_model_config_path(model_config['id'])

    # Check if the model already exists
    if model_dir.is_dir() and not overwrite:  # if the model already exists
        if model_path.is_file() and config_path.is_file():
            existing_model_config = json.loads(config_path.read_text(encoding='utf-8'))
            assert model_config == existing_model_config, f"""A model with id {model_config['id']} already exists but its model_config is different. Please check manually."""
            return ocr_vars.get_trainneddata_path(model_config['id'])

    # If the model does not already exist
    make_model_dirs(model_config['id'])  # Create the model's repository

    # if the desired model is a tess native model
    if model_config['id'] in [p.stem for p in ocr_vars.TESSDATA_DIR.glob('*.traineddata')]:
        # Copy the `.traineddata` file
        model_path.write_bytes((ocr_vars.TESSDATA_DIR / (model_config['id'] + '.traineddata')).read_bytes())
        # Give the model a model_config
        ocr_vars.get_model_config_path(model_config['id']).write_text(json.dumps(model_config, indent=2),
                                                                      encoding='utf-8')

    # Else, build the model from its model_config
    else:
        source_model_config = get_all_configs()['models'][model_config['source']]
        source_model_path = get_or_create_traineddata_path(source_model_config,
                                                           overwrite=overwrite)  # Gets the source model recursively
        model_path.write_bytes(source_model_path.read_bytes())

        # Change the wordlist if necessary
        change_traineddata_wordlist(model_config['id'], wordlist_name=model_config['wordlist'])

        # Give the model a model_config
        ocr_vars.get_model_config_path(model_config['id']).write_text(json.dumps(model_config, indent=2),
                                                                      encoding='utf-8')

        # train the model ?
        if model_config['train_dataset'] is not None:
            train_dataset_config = get_all_configs()['datasets'][model_config['train_dataset']]
            get_or_create_dataset_dir(train_dataset_config, overwrite=overwrite)
            train(model_config)

    return model_path


def get_training_command(model_config: dict) -> str:
    """Gets the tess training command given a model model_config"""

    # Prefixes
    command = f"\
cd {ocr_vars.TESSTRAIN_DIR} ;\
export TESSDATA_PREFIX={ocr_vars.TESSDATA_DIR} ; \
export LD_LIBRARY_PATH={ocr_vars.LD_LIBRARY_PATH} ; \
make training "

    # Get the corresponding tesstrain
    tesstrain_config = config_to_tesstrain_config(config=model_config)

    for k, v in tesstrain_config.items():
        command += f'{k}={v} '
    command += '; '

    command += f'cp {ocr_vars.get_model_train_dir(model_config["id"])}/{model_config["id"]}.traineddata {ocr_vars.get_traineddata_dir(model_config["id"])}/{model_config["id"]}.traineddata ; '
    return command


def train(model_config: dict):
    # Get the model's dir
    # model_dir = ocr_vars.get_model_dir(model_config['id'])

    # Get training dir
    train_dir = ocr_vars.get_model_train_dir(model_config['id'])
    command_path = train_dir / 'train_command.sh'

    # Creates the logging files
    log_path = train_dir / 'train_log.txt'
    logger.info(f"Starting training... See {log_path} for the training output.")

    command = get_training_command(model_config)
    command_path.write_text(command, encoding='utf-8')

    log_to_file(command, log_path)
    bash_command = command.encode('ascii')
    output = subprocess.run(['bash '], input=bash_command, shell=True, capture_output=True)

    # Write the log
    log_to_file(output.stdout.decode('ascii'), log_path)






# configs = get_all_configs()
# model_config = configs['models']['test']
#
# get_or_create_traineddata_path(model_config=model_config, overwrite=True)
# print('coucou')

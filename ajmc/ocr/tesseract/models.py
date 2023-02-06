import json
import subprocess
from pathlib import Path
from typing import List, Optional

from ajmc.commons.miscellaneous import get_custom_logger, log_to_file
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import config_to_tesstrain_config, CONFIGS
from ajmc.ocr.preprocessing.data_preparation import make_dataset
from ajmc.ocr.tesseract.dictionaries import change_traineddata_wordlist

logger = get_custom_logger(__name__)


def make_model_dirs(model_id: str):
    """Creates an empty model directory with its subdirectories"""
    ocr_vs.get_model_dir(model_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_traineddata_dir(model_id).mkdir(parents=True, exist_ok=True)
    ocr_vs.get_model_train_dir(model_id).mkdir(parents=True, exist_ok=True)


def make_model(model_config: dict, overwrite: bool = False) -> None:
    """Creates the model repo"""

    # Get the model's paths
    model_path = ocr_vs.get_trainneddata_path(model_config['id'])
    config_path = ocr_vs.get_model_config_path(model_config['id'])

    # Check if the model already exists
    if model_path.is_file() and config_path.is_file() and not overwrite:
        existing_model_config = json.loads(config_path.read_text(encoding='utf-8'))
        assert model_config == existing_model_config, f"""A model with id {model_config['id']} already exists but its model_config is different. Please check manually."""
        return None

    # If the model does not already exist
    make_model_dirs(model_config['id'])  # Create the model's repository

    # if the desired model is a tess native model
    if model_config['id'] in [p.stem for p in ocr_vs.TESSDATA_DIR.glob('*.traineddata')]:
        # Copy the `.traineddata` file
        model_path.write_bytes((ocr_vs.TESSDATA_DIR / (model_config['id'] + '.traineddata')).read_bytes())
        # Give the model a model_config
        ocr_vs.get_model_config_path(model_config['id']).write_text(json.dumps(model_config, indent=2),
                                                                    encoding='utf-8')

    else:  # Else, build the model from its model_config
        logger.info(f"Building model {model_config['id']} from its model_config.")
        source_model_config = CONFIGS['models'][model_config['source']]
        source_model_path = ocr_vs.get_trainneddata_path(source_model_config['id'])
        make_model(source_model_config, overwrite=overwrite)  # Gets the source model recursively
        model_path.write_bytes(source_model_path.read_bytes())

        # Change the wordlist if necessary
        change_traineddata_wordlist(model_config['id'], wordlist_name=model_config['wordlist'])

        # Give the model a model_config
        ocr_vs.get_model_config_path(model_config['id']).write_text(json.dumps(model_config, indent=2),
                                                                    encoding='utf-8')

        # train the model ?
        if model_config['train_dataset'] is not None:
            train_dataset_config = CONFIGS['datasets'][model_config['train_dataset']]
            make_dataset(train_dataset_config, overwrite=overwrite)
            train(model_config)

    # Write the config file
    config_path.write_text(json.dumps(model_config, indent=2), encoding='utf-8')


def get_training_command(model_config: dict) -> str:
    """Gets the tess training command given a model model_config"""

    # Prefixes
    command = f"\
cd {ocr_vs.TESSTRAIN_DIR} ;\
export TESSDATA_PREFIX={ocr_vs.TESSDATA_DIR} ; \
export LD_LIBRARY_PATH={ocr_vs.LD_LIBRARY_PATH} ; \
make training "

    # Get the corresponding tesstrain
    tesstrain_config = config_to_tesstrain_config(config=model_config)

    for k, v in tesstrain_config.items():
        command += f'{k}={v} '
    command += '; '

    command += f'cp {ocr_vs.get_model_train_dir(model_config["id"])}/{model_config["id"]}.traineddata {ocr_vs.get_traineddata_dir(model_config["id"])}/{model_config["id"]}.traineddata ; '
    return command


def train(model_config: dict):
    # Get the model's dir
    # model_dir = ocr_vs.get_model_dir(model_config['id'])

    # Get training dir
    train_dir = ocr_vs.get_model_train_dir(model_config['id'])
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


def run(img_dir: Path,
        output_dir: Path,
        langs: str,
        config: dict = None,
        psm: int = 3,
        img_suffix: str = '.png',
        tessdata_prefix: Path = ocr_vs.TESSDATA_DIR,
        ):
    """Runs tesseract on images in `img_dir`.

    Note:
        assumes tesseract is installed.

    Args:
        img_dir (Path): path to directory containing images to be OCR'd
        output_dir (Path): path to directory where OCR'd text will be saved
        langs (str): language(s) to use for OCR. Use '+' to separate multiple languages, e.g. 'eng+fra'
        config (dict): dictionary of config options to pass to tesseract. See https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
        psm (int): page segmentation mode. See https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
        img_suffix (str): suffix of images to be OCR'd
        tessdata_prefix (Path): path to directory containing tesseract language data
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the config
    if config:
        (output_dir / 'tess_config').write_text('\n'.join([f'{k} {v}' for k, v in config.items()]), encoding='utf-8')

    command = f"""\
cd {img_dir}; export TESSDATA_PREFIX={tessdata_prefix}; \
for i in *{img_suffix} ; \
do tesseract "$i" "{output_dir}/${{i::${{#i}}-4}}" \
-l {langs} \
--psm {psm} \
{(output_dir / 'tess_config') if config else ''}; \
done;"""

    # Writes the command to remember how this was run
    (output_dir / 'command.sh').write_text(command)

    # Write the data related metadata
    if (img_dir / 'metadata.json').is_file():
        (output_dir / 'data_metadata.json').write_bytes((img_dir / 'metadata.json').read_bytes())

    # Run the command
    bash_command = command.encode('ascii')
    logger.info(f"Running tesseract on {img_dir}...")
    subprocess.run(['bash'], input=bash_command, shell=True)


def make_models(models_ids: Optional[List[str]] = None, overwrite: bool = False):
    """Creates datasets.

    Args:
        models_ids: The list of models to create. If None, creates all models.
        overwrite: Wheter to overwrite existing models. Note that this function calls on `get_or_make_trainneddata_path``,
        which is recursive. If `overwrite` is True, all required models will be overwritten
        (i.e. also each models's source-model).
    """

    for model_id, model_config in CONFIGS['models'].items():
        if models_ids is None or model_id in models_ids:
            make_model(model_config, overwrite=overwrite)

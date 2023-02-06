from pathlib import Path

from ajmc.commons import variables as vs

# ======================================================================================================================
#                                VARIABLES
# ======================================================================================================================
SAMPLING_TYPES = ['work_id', 'script', 'language', 'font', 'split', 'random']
TRANSFORM_OPERATIONS = ['resize', 'rotate', 'blur', 'erode', 'dilate']
SEPARATOR = '-'

# ======================================================================================================================
#                                PATHS AND DIRS
# ======================================================================================================================
LOCAL = False if vs.EXEC_ENV == 'iccluster040' else True
XP_DIR = Path('/Users/sven/Desktop/tess_xps') if LOCAL else Path('/scratch/sven/ocr_exp/')
CONDA_INSTALL_DIR = Path('/Users/sven/opt/anaconda3/') if LOCAL else Path('/scratch/sven/anaconda3')
LD_LIBRARY_PATH = CONDA_INSTALL_DIR / 'lib'
TESSTRAIN_DIR = Path('/Users/sven/packages/tesseract') if LOCAL else XP_DIR / 'lib/tesstrain'
TESSDATA_DIR = Path('/Users/sven/packages/tesseract_/tessdata') if LOCAL else XP_DIR / 'lib/tessdata_best'
LANGDATA_DIR = None if LOCAL else XP_DIR / 'lib/langdata_lstm'
DICTIONARIES_DIR = XP_DIR / 'utils/dictionaries'
MODELS_DIR = XP_DIR / 'models'
DATASETS_DIR = XP_DIR / 'datasets'
EXPERIMENTS_DIR = XP_DIR / 'experiments'
CONFIGS_PATH = XP_DIR / 'configs.xlsx'
POG_SOURCE_DIR = XP_DIR / 'data/pogretra-v1.0/Data' if LOCAL else Path('/mnt/ajmcdata1/data/pogretra-v1.0/Data')

LINES_PER_TESTSET = 120

GT_TEXT_EXTENSION = '.gt.txt'
PRED_TEXT_EXTENSION = '.txt'
IMG_EXTENSION = '.png'


def get_dataset_dir(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name


def get_dataset_config_path(dataset_name: str) -> Path:
    return get_dataset_dir(dataset_name) / 'config.json'


def get_dataset_metadata_path(dataset_name: str) -> Path:
    return get_dataset_dir(dataset_name) / 'metadata.tsv'

def get_model_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name


def get_traineddata_dir(traineddata_name: str) -> Path:
    return get_model_dir(traineddata_name) / 'model'


def get_trainneddata_path(traineddata_name: str) -> Path:
    """Get the path to a traineddata file"""
    return get_traineddata_dir(traineddata_name) / (traineddata_name + '.traineddata')


def get_traineddata_unpacked_dir(traineddata_name: str) -> Path:
    """Get the path to the unpacked directory of a traineddata file"""
    return MODELS_DIR / traineddata_name / 'traineddata_unpacked'


def get_model_train_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name / 'train'


def get_model_config_path(model_name: str) -> Path:
    return MODELS_DIR / model_name / 'config.json'


def get_wordlist_path(wordlist_name: str) -> Path:
    return DICTIONARIES_DIR / (wordlist_name + '.txt')


def get_experiment_dir(experiment_id: str) -> Path:
    return EXPERIMENTS_DIR / experiment_id


def get_experiment_config_path(experiment_id: str) -> Path:
    return get_experiment_dir(experiment_id) / 'config.json'


def get_experiment_models_dir(experiment_id: str) -> Path:
    return get_experiment_dir(experiment_id) / 'models'


def get_experiment_model_outputs_dir(experiment_id: str) -> Path:
    return get_experiment_dir(experiment_id) / 'model_outputs'

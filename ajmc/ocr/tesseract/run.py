"""Train or evaluate tesseract"""
from ajmc.commons.miscellaneous import recursive_iterator
from ajmc.ocr.tesseract.models import get_model, train
from ajmc.ocr.config import get_all_configs
from ajmc.ocr.preprocessing import data_preparation


def pipeline(xp_config: dict):

    configs = get_all_configs()

    #

    # Check if the required models exists, build if not
    for model_id in xp_config['models']:
        model_config = configs['models'][model_id]
        _ = get_model(model_config)


    # Check whether the required datasets exist
    for dataset_id in recursive_iterator([model_config['train_dataset'], xp_config['test_dataset']]):
        dataset_config = configs['datasets'][dataset_id]
        _ = data_preparation.get_or_create_dataset_dir(dataset_config)

    # Check Whether training is required
    if model_config['train_dataset']:
        train(model_config)

    # evaluate the model











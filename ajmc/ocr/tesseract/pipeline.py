"""Train or evaluate tesseract"""
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr.tesseract.models import make_models
from ajmc.ocr.config import get_all_configs
from ajmc.ocr.preprocessing import data_preparation
import argparse

logger = get_custom_logger(__name__)

# Argument parsing
parser = argparse.ArgumentParser()

# Models
parser.add_argument('--make_models', action='store_true')
parser.add_argument('--model_ids', nargs='+', type=str,
                    help='The ids of the models to build, leave empty to build all models')

# Datasets
parser.add_argument('--make_datasets', action='store_true')
parser.add_argument('--dataset_ids', nargs='+', type=str,
                    help='The ids of the datasets to make, leave empty to make all datasets')

# Experiments
parser.add_argument('--make_experiments', action='store_true')
parser.add_argument('--experiments_ids', nargs='+', type=str,
                    help='The ids of the experiments to make, leave empty to make all experiments')

parser.add_argument('--overwrite', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    configs = get_all_configs()

    if args.make_models:
        logger.info('Making models')
        make_models(models_ids=args.model_ids if args.model_ids else None, overwrite=args.overwrite)

    if args.make_datasets:
        logger.info('Making datasets')
        data_preparation.make_datasets(dataset_ids=args.dataset_ids if args.dataset_ids else None,
                                       overwrite=args.overwrite)

    if args.make_experiments:
        logger.info('Making experiments')
        make_experiments(experiment_ids=args.experiment_ids if args.experiment_ids else None, overwrite=args.overwrite)

from pathlib import Path

from ajmc.commons.variables import PACKAGE_DIR
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import pre_batch_dataset

config_path = PACKAGE_DIR / 'configs/ocr_pytorch/1A_withbackbone_new.json'
config = get_config(config_path)

# Recreate the ancient config
config['filelists_dir'] = Path('/scratch/sven/ocr_exp/filelists')
config['datasets_root_dir'] = Path('/scratch/sven/ocr_exp/')
config['datasets_weights'] = {
    "ajmc": 20,
    "archiscribe": 2,
    "artificial_data": 1,
    "artificial_data_augmented": 1,
    "gt4histocr": 0,
    "pog": 3,
    "porta_fontium": 1
}

output_dir = Path('/scratch/sven/ocr_exp/batched_testset_delok')

pre_batch_dataset(config,
                  splits=['test'],
                  output_dir=output_dir,
                  restart=True)

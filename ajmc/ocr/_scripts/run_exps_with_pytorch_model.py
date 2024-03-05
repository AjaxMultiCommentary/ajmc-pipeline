from pathlib import Path

import torch

from ajmc.commons.file_management import walk_dirs
from ajmc.commons.miscellaneous import ROOT_LOGGER
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import TorchTrainingDataset
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('WARNING')

MODEL_DIR = Path('/scratch/sven/withbackbone_v2')
DATASETS_DIR = Path('/scratch/sven/ocr_exp/datasets')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#%%

for dataset_dir in walk_dirs(DATASETS_DIR):
    if 'test' not in dataset_dir.name:
        continue

    dataset = TorchTrainingDataset(classes_to_indices=config['classes_to_indices'],
                                   max_batch_size=64,
                                   img_height=config['chunk_height'],
                                   chunk_width=config['chunk_width'],
                                   chunk_overlap=config['chunk_overlap'],
                                   special_mapping=config.get('chars_to_special_classes', {}),
                                   data_dir=dataset_dir,
                                   loop_infinitely=False,
                                   shuffle=False)

    ocr_lines = []
    gt_lines = []
    for batch in dataset:
        ocr_lines += model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths)[0]
        gt_lines += batch.texts

    results = line_based_evaluation(gt_lines=gt_lines, ocr_lines=ocr_lines)[2]

    print(f'{dataset_dir.name}')
    print(results)
    print('------------------------------------------------')

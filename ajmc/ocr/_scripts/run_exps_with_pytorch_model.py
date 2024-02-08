from pathlib import Path

import torch

from ajmc.commons.miscellaneous import ROOT_LOGGER
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrIterDataset, OcrBatchedDataset, get_custom_dataloader
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('WARNING')
ENV = 'runai'  # 'local' or 'cluster'

if ENV == 'cluster':
    MODEL_DIR = Path('/scratch/sven/withbackbone_v2')
    DATASETS_DIR = Path('/scratch/sven/ocr_exp/datasets')

elif ENV == 'runai':
    MODEL_DIR = Path('/home/najem/dhlab-data/data/najem-data/outputs/withbackbone_v2')
    DATASETS_DIR = Path('/home/najem/dhlab-data/data/najem-data/batched_testset_delok')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#%%

if ENV == 'cluster':
    for dataset_dir in DATASETS_DIR.iterdir():
        if not dataset_dir.is_dir() or 'test' not in dataset_dir.stem:
            continue
        dataset = OcrIterDataset(classes=config['classes'],
                                 classes_to_indices=config['classes_to_indices'],
                                 max_batch_size=16,
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
            ocr_lines += model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths)
            gt_lines += batch.texts

        results = line_based_evaluation(gt_lines=gt_lines, ocr_lines=ocr_lines)[2]
        print(dataset_dir.name)
        print(results)

#%% try with the validation set
if ENV == 'cluster':
    from tqdm import tqdm

    FILELISTS_DIR = Path('/scratch/sven/ocr_exp/filelists')
    val_filelist = [Path(p) for txt in FILELISTS_DIR.glob('*_val.txt') for p in txt.read_text(encoding='utf-8').split('\n') if p]

    val_filelist = [p for p in val_filelist if p.exists()]

    dataset = OcrIterDataset(classes=config['classes'],
                             classes_to_indices=config['classes_to_indices'],
                             max_batch_size=16,
                             img_height=config['chunk_height'],
                             chunk_width=config['chunk_width'],
                             chunk_overlap=config['chunk_overlap'],
                             special_mapping=config.get('chars_to_special_classes', {}),
                             img_paths=val_filelist,
                             loop_infinitely=False,
                             shuffle=False)

    ocr_lines = []
    gt_lines = []

    for batch in tqdm(dataset):
        ocr_lines += model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths)
        gt_lines += batch.texts

    results = line_based_evaluation(gt_lines=gt_lines, ocr_lines=ocr_lines)[2]
    print(results)
# %% Try on the batched datasets on runai


if ENV == 'runai':
    from tqdm import tqdm

    for dataset_dir in DATASETS_DIR.iterdir():

        if not dataset_dir.is_dir() or dataset_dir.stem in ['train', 'val', ]:
            continue
        print('Processing', dataset_dir.name)

        dataset = OcrBatchedDataset(source_dir=dataset_dir,
                                    cache_dir=dataset_dir,
                                    num_workers=1,
                                    chars_to_special_classes=config['chars_to_special_classes'],
                                    classes_to_indices=config['classes_to_indices'])

        dataloader = get_custom_dataloader(dataset)
        ocr_lines = []
        gt_lines = []

        for batch in tqdm(dataloader):
            ocr_lines += model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths)
            gt_lines += batch.texts

        results = line_based_evaluation(gt_lines=gt_lines, ocr_lines=ocr_lines)[2]
        for gt_line, ocr_line in zip(gt_lines, ocr_lines):
            print('GT:', gt_line)
            print('OCR:', ocr_line)
            print('------------------------------------------------')

        print(results)
        print('------------------------------------------------')

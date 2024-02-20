import shutil
from pathlib import Path

import torch

from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import pre_batch_filelist, OcrIterDataset

MODEL_DIR = Path('/scratch/sven/withbackbone_v2')
DATASETS_DIR = Path('/scratch/sven/ocr_exp/testing_data/test_datasets/')

filelist = sorted((DATASETS_DIR / 'ajmc_grc_test').glob('*.png'), key=lambda p: p.stem)

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')

for bs in [8, 64]:
    config['max_batch_size'] = bs

    # Batch the dataset the ancient way
    output_dir = DATASETS_DIR / f'ajmc_grc_test_prebatched_{bs}'
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    pre_batch_filelist(filelist,
                       config,
                       output_dir=output_dir,
                       shuffle=False,
                       restart=False)

    # Batch the dataset the ocriter way
    output_dir = DATASETS_DIR / f'ajmc_grc_test_batched_ocriter_{bs}'
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    unbatched_dataset = OcrIterDataset(classes=config['classes'],
                                       classes_to_indices=config['classes_to_indices'],
                                       max_batch_size=config['max_batch_size'],
                                       img_height=config['chunk_height'],
                                       chunk_width=config['chunk_width'],
                                       chunk_overlap=config['chunk_overlap'],
                                       special_mapping=config.get('chars_to_special_classes', {}),
                                       img_paths=filelist,
                                       loop_infinitely=False,
                                       shuffle=False)

    for i, batch in enumerate(unbatched_dataset):
        torch.save(batch.to_dict(), output_dir / f'{i}.pt')

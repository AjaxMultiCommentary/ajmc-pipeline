from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ajmc.commons.miscellaneous import ROOT_LOGGER
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrIterDataset, OcrBatchedDataset, get_custom_dataloader
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('WARNING')

MODEL_DIR = Path('/scratch/sven/withbackbone_v2')
DATASETS_DIR = Path('/scratch/sven/ocr_exp/testing_data/test_datasets')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#%%

for _ in range(1):
    # Run the experiments with the unbatched dataset, with the two possible batch sizes
    dataset_dir = DATASETS_DIR / f'ajmc_grc_test'
    for bs in [8, 64]:
        config['max_batch_size'] = bs
        dataset = OcrIterDataset(classes=config['classes'],
                                 classes_to_indices=config['classes_to_indices'],
                                 max_batch_size=config['max_batch_size'],
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
        print(f'Unbatched dataset, batch size {bs}:')
        print(results)
        print('------------------------------------------------')

    for dataset_dir in DATASETS_DIR.glob('*'):
        if not dataset_dir.is_dir() or 'batched' not in dataset_dir.name:
            continue

        dataset = OcrBatchedDataset(source_dir=dataset_dir,
                                    cache_dir=dataset_dir,
                                    num_workers=1,
                                    chars_to_special_classes=config['chars_to_special_classes'],
                                    classes_to_indices=config['classes_to_indices'])

        dataset = get_custom_dataloader(dataset)

        ocr_lines = []
        gt_lines = []
        for batch in dataset:
            ocr_lines += model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths)
            gt_lines += batch.texts

        results = line_based_evaluation(gt_lines=gt_lines, ocr_lines=ocr_lines)[2]
        print(f'Batched dataset {dataset_dir.name}:')
        print(results)
        print('------------------------------------------------')

    print('------------------------------------------------')

#%%

from ajmc.ocr.pytorch.train_parallel import OcrModelTrainer

print('------------------------------------------------')
print('------------------------------------------------')
print('RUNNING WITH TRAIN PARALLEL')
import wandb

wandb.init(project='ocr_' + config['output_dir'].name,
           name=config['output_dir'].name,
           config={k: v for k, v in config.items() if 'classes' not in k},
           mode='disabled',
           resume=True)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=config['learning_rate'], weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=config['scheduler_patience'], min_lr=0.00001)

for dataset_dir in DATASETS_DIR.glob('*'):
    if not dataset_dir.is_dir() or 'batched' not in dataset_dir.name:
        continue

    dataset = OcrBatchedDataset(source_dir=dataset_dir,
                                cache_dir=dataset_dir,
                                num_workers=1,
                                chars_to_special_classes=config['chars_to_special_classes'],
                                classes_to_indices=config['classes_to_indices'])

    dataset = get_custom_dataloader(dataset)

    trainer = OcrModelTrainer(model=model,
                              train_dataloader=dataset,
                              val_dataloader=dataset,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              validation_rate=1,
                              save_rate=150000,
                              output_dir=DATASETS_DIR.parent / 'training_output_delok',
                              chunk_overlap=config['chunk_overlap'],
                              device=device,
                              )

    results = trainer.evaluate_during_training()
    print(f'Batched dataset {dataset_dir.name}:')
    print(results)
    print('------------------------------------------------')

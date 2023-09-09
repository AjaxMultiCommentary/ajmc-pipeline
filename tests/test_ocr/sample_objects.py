import random
import shutil

import torch
from torchvision.transforms import transforms

from ajmc.commons.variables import PACKAGE_DIR
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrIterDataset


def get_sample_config(mode='cpu') -> dict:
    config = get_config(PACKAGE_DIR / 'tests/test_ocr/config_test.json')
    config['train_data_dir'] = PACKAGE_DIR / 'tests/test_ocr/data'
    config['test_data_dir'] = PACKAGE_DIR / 'tests/test_ocr/data'
    config['snapshot_path'] = PACKAGE_DIR / 'tests/test_ocr/data/test_snapshot.pt'
    config['evaluation_rate'] = 2
    config['snapshot_rate'] = 2
    config['scheduler_step_rate'] = 2


    if mode == 'cpu':
        config['num_workers'] = 1
        config['device'] = 'cpu'

    if mode == 'single_gpu':
        config['num_workers'] = 1
        config['device'] = 'cuda'

    if mode == 'multi_gpu':
        config['num_workers'] = 2
        config['device'] = 'cuda'

    return config


def get_single_img_tensor(config):
    img_width = int(2.5 * config['input_shape'][2])
    return torch.tensor(list(range(config['input_shape'][1] * img_width))).reshape(1, config['input_shape'][1], img_width)


def get_batch_img_tensor(config, num_images=3):
    img_width = int(2.5 * config['input_shape'][2])
    img_height = config['input_shape'][1]
    return torch.tensor(list(range(num_images * img_width * img_height))).reshape(num_images, 1, img_height, img_width)


def get_sample_img(config):
    img_width = random.randint(int(0.5 * config['input_shape'][2]), int(3 * config['input_shape'][2]))
    img_height = config['input_shape'][1]
    img_tensor = torch.rand((1, img_height, img_width))
    return img_tensor


def get_and_write_sample_dataset(num_images,
                                 config: dict = get_sample_config()):
    # Create the directory
    shutil.rmtree(config['train_data_dir'], ignore_errors=True)
    config['train_data_dir'].mkdir(exist_ok=True, parents=True)

    # Create the images
    for i in range(num_images):
        img_tensor = get_sample_img(config)
        transforms.ToPILImage()(img_tensor).save(config['train_data_dir'] / f'{i}.png')
        (config['train_data_dir'] / f'{i}.txt').write_text(f'{i}-{i}', encoding='utf-8')

    return OcrIterDataset(data_dir=config['train_data_dir'],
                          classes=config['classes'],
                          max_batch_size=config['max_batch_size'],
                          img_height=config['input_shape'][1],
                          chunk_width=config['chunk_width'],
                          chunk_overlap=config['chunk_overlap'],
                          classes_to_indices=config['classes_to_indices'],
                          indices_to_classes=config['indices_to_classes'])


dataset = get_and_write_sample_dataset(10, get_sample_config())
#%%
from ajmc.ocr.pytorch.data_processing import get_custom_dataloader

dataloader = get_custom_dataloader(dataset, 1)

#%%
for i in range(15):
    batch = next(iter(dataloader))
    print(batch.texts)

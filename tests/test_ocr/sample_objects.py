import random
import shutil

import torch
from torchvision.transforms import transforms

from ajmc.commons.variables import PACKAGE_DIR
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import TorchTrainingDataset


def get_sample_config(mode='cpu') -> dict:
    config = get_config(PACKAGE_DIR / 'tests/test_ocr/config_test.json')
    config['train_data_dir'] = PACKAGE_DIR / 'tests/test_ocr/data'
    config['test_data_dir'] = PACKAGE_DIR / 'tests/test_ocr/data'
    config['save_to_path'] = PACKAGE_DIR / 'tests/test_ocr/data/test_snapshot.pt'
    config['validation_rate'] = 2
    config['snapshot_rate'] = 2
    config['scheduler_step_rate'] = 2

    if mode == 'cpu':
        config['device'] = 'cpu'

    if mode == 'single_gpu':
        config['device'] = 'cuda'

    if mode == 'multi_gpu':
        config['device'] = 'cuda'

    return config


def get_single_img_tensor(config):
    img_width = int(2.5 * config['chunk_width'])
    return torch.tensor(list(range(config['chunk_height'] * img_width))).reshape(1, config['chunk_height'], img_width)


def get_batch_img_tensor(config, num_images=3):
    img_width = int(2.5 * config['chunk_width'])
    img_height = config['chunk_height']
    return torch.tensor(list(range(num_images * img_width * img_height))).reshape(num_images, 1, img_height, img_width)


def get_sample_img(config):
    img_width = random.randint(int(0.5 * config['chunk_width']), int(6 * config['chunk_width']))
    img_height = config['chunk_height']
    img_tensor = torch.rand((1, img_height, img_width))
    return img_tensor


def get_and_write_sample_dataset(num_images,
                                 config: dict = get_sample_config(),
                                 loop_infinitely: bool = True,
                                 shuffle: bool = True):
    # Create the directory
    shutil.rmtree(config['train_data_dir'], ignore_errors=True)
    config['train_data_dir'].mkdir(exist_ok=True, parents=True)

    # Create the images
    for i in range(num_images):
        img_tensor = get_sample_img(config)
        transforms.ToPILImage()(img_tensor).save(config['train_data_dir'] / f'{i}.png')
        (config['train_data_dir'] / f'{i}.txt').write_text(f'{i}-{i}', encoding='utf-8')

    return TorchTrainingDataset(max_batch_size=config['max_batch_size'],
                                img_height=config['chunk_height'],
                                chunk_width=config['chunk_width'],
                                chunk_overlap=config['chunk_overlap'],
                                data_dir=config['train_data_dir'],
                                loop_infinitely=loop_infinitely,
                                shuffle=shuffle,
                                num_workers=config['num_workers'])

import argparse
import json
import random
import unicodedata
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.commons.unicode_utils import harmonise_unicode
from ajmc.commons.variables import PACKAGE_DIR
from ajmc.ocr.data_processing.data_augmentation import random_augment_line
from ajmc.ocr.pytorch.config import get_config

logger = get_ajmc_logger(__name__)
ROOT_LOGGER.setLevel('INFO')

config = get_config(PACKAGE_DIR / 'tests/test_ocr/config_test.json')
random.seed(config['random_seed'])

DO_AUGMENT = False
DATASETS_DIR = Path('/scratch/sven/ocr_exp/datasets')
SOURCE_DATASETS_DIR = Path('/scratch/sven/ocr_exp/source_datasets')
OUTPUT_DIR = Path('/scratch/sven/ocr_exp/filelists')
OUTPUT_DIR.mkdir(exist_ok=True)


def has_missing_unicode(text: str) -> bool:
    text = harmonise_unicode(text)
    text = unicodedata.normalize('NFD', text)
    return any(char not in config['classes'] for char in text)


def write_img_paths_list(imgs: List[Path], output_file: Path):
    logger.info(f'Writing list to {output_file.stem}...')
    with output_file.open('w') as f:
        f.write('\n'.join([str(p) for p in imgs]))


def get_splits(img_dir: Path, test_size: int = 150, val_size: int = 1000,
               check_unicode: bool = True,
               check_existing: bool = True) -> Dict[str, List[Path]]:
    logger.info(f'Getting splits for {img_dir.stem}...')
    # Get and filter the images
    img_paths = [fpath for fpath in img_dir.rglob('*.png')]
    if check_unicode:
        img_paths = [p for p in img_paths if not has_missing_unicode(p.with_suffix('.txt').read_text(encoding='utf-8'))]
    if check_existing:
        img_paths = clean_missing_txt_files(img_paths)

    # Shuffle
    random.shuffle(img_paths)

    # split
    test_imgs = img_paths[:test_size]
    img_paths = img_paths[test_size:]
    val_imgs = img_paths[:val_size]
    img_paths = img_paths[val_size:]

    return {'train': img_paths, 'val': val_imgs, 'test': test_imgs}


def augment(source_imgs_paths: List[Path], target_dir: Path, n: int = 2):
    target_dir.mkdir(exist_ok=True)
    for img_path in tqdm(source_imgs_paths, desc=f'Augmenting images to {target_dir}...'):
        for _ in range(n):
            random_augment_line(img_path, target_dir)


def augment_and_write(dataset_name: str,
                      dataset_dict: Dict[str, List[Path]],
                      datasets_stats: Dict[str, Dict[str, int]],
                      datasets_dir: Path = DATASETS_DIR,
                      do_augment: bool = DO_AUGMENT):
    # Augment
    augment_dir = datasets_dir / f'{dataset_name}_train_aug'
    if do_augment:
        augment(dataset_dict['train'], augment_dir, n=2)
    dataset_dict['train'] += [fpath for fpath in augment_dir.rglob('*.png')]

    # update datasets stats
    datasets_stats[dataset_name] = {a: len(b) for a, b in dataset_dict.items()}

    # Write
    for split_name, img_list in dataset_dict.items():
        write_img_paths_list(img_list, datasets_dir / f'filelists/{dataset_name}_{split_name}.txt')

    return datasets_stats


def clean_missing_txt_files(imgs_paths: List[Path]):
    new_imgs_paths = []
    for img_path in tqdm(imgs_paths, desc='Cleaning missing txt files...'):
        if not img_path.with_suffix('.txt').exists():
            img_path.unlink()
        else:
            new_imgs_paths.append(img_path)
    logger.info(f'Cleaned {len(imgs_paths) - len(new_imgs_paths)} missing txt files.')
    return new_imgs_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=[])

    args = parser.parse_args()
    datasets_stats = {}

    # AJMC
    # We need a custom soup to get the splits
    if 'ajmc' in args.datasets or 'all' in args.datasets:
        logger.info('Processing ajmc...')
        ajmc_imgs = {}
        ajmc_test_dirs = [DATASETS_DIR / x for x in ['ajmc_grc_test', 'ajmc_lat_eng_test', 'ajmc_mix_test']]
        ajmc_imgs['test'] = [fpath for test_dir in ajmc_test_dirs for fpath in test_dir.rglob('*.png')]
        # ajmc_imgs['test'] = [p for p in ajmc_imgs['test'] if not has_missing_unicode(p.with_suffix('.txt').read_text(encoding='utf-8'))]

        ajmc_imgs['train'] = [fpath for fpath in (DATASETS_DIR / 'ajmc').rglob('*.png') if fpath not in ajmc_imgs['test']]
        # ajmc_imgs['train'] = [p for p in ajmc_imgs['train'] if not has_missing_unicode(p.with_suffix('.txt').read_text(encoding='utf-8'))]

        random.shuffle(ajmc_imgs['train'])

        ajmc_imgs['val'] = ajmc_imgs['train'][:500]
        ajmc_imgs['train'] = ajmc_imgs['train'][500:]

        # Augment ajmc
        datasets_stats = augment_and_write('ajmc', ajmc_imgs, datasets_stats)

    # pog
    # We need a custom soup to get the splits
    if 'pog' in args.datasets or 'all' in args.datasets:
        logger.info('Processing pog...')
        pog_test_dir = DATASETS_DIR / 'pog_grc_test'

        pog_imgs = {}
        pog_imgs['test'] = [fpath for fpath in pog_test_dir.rglob('*.png')]
        # pog_imgs['test'] = [p for p in pog_imgs['test'] if not has_missing_unicode(p.with_suffix('.txt').read_text(encoding='utf-8'))]

        pog_imgs['train'] = [fpath for fpath in (DATASETS_DIR / 'pog').rglob('*.png') if fpath not in pog_imgs['test']]
        # pog_imgs['train'] = [p for p in pog_imgs['train'] if not has_missing_unicode(p.with_suffix('.txt').read_text(encoding='utf-8'))]

        random.shuffle(pog_imgs['train'])

        pog_imgs['val'] = pog_imgs['train'][:1000]
        pog_imgs['train'] = pog_imgs['train'][1000:]

        # Augment pog
        datasets_stats = augment_and_write('pog', pog_imgs, datasets_stats)

        # Other datasets
    for d in tqdm(['archiscribe', 'gt4histocr', 'porta_fontium'], desc='Processing other datasets'):
        if d in args.datasets or 'all' in args.datasets:
            d = SOURCE_DATASETS_DIR / d
            imgs = get_splits(d, test_size=150, val_size=800 if d.name != 'archiscribe' else 100, check_unicode=False)
            datasets_stats = augment_and_write(d.name, imgs, datasets_stats)

    for d in ['artificial_data', 'artificial_data_augmented']:
        if d in args.datasets or 'all' in args.datasets:
            d = SOURCE_DATASETS_DIR / d
            for subd in tqdm(d.iterdir(), desc=f'Processing {d.name}'):
                if subd.is_dir():
                    if subd.name == 'gibberish':
                        test_size = 20
                        val_size = 30
                    elif subd.name == 'capitals':
                        test_size = 40
                        val_size = 100
                    else:
                        test_size = 150
                        val_size = 800

                    imgs = get_splits(subd, test_size=test_size, val_size=val_size, check_unicode=False)

                    datasets_stats = augment_and_write(subd.name + 'aug' if d.name.endswith('augmented') else subd.name,
                                                       imgs, datasets_stats, do_augment=False)

    # Write stats
    (OUTPUT_DIR / 'stats.json').write_text(json.dumps(datasets_stats, indent=4, sort_keys=True), encoding='utf-8')

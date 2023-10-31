import tarfile
from pathlib import Path


SOURCE_DATASETS_DIR = Path('/scratch/sven/ocr_exp/source_datasets')


def divide_in_subdirs(source_dir: Path, files_per_subdir: int):
    img_paths = sorted(list(source_dir.glob('*.png')), key=lambda x: x.stem)

    for dir_index, file_index in enumerate(range(0, len(img_paths), files_per_subdir)):
        sub_dir = source_dir / str(dir_index)
        sub_dir.mkdir(exist_ok=True)
        for img_path in img_paths[file_index:file_index + files_per_subdir]:
            txt_path = img_path.with_suffix('.txt')
            img_path.rename(sub_dir / img_path.name)
            txt_path.rename(sub_dir / txt_path.name)


if __name__ == "__main__":
    for dataset_dir in ['artificial_data', 'artificial_data_augmented']:
        print(f'Processing {dataset_dir}')
        dataset_dir = SOURCE_DATASETS_DIR / dataset_dir
        for fonts_dir in dataset_dir.iterdir():
            if fonts_dir.is_dir():
                print(f'    Processing {fonts_dir.name}')
                divide_in_subdirs(fonts_dir, 100_000)
                for batch_dir in fonts_dir.iterdir():
                    if batch_dir.is_dir():
                        print(f'        Taring {batch_dir.name}')
                        tar_path = batch_dir.with_suffix('.tar.gz')
                        with tarfile.open(tar_path, mode='w:gz') as tar:
                            tar.add(batch_dir, arcname=batch_dir.name)

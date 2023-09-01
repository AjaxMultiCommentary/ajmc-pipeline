"""Utils for OCR data preparation and dataset manipulations."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import CONFIGS
from ajmc.ocr.preprocessing.get_source_datasets import make_clean_ajmc_dataset, make_clean_pogretra_dataset, compute_dataset_metadata

logger = get_custom_logger(__name__)

def split_root_dataset_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """This function is a custom split for ajmc and pog."""
    from simple_splitter.split import split

    # Get the metadata
    groups = metadata.groupby(['root_dataset', 'script'])

    def get_custom_split_ratio(group_name: Tuple[str, str], group_df: pd.DataFrame):
        df_len = len(group_df)
        # For ajmc, we keep a `ocr_vs.LINES_PER_TESTSET` lines per subset
        if group_name[0] == 'ajmc' and group_name[1] in ['grc', 'lat', 'mix']:
            return [('test', ocr_vs.LINES_PER_TESTSET / df_len),
                    ('train', (df_len - ocr_vs.LINES_PER_TESTSET) / df_len)]

        # For pog, we keep only ocr_vs.LINES_PER_TESTSET for the fully greek and give the whole rest to training
        elif group_name in [('pog', 'grc')]:
            return [('test', ocr_vs.LINES_PER_TESTSET / df_len),
                    ('train', (df_len - ocr_vs.LINES_PER_TESTSET) / df_len)]
        else:
            return [('train', 1.00)]

    stratification = ['work_id', 'script', 'language']

    # for ajmc, create a split column with a total x lines, statified on work_id, script
    recomposed = pd.DataFrame()
    for group_name, group_df in groups:
        group_df['split'] = split(splits=get_custom_split_ratio(group_name, group_df),
                                  stratification_columns=[group_df[s].to_list() for s in stratification])
        recomposed = pd.concat([recomposed, group_df], axis=0)

    return recomposed


def write_dataset_metadata(metadata: pd.DataFrame, dataset_dir: Path):
    """Writes the metadata to a tsv file."""
    metadata_path = ocr_vs.get_dataset_metadata_path(dataset_dir.name)
    metadata.to_csv(metadata_path, sep='\t', index=False, encoding='utf-8')


def import_dataset_metadata(dts_dir: Path) -> pd.DataFrame:
    """Imports a dataset metadata file."""
    metadata_path = ocr_vs.get_dataset_metadata_path(dts_dir.name)
    return pd.read_csv(metadata_path, sep='\t', encoding='utf-8', index_col=False)


def safe_check_dataset(dts_config: dict) -> bool:
    """Checks if a dataset is valid."""
    dts_dir = ocr_vs.get_dataset_dir(dts_config['id'])
    dts_metadata_path = ocr_vs.get_dataset_metadata_path(dts_config['id'])
    dts_config_path = ocr_vs.get_dataset_config_path(dts_config['id'])

    # First check the required files exist
    if not (dts_dir.exists() and dts_metadata_path.is_file() and dts_config_path.is_file()):
        return False

    # Then check the metadata is valid
    dts_metadata = import_dataset_metadata(dts_dir)
    if set(dts_metadata['path']) != set([str(p) for p in dts_dir.glob(f'*{ocr_vs.IMG_EXTENSION}')]) or len(
            dts_metadata) == 0:
        return False

    # Finally check the config is valid
    existing_dts_config = json.loads(dts_config_path.read_text(encoding='utf-8'))
    if existing_dts_config != dts_config:
        return False

    return True


def sample_dataset_metadata(dts_metadata: pd.DataFrame, dts_config: dict) -> pd.DataFrame:
    """Samples a dataset metadata dataframe according to the given dataset configuration.

    Args:
        dts_metadata: Dataset metadata dataframe.
        dts_config: Dataset configuration.

    Returns:
        A sampled dataset metadata dataframe.
    """

    for k, v in dts_config['sampling'].items():
        if type(v) == list:
            filter_ = dts_metadata.apply(lambda x: x[k] in v, axis=1)
            dts_metadata = dts_metadata[filter_]
        elif type(v) == str:
            filter_ = dts_metadata.apply(lambda x: x[k] == v, axis=1)
            dts_metadata = dts_metadata[filter_]
        else:
            continue

    if dts_config['sampling']['random'] is not None:
        dts_metadata = dts_metadata.sample(frac=dts_config['sampling']['random'], random_state=42)

    return dts_metadata


def gather_and_transform_dataset(dts_config: dict,
                                 dts_metadata: pd.DataFrame) -> pd.DataFrame:
    """Samples and transforms a dataset, dealing with images and their corresponding .gt.txt file.

    What this does is:
    1. Sampling the datasets by iterating only over the image contained in the metadata
    2. Transforming the images by applying the transformations specified in the config
    3. Saving the images and their corresponding .txt file in the output_dir
    4. Updating and saving the metadata in the output_dir dataset_dir
    Args:
        dts_config: A dataset config
        dts_metadata: A dataset metadata

    Returns:
        The updated dataset metadata
    """

    dts_dir = ocr_vs.get_dataset_dir(dts_config['id'])

    for src_img_path in dts_metadata['path']:
        src_img_path = Path(src_img_path)
        src_text_path = src_img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION)

        img = PILImage.open(src_img_path)
        img_id = src_img_path.stem

        if dts_config['transform']['resize'] is not None:
            img = img.resize(size=(int(dts_config['transform']['resize'] * img.width / img.height),
                                   dts_config['transform']['resize'],))
            img_id += f"_res{dts_config['transform']['resize']}"

        if dts_config['transform']['rotate'] is not None:
            img = img.rotate(angle=dts_config['transform']['rotate'], expand=True,
                             fillcolor=tuple([255] * len(img.getbands())))
            img_id += f"_rot{dts_config['transform']['rotate']}"

        if dts_config['transform']['blur'] is not None:
            img = img.filter(ImageFilter.GaussianBlur(dts_config['transform']['blur']))
            img_id += f"_blu{dts_config['transform']['blur']}"

        if dts_config['transform']['erode'] is not None:
            img = img.filter(ImageFilter.MaxFilter(dts_config['transform']['erode']))
            img_id += f"_ero{dts_config['transform']['erode']}"

        if dts_config['transform']['dilate'] is not None:
            img = img.filter(ImageFilter.MinFilter(dts_config['transform']['dilate']))
            img_id += f"_dil{dts_config['transform']['dilate']}"

        # rename image name and write it
        dst_img_path = dts_dir / (img_id + ocr_vs.IMG_EXTENSION)
        img.save(dst_img_path)

        # Deal with the corresponding .gt.txt
        dst_text_path = dts_dir / (img_id + ocr_vs.GT_TEXT_EXTENSION)
        dst_text_path.write_bytes(src_text_path.read_bytes())

        # Update the metadata
        dts_metadata.loc[dts_metadata['img_id'] == src_img_path.stem, 'img_id'] = img_id
        dts_metadata.loc[dts_metadata['img_id'] == src_img_path.stem, 'image_height'] = img.height
        dts_metadata.loc[dts_metadata['img_id'] == src_img_path.stem, 'image_width'] = img.width
        dts_metadata.loc[dts_metadata['img_id'] == src_img_path.stem, 'path'] = dst_img_path

    return dts_metadata


def make_dataset(dts_config: dict, overwrite: bool = False) -> None:
    """Returns the path to a dataset's dir, creating the dataset if it doesn't exist"""

    if safe_check_dataset(dts_config) and not overwrite:
        return None

    logger.info(f'Making dataset {dts_config["id"]}')
    dts_dir = ocr_vs.get_dataset_dir(dts_config['id'])
    dts_config_path = ocr_vs.get_dataset_config_path(dts_config['id'])
    dts_dir.mkdir(parents=True, exist_ok=True)

    if dts_config['id'] == 'ajmc':  # Special methods for root datasets
        make_clean_ajmc_dataset(output_dir=dts_dir)
        dts_metadata = compute_dataset_metadata(dts_dir=dts_dir)
        dts_metadata = split_root_dataset_metadata(dts_metadata)

    elif dts_config['id'] == 'pog':
        make_clean_pogretra_dataset(output_dir=dts_dir)
        dts_metadata = compute_dataset_metadata(dts_dir=dts_dir)
        dts_metadata = split_root_dataset_metadata(dts_metadata)

    else:
        dts_metadata = pd.DataFrame()  # Merge the sources' metadatas to get the final metadata
        for source in dts_config['source']:
            source_config = CONFIGS['datasets'][source]
            source_dir = ocr_vs.get_dataset_dir(source_config['id'])
            make_dataset(dts_config=source_config)
            dts_metadata = pd.concat([dts_metadata, import_dataset_metadata(source_dir)], axis=0)

        # transform the dataset if needed
        dts_metadata = sample_dataset_metadata(dts_metadata, dts_config)
        dts_metadata = gather_and_transform_dataset(dts_config=dts_config, dts_metadata=dts_metadata)

    # Write config and metadata
    dts_config_path.write_text(json.dumps(dts_config, indent=4), encoding='utf-8')
    write_dataset_metadata(dts_metadata, dts_dir)


def make_datasets(dts_ids: Optional[List[str]] = None, overwrite: bool = False):
    """Creates datasets.

    Args:
        dts_ids: The list of dataset ids to create. If none, creates all datasets in the config.
        overwrite: Wheter to overwrite existing datasets. Note that this function calls on `make_dataset`,
        which is recursive. If `overwrite` is True, all required datasets will be overwritten
        (i.e. also each dataset's source-dataset).
    """

    for dts_id, dts_config in tqdm(CONFIGS['datasets'].items(), desc='Making datasets...'):
        if dts_ids is None or dts_id in dts_ids:
            make_dataset(dts_config, overwrite=overwrite)

"""Utils for OCR data preparation and dataset manipulations."""

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import CONFIGS
from ajmc.ocr.utils import count_chars_by_charset, is_greek_string, is_latin_string, is_numeral_string
from ajmc.text_processing.canonical_classes import CanonicalCommentary

logger = get_custom_logger(__name__)


def get_root_dataset_id(dts_id: str) -> str:
    """Returns the root dataset (e.g. ajmc, pog...) from a line id."""
    return 'ajmc' if dts_id.split('_')[0] in vs.ALL_COMM_IDS else 'pog'


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


def compute_dataset_metadata(dts_dir: Path) -> pd.DataFrame:
    """Returns a DataFrame containing the length of each txt and its proportion of Greek characters.

    Args:
        dts_dir: Path to directory containing images and their corresponding text files.
    """

    metadata = {k: [] for k in ['work_id', 'img_id', 'root_dataset', 'path',  # Commentary work_id and id
                                'image_height', 'image_width',  # Image stats
                                'total_chars', 'total_words',
                                'grc_chars', 'lat_chars', 'num_chars',
                                'script', 'language', 'font',
                                'text']}

    for img_path in dts_dir.glob(f'*{ocr_vs.IMG_EXTENSION}'):
        work_id = img_path.stem.split('_')[0]
        metadata['work_id'].append(work_id)  # Commentary id or pogretra family id
        metadata['root_dataset'].append(get_root_dataset_id(img_path.stem))
        metadata['img_id'].append(img_path.stem)  # line image id

        # Image stats
        img = PILImage.open(img_path)
        metadata['image_height'].append(img.height)
        metadata['image_width'].append(img.width)

        # Text stats
        text = img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION).read_text(encoding='utf-8')
        metadata['total_chars'].append(len(text))
        metadata['total_words'].append(len(text.split()))
        metadata['grc_chars'].append(count_chars_by_charset(text, charset='greek'))
        metadata['lat_chars'].append(count_chars_by_charset(text, charset='latin'))
        metadata['num_chars'].append(count_chars_by_charset(text, charset='numbers'))

        script = 'grc' if is_greek_string(text, 1) else \
            'lat' if is_latin_string(text, 1) else \
                'num' if is_numeral_string(text, 1) else \
                    'mix' if (count_chars_by_charset(text, charset='greek') > 0 and
                              count_chars_by_charset(text, charset='latin') > 0) else 'unk'
        metadata['script'].append(script)

        # Language
        if script in ['grc', 'num']:
            metadata['language'].append(script[:3])
        else:
            if work_id in vs.ALL_COMM_IDS:
                if script == 'lat':
                    metadata['language'].append(vs.COMM_IDS_TO_LANG[work_id])
                elif script == 'mix':
                    metadata['language'].append('gre' + vs.COMM_IDS_TO_LANG[work_id])
                else:
                    metadata['language'].append('unk')
            else:
                metadata['language'].append('unk')

        # Font
        metadata['font'].append('unk')
        metadata['text'].append(text)
        metadata['path'].append(str(img_path))

    metadata = pd.DataFrame(metadata)

    grouped = metadata.groupby('work_id')
    metadata['initial_normalized_height'] = metadata.apply(
            lambda x: x['image_height'] / grouped.mean()['image_height'][x['work_id']], axis=1)

    return metadata


def clean_dataset(dts_dir: Union[Path, str],
                  output_dir: Union[Path, str],
                  unicode_form: str = 'NFC',
                  double_line_threshold: float = 1.8):
    """What this function does is:

    1. Exporting the OCR data in a single directory, thereby droping:
        a. non-files items (e.g. directories and symbolic links)
        b. pngs or txts with no corresponding png or txt files
        c. empty txts
    2. Removes trailing whitespaces and newlines from txts
    3. Normalizing the OCR data to a single unicode form
    4. Removes double-height lines

    Args:
        dts_dir: Path to directory containing images and their corresponding text files.
        output_dir: Path to directory where cleaned images and texts will be exported.
        unicode_form: Unicode form to normalize the OCR data to.
        double_line_threshold: Threshold for detecting double-height lines.
            Example: If set to 1.8, all lines above 1.8 times the average line height of a commentary will be removed.
    """

    dts_dir = Path(dts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_pairs, empty_txts, double_height_lines = 0, 0, 0

    # Get rid of symlinks and empty txt files, centralize all images in output_dir
    for img_path in tqdm(dts_dir.rglob(f'*{ocr_vs.IMG_EXTENSION}'),
                         desc=f'Cleaning dataset in {dts_dir.name} and exporting to {output_dir.name}'):
        if img_path.is_file():
            text_path = img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION)
            if text_path.is_file():
                text = text_path.read_text(encoding='utf-8')
                if re.sub(r'[\s+]', '', text) != '':
                    (output_dir / img_path.name).write_bytes(img_path.read_bytes())
                    # Harmonize the txt
                    text = text.strip('\n').strip()
                    text = unicodedata.normalize(unicode_form, text)
                    (output_dir / text_path.name).write_text(text, encoding='utf-8')
                else:
                    empty_txts += 1
            else:
                missing_pairs += 1

    # Remove double-height lines
    metadata = compute_dataset_metadata(output_dir)

    for img_id in metadata[metadata['initial_normalized_height'] >= double_line_threshold]['img_id']:
        (output_dir / f'{img_id}.png').unlink()
        (output_dir / f'{img_id}.gt.txt').unlink()
        double_height_lines += 1

    logger.info(
            f'Cleaning done. Removed {missing_pairs} missing pairs, {empty_txts} empty txts and {double_height_lines} double-height lines.')


def make_clean_ajmc_dataset(output_dir: Path = ocr_vs.get_dataset_dir('ajmc'),
                            comm_ids: List[str] = vs.ALL_COMM_IDS,
                            unicode_format: str = 'NFC',
                            base_dir=Path(vs.COMMS_DATA_DIR)):
    """Uses`CanonicalCommentary.export_gt_file_pairs` to export an ocr dataset for given commentary ids."""

    temp_dir = output_dir / 'temp'
    for commentary_id in tqdm(comm_ids, desc='Importing ajmc commentaries...'):
        try:
            can_path = next(((base_dir / commentary_id / 'canonical/v2').glob('*tess_base.json')))
        except StopIteration:
            continue
        commentary = CanonicalCommentary.from_json(str(can_path))
        commentary.export_ocr_gt_file_pairs(temp_dir, unicode_format=unicode_format)

    clean_dataset(temp_dir, output_dir, unicode_form=unicode_format)
    # Remove the tmpdir
    shutil.rmtree(temp_dir)


def make_clean_pogretra_dataset(output_dir: Path = ocr_vs.get_dataset_dir('pog'),
                                pogretra_source_dir: Optional[Path] = ocr_vs.POG_SOURCE_DIR,
                                url: str = 'https://zenodo.org/record/4774201/files/pogretra-v1.0.tar.gz'):
    """Creates a cleaned dataset from the Pogretra dataset.

    Args:
        output_dir: Path to directory where the cleaned dataset will be exported.
        pogretra_source_dir: Path to directory containing the Pogretra git repository. If none, the repository will be
        downloaded from `url`.
        url: URL to download the Pogretra git repository from.
    """

    # Import Pogretra
    if pogretra_source_dir is None:
        import urllib.request
        import tarfile
        logger.info(f'Downloading Pogretra from {url}...')
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, temp_dir / 'pogretra-v1.0.tar.gz')

        # Extract Pogretra
        with tarfile.open(temp_dir / 'pogretra-v1.0.tar.gz') as tar:
            tar.extractall(temp_dir)

        # Clean and move to output_dir
        clean_dataset(dts_dir=(temp_dir / 'pogretra-v1.0' / 'Data'), output_dir=output_dir)

        # Remove temp_dir
        shutil.rmtree(temp_dir)

    else:
        clean_dataset(pogretra_source_dir, output_dir)

    # remove the files which are already in ajmc
    for file_path in output_dir.glob('*'):
        if file_path.stem.split('_')[0] in vs.ALL_COMM_IDS:
            file_path.unlink()


# Todo come back here once tesseract experiments are done
# Create a function which synthesizes an ocr dataset from given text lines, image heights and fonts
def make_clean_synthetic_dataset(text_lines: Dict[str, str],
                                 image_height: int,
                                 fonts: List[str],
                                 output_dir: Path,
                                 padding: int = 0,
                                 image_suffix: str = '.png',
                                 txt_suffix: str = '.gt.txt',
                                 ) -> None:
    """Synthesizes an OCR dataset from given text lines, image heights and fonts.

    Args:
        text_lines: A mapping between ids and corresponding text lines to be used for synthesizing the dataset,
            e.g. {'sophoclesplaysa05campgoog_0012_01': 'This is a text line'}
        image_height: List of image heights to be used for synthesizing the dataset.
        fonts: List of fonts to be used for synthesizing the dataset.
        output_dir: Path to directory where the dataset will be exported.
        image_suffix: Suffix of raw images.
        txt_suffix: Suffix of raw text files.
        unicode_format: Unicode format to be used for normalizing the text.
    """

    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # for id_, text_line in tqdm(text_lines.items(), desc='Synthesizing dataset'):
    #     for font in fonts:
    #         output_file = output_dir / f'{id}_{font}_{image_height}{image_suffix}'
    #         output_file.write_text(text_line, encoding='utf-8')
    #         create_text_image(text=text_line, font_path=font, padding=padding, image_height=image_height,
    #                           output_file=output_file.with_suffix(txt_suffix))

    raise NotImplementedError


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
    3. Saving the images and their corresponding .gt.txt file in the output_dir
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

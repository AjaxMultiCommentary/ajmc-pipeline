import os
import re
import shutil
from typing import Optional, List, Union, Dict, Tuple
import cv2
import pandas as pd
from tqdm import tqdm
import unicodedata
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr.utils import count_chars_by_charset, is_greek_string, is_latin_string, is_number_string
from ajmc.commons import variables
from ajmc.ocr import variables as ocr_vars
from pathlib import Path
from PIL import Image as PILImage, ImageFilter
from ajmc.ocr.config import get_all_configs

from ajmc.text_processing.canonical_classes import CanonicalCommentary

logger = get_custom_logger(__name__)


def get_source_dataset_from_line_id(line_id: str) -> str:
    """Returns the source dataset of a line id."""
    return 'ajmc' if line_id.split('_')[0] in variables.ALL_COMMENTARY_IDS else 'pogretra'


def get_ocr_dataset_metadata(dataset_dir: Path,
                             img_suffix: str = '.png',
                             txt_suffix: str = '.gt.txt',
                             write_tsv: bool = True,
                             from_existing: bool = False) -> pd.DataFrame:
    """Returns a DataFrame containing the length of each txt and its proportion of Greek characters."""

    if from_existing:
        metadata_path = dataset_dir / 'metadata.tsv'
        if metadata_path.is_file():
            metadata = pd.read_csv(metadata_path, sep='\t', index_col=False, encoding='utf-8')
            if len(metadata) != len(list(dataset_dir.glob('*.png'))):
                logger.warning('Missmatching metadata file, re-computing it...')
                metadata = get_ocr_dataset_metadata(dataset_dir)
        else:
            metadata = get_ocr_dataset_metadata(dataset_dir)


    else:
        logger.info('Creating dataset metadata')

        metadata = {k: [] for k in ['family', 'id', 'source',  # Commentary family and id
                                    'image_height', 'image_width',  # Image stats
                                    'total_chars', 'total_words',
                                    'grc_chars', 'lat_chars', 'num_chars',
                                    'script', 'language', 'font',
                                    'text']}

        for img in dataset_dir.glob(f'*{img_suffix}'):
            if img.is_file():  # Pogretra's simlinks....
                family_id = img.stem.split('_')[0]
                metadata['family'].append(family_id)  # Commentary id or pogretra family
                metadata['source'].append(get_source_dataset_from_line_id(img.stem))
                metadata['id'].append(img.stem)  # line id

                # Image stats
                gt_image = cv2.imread(str(img))
                metadata['image_height'].append(gt_image.shape[0])
                metadata['image_width'].append(gt_image.shape[1])

                # Text stats
                gt_text = img.with_suffix(txt_suffix).read_text(encoding='utf-8')
                metadata['total_chars'].append(len(gt_text))
                metadata['total_words'].append(len(gt_text.split()))
                metadata['grc_chars'].append(count_chars_by_charset(gt_text, charset='greek'))
                metadata['lat_chars'].append(count_chars_by_charset(gt_text, charset='latin'))
                metadata['num_chars'].append(count_chars_by_charset(gt_text, charset='numbers'))

                script = 'grc' if is_greek_string(gt_text, 1) else \
                    'lat' if is_latin_string(gt_text, 1) else \
                        'num' if is_number_string(gt_text, 1) else \
                            'mix' if (count_chars_by_charset(gt_text, charset='greek') > 0 and
                                      count_chars_by_charset(gt_text, charset='latin') > 0) else 'unk'
                metadata['script'].append(script)

                # Language
                if script in ['grc', 'num']:
                    metadata['language'].append(script[:3])
                else:
                    if family_id in variables.ALL_COMMENTARY_IDS:
                        if script == 'lat':
                            metadata['language'].append(variables.COMMENTARY_IDS_TO_LANG[family_id])
                        elif script == 'mix':
                            metadata['language'].append('gre' + variables.COMMENTARY_IDS_TO_LANG[family_id])
                        else:
                            metadata['language'].append('unk')
                    else:
                        metadata['language'].append('unk')

                # Font
                metadata['font'].append('unk')

                metadata['text'].append(gt_text)

        metadata = pd.DataFrame(metadata)

        grouped = metadata.groupby('family')
        metadata['normalized_image_height'] = metadata.apply(
            lambda x: x['image_height'] / grouped.mean()['image_height'][x['family']], axis=1)

    if write_tsv:
        metadata.to_csv(os.path.join(dataset_dir, 'metadata.tsv'), sep='\t', index=False, encoding='utf-8')

    return metadata


def filter_ocr_dataset_by_text(dataset_dir: Union[Path, str],
                               target_dir: Union[Path, str],
                               filter_func: callable,
                               threshold: float = 0.5,
                               txt_suffix: str = ".gt.txt",
                               img_suffix: str = ".png"):
    """Filters dataset by applying `filter_func` to each text file in `data_dir` and exports it to `target_dir`.

    Args:
        dataset_dir: Path to directory containing images and their corresponding text files.
        target_dir: Path to directory where filtered images and texts will be exported.
        filter_func: Function that takes a text and returns True if the text should be kept, False otherwise.
        threshold: Threshold for `filter_func`.
        txt_suffix: Suffix of text files.
        img_suffix: Suffix of images.
    """

    dataset_dir = Path(dataset_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for img in tqdm(dataset_dir.rglob(f'*{img_suffix}'), desc='Filtering dataset'):
        txt = img.with_suffix(txt_suffix)
        if filter_func(txt.read_text(encoding='utf-8'), threshold):
            shutil.copy(txt, target_dir)
            shutil.copy(img, target_dir)


def filter_ocr_dataset_by_metadata(dataset_dir: Path,
                                   filter_func: callable,
                                   target_dir: Optional[Union[Path, str]] = None,
                                   inplace: bool = False,
                                   img_suffix: str = ".png",
                                   txt_suffix: str = ".gt.txt",
                                   ):
    metadata = get_ocr_dataset_metadata(dataset_dir, from_existing=True)

    metadata['keep'] = metadata.apply(lambda x: filter_func(x), axis=1)

    for line_id in tqdm(metadata[metadata['keep'] == True]['id'], desc='Filtering dataset'):
        shutil.copy(dataset_dir / f'{line_id}{img_suffix}', target_dir)
        shutil.copy(dataset_dir / f'{line_id}{txt_suffix}', target_dir)

    if inplace:
        assert target_dir is not None
        for line_id in tqdm(metadata[metadata['keep'] == False]['id'], desc='Removing lines'):
            dataset_dir.joinpath(f'{line_id}{img_suffix}').unlink()


def clean_ocr_dataset(dataset_dir: Union[Path, str],
                      output_dir: Union[Path, str],
                      unicode_form: str = 'NFC',
                      double_line_threshold: float = 1.8,
                      ) -> None:
    """What this function does is:

    1. Exporting the OCR data in a single directory, thereby droping:
        a. non-files items (e.g. directories and symbolic links)
        b. pngs or txts with no corresponding png or txt files
        c. empty txts
    2. Removes trailing whitespaces and newlines from txts
    3. Normalizing the OCR data to a single unicode form
    4. Removes double-height lines
    """

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Cleaning dataset...')

    missing_pairs, empty_txts = 0, 0

    # Get rid of simlinks and empty txt files, centralize all images in output_dir
    for png in tqdm(dataset_dir.rglob('*.png'), desc=f'Cleaning and exporting dataset to {output_dir}'):
        txt = png.with_suffix('.gt.txt')
        if png.is_file() and txt.is_file():
            if re.sub(r'[\s+]', '', txt.read_text('utf-8')) != '':
                (output_dir / png.name).write_bytes(png.read_bytes())

                # Harmonize the txt
                text = txt.read_text('utf-8').strip('\n').strip()
                text = unicodedata.normalize(unicode_form, text)
                (output_dir / txt.name).write_text(text, encoding='utf-8')
            else:
                empty_txts += 1
        else:
            missing_pairs += 1

    # Remove double-height lines
    metadata = get_ocr_dataset_metadata(output_dir, write_tsv=False, )

    for line_id in tqdm(metadata[metadata['normalized_image_height'] >= double_line_threshold]['id'],
                        desc=f'Removing double-height lines from...'):
        (output_dir / f'{line_id}.png').unlink()
        (output_dir / f'{line_id}.gt.txt').unlink()

    logger.info(f'Removed {missing_pairs} missing pairs.')
    logger.info(f'Removed {empty_txts} empty txts.')
    logger.info(
        f'Removed {len(metadata[metadata["normalized_image_height"] >= double_line_threshold])} double-height lines.')


def make_clean_ajmc_dataset(output_dir: Path,
                            comm_ids: List[str] = variables.ALL_COMMENTARY_IDS,
                            unicode_format: str = 'NFC',
                            base_dir=Path(variables.PATHS['base_dir'])):
    """Uses`CanonicalCommentary.export_gt_file_pairs` to export an ocr dataset for given commentary ids."""

    temp_dir = output_dir / 'temp'
    for commentary_id in tqdm(comm_ids, desc='Instantiating commentaries'):
        try:
            can_path = next(((base_dir / commentary_id / 'canonical/v2').glob('*tess_base.json')))
        except StopIteration:
            continue
        commentary = CanonicalCommentary.from_json(str(can_path))
        commentary.export_ocr_gt_file_pairs(temp_dir, unicode_format=unicode_format)

    clean_ocr_dataset(temp_dir, output_dir, unicode_form=unicode_format)
    # Remove the tmpdir
    shutil.rmtree(temp_dir)

    # Write metadata
    custom_split_source_dataset(output_dir, write_tsv=True)


def make_clean_pogretra_dataset(output_dir: Path,
                                pogretra_source_dir: Optional[Path] = None,
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
        logger.info(f'Downloading Pogretra from {url}...')
        import urllib.request
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, temp_dir / 'pogretra-v1.0.tar.gz')

        # Extract Pogretra
        import tarfile
        with tarfile.open(temp_dir / 'pogretra-v1.0.tar.gz') as tar:
            tar.extractall(temp_dir)

        # Clean and move to output_dir
        clean_ocr_dataset((temp_dir / 'pogretra-v1.0' / 'Data'),
                          output_dir)

        # Remove temp_dir
        shutil.rmtree(temp_dir)
        try:
            temp_dir.rmdir()
        except:
            pass

    else:
        clean_ocr_dataset(pogretra_source_dir, output_dir)

    # remove the files which are already in ajmc
    for file_path in output_dir.glob('*'):
        if file_path.stem.split('_')[0] in variables.ALL_COMMENTARY_IDS:
            file_path.unlink()

    # Rewrite metadata
    get_ocr_dataset_metadata(output_dir, from_existing=False)

    custom_split_source_dataset(output_dir, write_tsv=True)


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


def custom_split_source_dataset(dataset_dir: Path,
                                write_tsv: bool = True):
    """This function creates a custom split for the dataset and re-exports the metadata."""
    from simple_splitter.split import split

    # Get the metadata
    metadata = get_ocr_dataset_metadata(dataset_dir, from_existing=True)
    groups = metadata.groupby(['source', 'script'])

    def get_custom_split_ratio(group_name: Tuple[str, str], group_df: pd.DataFrame):
        df_len = len(group_df)
        # For ajmc, we keep a hundred lines per subset
        if group_name[0] == 'ajmc' and group_name[1] in ['grc', 'lat', 'mix']:
            return [('test', 100 / df_len),
                    ('train', (df_len - 100) / df_len)]

        # For pog, we keep only 50 for the fully greek and give the whole rest to training
        elif group_name in [('pogretra', 'grc')]:
            return [('test', 100 / df_len), ('train', (df_len - 100) / df_len)]
        else:
            return [('train', 1.00)]

    stratification = ['family', 'script', 'language']

    # for ajmc, create a split column with a total x lines, statified on family, script
    recomposed = pd.DataFrame()
    for group_name, group_df in groups:
        group_df['split'] = split(splits=get_custom_split_ratio(group_name, group_df),
                                  stratification_columns=[group_df[s].to_list() for s in stratification])
        recomposed = pd.concat([recomposed, group_df], axis=0)

    if write_tsv:
        recomposed.to_csv((dataset_dir / 'metadata.tsv'), sep='\t', index=False)


def sample_metadata(metadata, config):
    # sample metadata
    for k, v in config['sampling'].items():
        if type(v) == list:
            filter_ = metadata.apply(lambda x: x[k] in v, axis=1)
            metadata = metadata[filter_]
        elif type(v) == str:
            filter_ = metadata.apply(lambda x: x[k] == v, axis=1)
            metadata = metadata[filter_]
        else:
            continue

    if config['sampling']['random'] is not None:
        metadata = metadata.sample(frac=config['sampling']['random'], random_state=42)

    return metadata


def transform_ocr_dataset(config: dict,
                          dataset_dir: Optional[Path] = None,
                          file_list: List[Path] = None,
                          output_dir: Optional[Path] = None):
    """Does transform operation on a dataset or a file list, doing it inplace if `output_dir` is not given.

    Deals with the corresponding .gt.txt files.

    Args:
        dataset_dir: `Path` to the dataset's directory.
        config: A dataset config
        file_list: A `List[Path]` to the image files to be changed.
        output_dir: `Path` to the directory to which to export the new dataset, leaving the source untouched
    """

    assert dataset_dir is not None or file_list is not None, """`dataset_dir` or `file_list` must be given."""

    img_paths_generator = dataset_dir.glob('.png') if file_list is None else file_list

    for img_path in img_paths_generator:
        # import image
        img = PILImage.open(img_path)
        # rotate image
        img_id = img_path.stem

        transform_config = config['transform']

        if transform_config['resize'] is not None:
            img = img.resize(size=(int(transform_config['resize'] * img.width / img.height), transform_config['resize'],))
            img_id += f'_res{transform_config["resize"]}'

        if transform_config['rotate'] is not None:
            img = img.rotate(angle=transform_config['rotate'], expand=True, fillcolor=(255, 255, 255))
            img_id += f'_rot{transform_config["rotate"]}'

        if transform_config['blur'] is not None:
            img = img.filter(ImageFilter.GaussianBlur(transform_config['blur']))
            img_id += f'_blu{transform_config["blur"]}'

        if transform_config['erode'] is not None:
            img = img.filter(ImageFilter.MaxFilter(transform_config['erode']))
            img_id += f'_ero{transform_config["erode"]}'

        if transform_config['dilate'] is not None:
            img = img.filter(ImageFilter.MinFilter(transform_config['dilate']))
            img_id += f'_dil{transform_config["dilate"]}'

        # rename image name
        new_path = (img_path.parent if not output_dir else output_dir) / (img_id + '.png')
        # delete old image
        if not output_dir:
            img_path.unlink()
        # write image
        img.save(new_path)

        # Deal with the corresponding .gt.txt
        txt_path = img_path.with_suffix('.gt.txt')
        if output_dir is not None: # Create a new txt
            new_path.with_suffix('.gt.txt').write_text(txt_path.read_text(encoding='utf-8'), encoding='utf-8')
        else:  # renames it
            txt_path.rename(new_path.with_suffix('.gt.txt'))


def get_or_create_dataset_dir(dataset_config: dict,
                              overwrite: bool= False) -> Path:
    """Returns the path to a dataset's dir, creating the dataset if it doesn't exist"""

    dataset_dir = ocr_vars.get_dataset_dir(dataset_config['id'])
    all_dataset_configs: dict = get_all_configs()['datasets']

    if dataset_dir.is_dir() and not overwrite: # If the dataset already exists
        return dataset_dir

    # else
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Special methods for ajmc and pogretra source datasets
    if dataset_config['id'] == 'ajmc':
        make_clean_ajmc_dataset(dataset_dir)

    elif dataset_config['id'] == 'pogretra':
        make_clean_pogretra_dataset(dataset_dir)

    else:
        # Merge the sources' metadatas
        metadata = pd.DataFrame()
        for source in dataset_config['source']:
            source_config = all_dataset_configs[source]
            source_dir = get_or_create_dataset_dir(source_config)
            source_metadata = get_ocr_dataset_metadata(source_dir, from_existing=True)
            source_metadata['path'] = source_metadata['id'].apply(lambda x: (source_dir / (x + '.png')))
            metadata = pd.concat([metadata, source_metadata], axis=1)

        # Sample
        metadata = sample_metadata(metadata=metadata, config=dataset_config)

        # transform and copies the images to new datasets
        transform_ocr_dataset(config=dataset_config,
                              file_list=metadata['path'].tolist(),
                              output_dir=dataset_dir)

        # Write metadata
        metadata.drop('path', inplace=True, axis=1)
        metadata.to_csv((dataset_dir / 'metadata.tsv'), sep='\t', index=False)

    return dataset_dir


def create_all_datasets(overwrite: bool = False):
    configs = get_all_configs()

    for dataset_config in configs['datasets'].values():
        get_or_create_dataset_dir(dataset_config, overwrite=overwrite)





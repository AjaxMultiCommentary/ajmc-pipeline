import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional, Dict, Union

import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.commons.unicode_utils import count_chars_by_charset, is_charset_string
from ajmc.ocr import variables as ocr_vs
from ajmc.text_processing.canonical_classes import CanonicalCommentary

logger = get_ajmc_logger(__name__)


def controle_overwrite(output_dir: Path, overwrite: bool):
    """Checks if the output directory exists and if it does, deletes it if overwrite is True."""

    if output_dir.exists():
        if not overwrite:
            logger.warning(f'{output_dir.name} already exists. Skipping.')
            return
        else:
            logger.warning(f'{output_dir.name} already exists. Deleting it.')
            shutil.rmtree(output_dir)


def basic_text_cleaning(text: str, unicode_form: str = ocr_vs.UNICODE_FORM) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = unicodedata.normalize(unicode_form, text)
    return text


def regularize_dataset(dts_dir: Path):
    """Simple python wrapper around GT4HistOCR regularize.pl"""

    logger.info(f'Regularizing dataset in {dts_dir.name}...')

    command = f"perl {ocr_vs.SOURCE_DATASETS_DIR / 'tools' / 'regularize.pl'} {dts_dir}"
    os.system(command)

    for f in dts_dir.glob('*.txt'):
        f.unlink()

    for f in dts_dir.glob('*.txt.reg'):
        f.rename(f.parent / f'{f.stem.split(".")[0]}.txt')

    logger.info(f'Done regularizing dataset in {dts_dir.name}.')


def compute_dataset_metadata(dts_dir: Path,
                             root_dataset_name: str,
                             compute_initial_normalized_height: bool = True) -> pd.DataFrame:
    """Returns a DataFrame containing the length of each txt and its proportion of Greek characters.

    Args:
        dts_dir: Path to directory containing images and their corresponding text files.
    """

    logger.info(f'Computing metadata for dataset {root_dataset_name}...')

    metadata = {k: [] for k in ['work_id', 'img_id', 'root_dataset', 'path',  # Commentary work_id and id
                                'image_height', 'image_width',  # Image stats
                                'total_chars', 'total_words',
                                'grc_chars', 'lat_chars', 'num_chars',
                                'script', 'language', 'font',
                                'text']}

    for img_path in tqdm(dts_dir.glob(f'*{ocr_vs.IMG_EXTENSION}'), desc='Computing metadata...'):
        work_id = img_path.stem.split('_')[0]
        metadata['work_id'].append(work_id)  # Commentary id or pogretra family id
        metadata['root_dataset'].append(root_dataset_name)  # Dataset id
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
        metadata['num_chars'].append(count_chars_by_charset(text, charset='numeral'))

        script = 'grc' if is_charset_string(text, charset='greek', threshold=1, strict=False) else \
            'lat' if is_charset_string(text, charset='latin', threshold=1, strict=False) else \
                'num' if is_charset_string(text, charset='numeral', threshold=1) else \
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

    if compute_initial_normalized_height:
        grouped = metadata.groupby('work_id')
        metadata['initial_normalized_height'] = metadata.apply(
                lambda x: x['image_height'] / grouped.mean()['image_height'][x['work_id']], axis=1)
    else:
        metadata['initial_normalized_height'] = 1

    return metadata


def clean_source_dataset(dts_dir: Union[Path, str],
                         output_dir: Union[Path, str],
                         root_dataset_name: str,
                         unicode_form: str = 'NFC',
                         img_extension: str = ocr_vs.IMG_EXTENSION,
                         txt_extension: str = ocr_vs.GT_TEXT_EXTENSION,
                         double_line_threshold: float = -1,
                         rename_with_parents_dir=False):
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
        root_dataset_name: Name of the dataset.
        unicode_form: Unicode form to normalize the OCR data to.
        img_extension: Extension of the image files.
        txt_extension: Extension of the text files.
        double_line_threshold: Threshold for detecting double-height lines. If set to -1, no double-height lines will be removed.
            Example: If set to 1.8, all lines above 1.8 times the average line height of a work will be removed.
    """

    dts_dir = Path(dts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_pairs, empty_txts, double_height_lines = 0, 0, 0

    # Get rid of symlinks and empty txt files, centralize all images in output_dir
    for img_path in tqdm(dts_dir.rglob(f'*{img_extension}'),
                         desc=f'Cleaning dataset in {dts_dir.name} and exporting to {output_dir.name}'):
        if img_path.is_file():
            img_id = img_path.stem.split('.')[0]
            text_path = img_path.parent / (img_id + txt_extension)
            if text_path.is_file():
                text = text_path.read_text(encoding='utf-8')
                if re.sub(r'\s+', '', text) != '':

                    if rename_with_parents_dir:
                        out_img_path = output_dir / (img_path.parent.name + '_' + img_id + ocr_vs.IMG_EXTENSION)
                        out_txt_path = output_dir / (img_path.parent.name + '_' + img_id + ocr_vs.GT_TEXT_EXTENSION)
                    else:
                        out_img_path = output_dir / (img_id + ocr_vs.IMG_EXTENSION)
                        out_txt_path = output_dir / (img_id + ocr_vs.GT_TEXT_EXTENSION)

                    out_img_path.write_bytes(img_path.read_bytes())
                    text = basic_text_cleaning(text, unicode_form=unicode_form)
                    out_txt_path.write_text(text, encoding='utf-8')

                else:
                    empty_txts += 1
            else:
                missing_pairs += 1
                print(f'No text file found for image {img_path}.')

    # Remove double-height lines
    metadata = compute_dataset_metadata(output_dir, root_dataset_name, compute_initial_normalized_height=double_line_threshold > 0)

    if double_line_threshold > 0:

        for img_id in metadata[metadata['initial_normalized_height'] >= double_line_threshold]['img_id']:
            (output_dir / f'{img_id}{ocr_vs.IMG_EXTENSION}').unlink()
            (output_dir / f'{img_id}{ocr_vs.GT_TEXT_EXTENSION}').unlink()
            double_height_lines += 1

    logger.info(
            f'Cleaning done. Removed {missing_pairs} missing pairs, {empty_txts} empty txts and {double_height_lines} double-height lines.')


def make_clean_ajmc_dataset(output_dir: Path = ocr_vs.get_dataset_dir('ajmc'),
                            comm_ids: List[str] = vs.ALL_COMM_IDS,
                            unicode_form: str = ocr_vs.UNICODE_FORM,
                            base_dir=Path(vs.COMMS_DATA_DIR),
                            overwrite=False):
    """Uses``CanonicalCommentary.export_gt_file_pairs`` to export an ocr dataset for given commentary ids."""

    controle_overwrite(output_dir, overwrite=overwrite)

    temp_dir = output_dir / 'temp'
    for commentary_id in tqdm(comm_ids, desc='Importing ajmc commentaries...'):
        try:
            can_path = next(((base_dir / commentary_id / 'canonical/v2').glob('*tess_base.json')))
        except StopIteration:
            continue
        commentary = CanonicalCommentary.from_json(str(can_path))
        commentary.export_ocr_gt_file_pairs(temp_dir, unicode_format=unicode_form)

    clean_source_dataset(temp_dir, output_dir,
                         root_dataset_name='ajmc',
                         unicode_form=unicode_form, img_extension=ocr_vs.IMG_EXTENSION,
                         txt_extension=ocr_vs.GT_TEXT_EXTENSION, double_line_threshold=1.8, )

    # Remove the tmpdir
    shutil.rmtree(temp_dir)


def make_clean_archiscribe_dataset(output_dir: Path = ocr_vs.get_source_dataset_dir('archiscribe'),
                                   overwrite: bool = False):
    controle_overwrite(output_dir, overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    clone_command = f'cd {str(output_dir)} ; git clone https://github.com/jbaiter/archiscribe-corpus'
    os.system(clone_command)

    clean_source_dataset(dts_dir=(output_dir / 'archiscribe-corpus' / 'transcriptions'),
                         output_dir=output_dir,
                         root_dataset_name='archiscribe',
                         unicode_form=ocr_vs.UNICODE_FORM,
                         img_extension='.png',
                         txt_extension='.txt',
                         double_line_threshold=-1)

    shutil.rmtree(output_dir / 'archiscribe-corpus')


def make_clean_gt4histocr_dataset(output_dir: Path = ocr_vs.get_source_dataset_dir('gt4histocr'),
                                  overwrite: bool = False):
    controle_overwrite(output_dir, overwrite)

    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    download_command = f'cd {str(temp_dir)} ; wget https://zenodo.org/record/1344132/files/GT4HistOCR.tar; tar -xvf GT4HistOCR.tar'
    if (ocr_vs.SOURCE_DATASETS_DIR / 'GT4HistOCR.tar').exists():
        download_command = f'cd {str(temp_dir)} ; tar -xvf {str(ocr_vs.SOURCE_DATASETS_DIR / "GT4HistOCR.tar")}'
    os.system(download_command)

    # Unzip all archives
    for archive_path in tqdm((temp_dir / 'corpus').glob('*.tar.bz2'), desc='Unzipping nested archives...'):
        command = f'cd {str(archive_path.parent)} ; tar -xf {str(archive_path.name)}'
        os.system(command)

    # Clean dataset
    clean_source_dataset(dts_dir=temp_dir / 'corpus',
                         output_dir=output_dir,
                         root_dataset_name='gt4histocr',
                         unicode_form=ocr_vs.UNICODE_FORM,
                         img_extension='.png',
                         txt_extension='.gt.txt',
                         double_line_threshold=-1,
                         rename_with_parents_dir=True)

    shutil.rmtree(temp_dir)

    regularize_dataset(output_dir)


def make_clean_porta_fontium_dataset(output_dir: Path = ocr_vs.get_source_dataset_dir('porta_fontium'),
                                     overwrite: bool = False):
    controle_overwrite(output_dir, overwrite)

    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    if (ocr_vs.SOURCE_DATASETS_DIR / 'hist_ge_ocr_corpus_tols.tar.xz').exists():
        download_command = f'cd {str(temp_dir)} ; tar -xf {str(ocr_vs.SOURCE_DATASETS_DIR / "hist_ge_ocr_corpus_tols.tar.xz")}'
        logger.info(f'Found local tar at {str(ocr_vs.SOURCE_DATASETS_DIR / "hist_ge_ocr_corpus_tols.tar.xz")}')
    else:
        raise NotImplementedError('This dataset must be downloaded manually from http://ocr-corpus.kiv.zcu.cz/')
    os.system(download_command)

    logger.info('Unzipping annotated corpus...')
    unzip_command = f'cd {str(temp_dir)}; mkdir corpus_temp; cd corpus_temp ; tar -xf {str(temp_dir / "annotated_corpus.tar.xz")}'
    os.system(unzip_command)

    clean_source_dataset(dts_dir=temp_dir / 'corpus_temp',
                         output_dir=output_dir,
                         root_dataset_name='porta_fontium',
                         unicode_form=ocr_vs.UNICODE_FORM,
                         img_extension='.png',
                         txt_extension='.txt',
                         double_line_threshold=-1,
                         rename_with_parents_dir=False)

    # Remove the tmpdir
    shutil.rmtree(temp_dir)


def make_clean_pogretra_dataset(output_dir: Path = ocr_vs.get_dataset_dir('pog'),
                                pogretra_source_dir: Optional[Path] = ocr_vs.POG_SOURCE_DIR,
                                url: str = 'https://zenodo.org/record/4774201/files/pogretra-v1.0.tar.gz',
                                overwrite: bool = False):
    """Creates a cleaned dataset from the Pogretra dataset.

    Args:
        output_dir: Path to directory where the cleaned dataset will be exported.
        pogretra_source_dir: Path to directory containing the Pogretra git repository. If none, the repository will be
        downloaded from ``url``.
        url: URL to download the Pogretra git repository from.
    """

    controle_overwrite(output_dir, overwrite)

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
        clean_source_dataset(dts_dir=(temp_dir / 'pogretra-v1.0' / 'Data'), output_dir=output_dir)

        # Remove temp_dir
        shutil.rmtree(temp_dir)

    else:
        clean_source_dataset(pogretra_source_dir,
                             output_dir, root_dataset_name='pog',
                             unicode_form=ocr_vs.UNICODE_FORM,
                             img_extension='.png',
                             txt_extension='.txt',
                             double_line_threshold=1.8, )

    # remove the files which are already in ajmc
    for file_path in output_dir.glob('*'):
        if file_path.stem.split('_')[0] in vs.ALL_COMM_IDS:
            file_path.unlink()


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
        unicode_form: Unicode format to be used for normalizing the text.
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


def make_clean_source_datasets(overwrite: bool = False):
    """Creates the source datasets from the raw data.

    Args:
        overwrite: If True, the source datasets will be recreated even if they already exist.
    """

    # Create the source datasets
    make_clean_archiscribe_dataset(overwrite=overwrite)
    make_clean_ajmc_dataset(overwrite=overwrite)
    make_clean_gt4histocr_dataset(overwrite=overwrite)
    make_clean_pogretra_dataset(overwrite=overwrite)
    make_clean_porta_fontium_dataset(overwrite=overwrite)

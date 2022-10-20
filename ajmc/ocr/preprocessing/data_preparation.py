import glob
import os
import shutil
from typing import Optional

import cv2
import pandas as pd
from tqdm import tqdm

from ajmc.commons.file_management.utils import walk_files
from ajmc.commons.image import resize_image
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr.evaluation.utils import count_chars_by_charset


def is_greek(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of alphabet chars in text are Greek, False otherwise."""
    alphanum_text = "".join([c for c in text if c.isalpha()])  # cleaning the text from non-alphabetical characters
    if alphanum_text:
        proportion_greek_chars = count_chars_by_charset(string=alphanum_text, charset='greek') / len(alphanum_text)
        return proportion_greek_chars >= threshold
    else:
        return False


def is_latin(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of alphabet chars in text are Greek, False otherwise."""
    alphanum_text = "".join([c for c in text if c.isalpha()])  # cleaning the text from non-alphabetical characters
    if alphanum_text:
        proportion_latin_chars = count_chars_by_charset(string=alphanum_text, charset='latin') / len(alphanum_text)
        return proportion_latin_chars >= threshold
    else:
        return False


def get_ocr_dataset_text_stats(dataset_dir: str,
                               txt_suffix: str = ".gt.txt",
                               is_greek_thresh: float = 1,
                               is_latin_thresh: float = 1) -> pd.DataFrame:
    """Returns a DataFrame containing the length of each txt and its proportion of Greek characters."""

    stats = {k: [] for k in ['file_name', 'total_chars', 'greek_chars', 'latin_chars', 'is_greek', 'is_latin']}

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(txt_suffix) and not os.path.islink(os.path.join(root, file)):  # Pogretra's simlinks....
                with open(os.path.join(root, file), 'r') as f:
                    gt_text = f.read()
                stats['file_name'].append(file)
                stats['total_chars'].append(len(gt_text))
                stats['greek_chars'].append(count_chars_by_charset(gt_text, charset='greek'))
                stats['latin_chars'].append(count_chars_by_charset(gt_text, charset='latin'))
                stats['is_greek'].append(is_greek(gt_text, is_greek_thresh))
                stats['is_latin'].append(is_latin(gt_text, is_latin_thresh))

    df = pd.DataFrame(stats)
    print(df.describe())

    return df


def filter_ocr_dataset(dataset_dir: str,
                       target_dir: str,
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

    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in tqdm(os.walk(dataset_dir)):
        for file in files:
            if file.endswith(txt_suffix) and not os.path.islink(os.path.join(root, file)):  # Pogretra's simlinks....
                txt_name = file
                txt_path = os.path.join(root, file)
                img_name = txt_name.replace(txt_suffix, img_suffix)

                if img_name in files and not os.path.islink(os.path.join(root, img_name)):
                    with open(txt_path, 'r') as f:
                        gt_text = f.read()

                    if filter_func(gt_text, threshold=threshold):
                        shutil.copyfile(txt_path, os.path.join(target_dir, txt_name))
                        shutil.copyfile(os.path.join(root, img_name), os.path.join(target_dir, img_name))


logger = get_custom_logger(__name__)


def resize_ocr_dataset(dataset_dir: str,
                       output_dir: str,
                       target_height: int,
                       image_suffix: str = ".png",
                       txt_suffix: str = ".gt.txt",
                       return_stats: bool = False) -> Optional[pd.DataFrame]:
    """Resize OCR dataset to `target_height` and exports it to `output_dir`.

    Args:
        dataset_dir: Path to directory containing raw images.
        output_dir: Path to directory where resized images will be exported.
        target_height: Target height of resized images.
        image_suffix: Suffix of raw images.
        txt_suffix: Suffix of raw text files.
        return_stats: If True, returns a DataFrame mapping each image to its height.
      """

    height_map = {}

    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(walk_files(dataset_dir, filter=lambda x: x.suffix == image_suffix, recursive=True),
                         desc=f"Resizing images in {dataset_dir} to {target_height}"):

        # Process the image
        img = cv2.imread(str(img_path))
        height_map[img_path.name] = img.shape[0]
        resized_img = resize_image(img, target_height)

        # Get the corresponding txt file
        txt_path = img_path.with_suffix(txt_suffix)

        # Save the resized image and txt
        cv2.imwrite(os.path.join(output_dir, img_path.name), resized_img)
        shutil.copyfile(txt_path, os.path.join(output_dir, txt_path.name))

    # Print the stats
    stats = pd.DataFrame(data={'heights':list(height_map.values())}, index=list(height_map.keys()))
    print(f"Dataset successfully resized to target height of {target_height} and exported to `{output_dir}`. \n"
          f"Stats of initial heights are shown below.")
    print(stats.describe())

    if return_stats:
        return stats



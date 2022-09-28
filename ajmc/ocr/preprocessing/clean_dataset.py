import glob
import os
import shutil
import pandas as pd
from tqdm import tqdm
from ajmc.ocr.evaluation.utils import count_chars_by_charset


def is_greek(text: str, threshold: float = 0.5) -> bool:
    """Returns True if more than `threshold` of alphabet chars in text are Greek, False otherwise."""
    alphanum_text = "".join([c for c in text if c.isalpha()])  # cleaning the text from non-alphabetical characters
    if alphanum_text:
        proportion_greek_chars = count_chars_by_charset(string=alphanum_text, charset='greek') / len(alphanum_text)
        return proportion_greek_chars > threshold
    else:
        return False


def get_ocr_dataset_text_stats(dataset_dir: str, txt_suffix: str = ".gt.txt") -> pd.DataFrame:
    """Returns a DataFrame containing the length of each txt and its proportion of Greek characters."""

    stats = {k: [] for k in ['file_name', 'total_chars', 'greek_chars', 'greek_proportion']}

    for txt_path in glob.glob(os.path.join(dataset_dir, '*' + txt_suffix)):
        txt_name = os.path.basename(txt_path)

        with open(txt_path, 'r') as f:
            gt_text = f.read()

        stats['file_name'].append(txt_name)
        stats['total_chars'].append(len(gt_text))
        stats['greek_chars'].append(count_chars_by_charset(string=gt_text, charset='greek'))
        stats['greek_proportion'].append(count_chars_by_charset(string=gt_text, charset='greek') / len(gt_text))

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

    for txt_path in tqdm(glob.glob(os.path.join(dataset_dir, '*' + txt_suffix)), desc='Filtering OCR dataset'):
        # Get file names
        txt_name = os.path.basename(txt_path)
        img_name = txt_name.replace(txt_suffix, img_suffix)

        with open(txt_path, 'r') as f:
            gt_text = f.read()

        if filter_func(gt_text, threshold=threshold):
            shutil.copyfile(txt_path, os.path.join(target_dir, txt_name))
            shutil.copyfile(os.path.join(dataset_dir, img_name), os.path.join(target_dir, img_name))

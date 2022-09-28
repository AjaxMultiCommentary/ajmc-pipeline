import glob
import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
from typing import Optional

from ajmc.commons.image import resize_image
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)


def resize_ocr_dataset(raw_imgs_dir: str,
                       resized_imgs_dir: str,
                       target_height: int,
                       image_suffix: str = ".png",
                       txt_suffix: str = ".gt.txt",
                       return_stats: bool = False) -> Optional[pd.DataFrame]:
    """Resize OCR dataset to `target_height` and exports it to `resized_imgs_dir`.

    Args:
        raw_imgs_dir: Path to directory containing raw images.
        resized_imgs_dir: Path to directory where resized images will be exported.
        target_height: Target height of resized images.
        image_suffix: Suffix of raw images.
        txt_suffix: Suffix of raw text files.
        return_stats: If True, returns a DataFrame mapping each image to its height.
      """

    height_map = {}

    os.makedirs(resized_imgs_dir, exist_ok=True)

    for img_path in tqdm(glob.glob(os.path.join(raw_imgs_dir, f'*{image_suffix}')),
                         desc=f"Resizing images in {raw_imgs_dir} to {target_height}"):

        # Process the image
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        height_map[img_path] = img.shape[0]
        resized_img = resize_image(img, target_height)

        # Get the corresponding txt file
        txt_path = os.path.join(raw_imgs_dir, img_name.replace(image_suffix, txt_suffix))

        # Save the resized image and txt
        cv2.imwrite(os.path.join(resized_imgs_dir, img_name), resized_img)
        shutil.copyfile(txt_path, os.path.join(resized_imgs_dir, os.path.basename(txt_path)))

    # Print the stats
    stats = pd.DataFrame(data={'heights':list(height_map.values())}, index=list(height_map.keys()))
    print(f"Dataset successfully resized to target height of {target_height} and exported to `{resized_imgs_dir}`. \n"
          f"Stats of initial heights are shown below.")
    print(stats.describe())

    if return_stats:
        return stats

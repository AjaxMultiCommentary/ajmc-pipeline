import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(PROJECT_DIR))

from ajmc.ocr.run.run_tesseract import PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS
from ajmc.ocr.run.run_tesseract import get_commentary_dir

def merge_dataset(clean_suffix="clean-threshold", img_suffix=".png"):
    clean_folder = get_commentary_dir(clean_suffix, "pogretra", cleaned_suffix="merge", create_if_missing=True)

    for subdir_name in POGRETRA_COMMENTARY_DIRS:
        print(subdir_name)
        folder = get_commentary_dir(subdir_name, "pogretra", cleaned_suffix=clean_suffix, create_if_missing=False)
        

        for filename in tqdm(list(os.listdir(folder))):
            if not filename.endswith(".gt.txt"):
                continue
            img_filename = filename.replace(".gt.txt", img_suffix)
            img_file = os.path.join(folder, img_filename)
            if not os.path.isfile(img_file):
                continue
            output_file = os.path.join(clean_folder, img_filename)
            shutil.copyfile(img_file, output_file)

            gt_file = os.path.join(folder, filename)
            shutil.copyfile(gt_file, os.path.join(clean_folder, filename))

for resize in [20,30,40,50,60,70]:
    merge_dataset(clean_suffix=f"clean-threshold-1.0-resize-{resize}", img_suffix=".png")
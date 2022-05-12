import os
import sys
import cv2
import shutil
import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(PROJECT_DIR))

from ajmc.ocr.run.run_tesseract import PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS
from ajmc.ocr.run.run_tesseract import get_commentary_dir

def print_height_stats(h_map):
    height_list = list(h_map.values())
    max_height = np.max(height_list)
    min_height = np.min(height_list)
    print(min_height, max_height, np.quantile(height_list, [0.25, 0.5, 0.75]))
    max_filenames = [k for k in h_map if h_map[k] == max_height]
    min_filenames = [k for k in h_map if h_map[k] == min_height]
    print(len(max_filenames))
    if len(max_filenames) <= 20:
        print(max_filenames)
    print(len(min_filenames))
    if len(min_filenames) <= 20:
        print(min_filenames)

def resize_dataset(clean_suffix="clean-threshold", img_suffix=".png", target_height=36):
    height_map = {}
    height_map_resize = {}
    try: 
        for subdir_name in POGRETRA_COMMENTARY_DIRS:
            print(subdir_name)
            folder = get_commentary_dir(subdir_name, "pogretra", cleaned_suffix=clean_suffix, create_if_missing=False)
            clean_folder = get_commentary_dir(subdir_name, "pogretra", cleaned_suffix=f"{clean_suffix}-resize-{target_height}", create_if_missing=True)

            for filename in tqdm(list(os.listdir(folder))):
                if not filename.endswith(".gt.txt"):
                    continue
                img_filename = filename.replace(".gt.txt", img_suffix)
                img_file = os.path.join(folder, img_filename)
                if not os.path.isfile(img_file):
                    continue
                img = cv2.imread(img_file)
                dimensions = img.shape
                height = dimensions[0]
                height_map[img_file] = height

                scale_percent = target_height / img.shape[0] # percent of original size
                width = int(img.shape[1] * scale_percent)
                height = target_height
                dim = (width, height)
    
                # resize image
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

                output_file = os.path.join(clean_folder, img_filename)
                cv2.imwrite(output_file, resized)

                gt_file = os.path.join(folder, filename)
                shutil.copyfile(gt_file, os.path.join(clean_folder, filename))

                height_map_resize[output_file] = resized.shape[0]

    except KeyboardInterrupt as e:
        print("Code is interrupted by user.")

    finally:
        print_height_stats(height_map)
        print_height_stats(height_map_resize)

resize_dataset(clean_suffix="clean-threshold", img_suffix=".png", target_height=20)
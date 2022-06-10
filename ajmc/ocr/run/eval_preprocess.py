import os
import cv2
import sys
from matplotlib import pyplot as plt
PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(PROJECT_DIR))

from ajmc.ocr.run.run_tesseract import PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS, POGRETRA_DATA_DIR
from ajmc.ocr.run.run_tesseract import get_fig_name, test_ocr, train, get_fig_idxs, show_fig, batch_ocr, train, clean_data_tesstrain, check_missing_gt, clean_gt_folder, test_ocr_raw, evaluate_model, check_dataset_size, get_commentary_dir
from ajmc.ocr.preprocess import toolbox

mode="pogretra"

custom_ckpts = [
    # "grc")
    # "finetune-grc-pogretra-clean-googletrans-resize-20-v3",
    # "finetune-grc-pogretra-clean-threshold-1.0-resize-70-epoch4-valid0.05-lr0.0001-new",
    # "re-pogretra-clean-threshold-1.0-resize-70-epoch5-valid0.05-lr0.002-new",
    "baseline"
]

for custom_ckpt in custom_ckpts:
    custom_lang = {
        "cu31924087948174": [
            # replace grc with our ckpt
            # custom_ckpt,
            # f"{custom_ckpt}+eng",
            # f"{custom_ckpt}+eng+GT4HistOCR_50000000.997_191951",
            # append our ckpt
            # f"grc+eng+{custom_ckpt}",
            # f"grc+eng+GT4HistOCR_50000000.997_191951+{custom_ckpt}",
            "grc+eng+GT4HistOCR_50000000.997_191951"
        ],
        "sophokle1v3soph": [
            # replace grc with our ckpt
            # custom_ckpt,
            # f"{custom_ckpt}+deu",
            # f"{custom_ckpt}+deu+GT4HistOCR_50000000.997_191951",
            # append our ckpt
            # f"grc+deu+{custom_ckpt}",
            # f"grc+deu+GT4HistOCR_50000000.997_191951+{custom_ckpt}",
            "grc+deu+GT4HistOCR_50000000.997_191951"
        ],
    }
    evaluate_model(["cu31924087948174", "sophokle1v3soph"], 
        TESSDATA_BEST_DIR, 
        custom_ckpt,
        custom_lang=custom_lang,
        test_preprocess=True)
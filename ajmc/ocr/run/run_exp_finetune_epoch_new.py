import os
import cv2
import sys
from matplotlib import pyplot as plt
PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(PROJECT_DIR))

from ajmc.ocr.run.run_tesseract import PARENT_DIR, RAW_COMMENTARY_DIRS, TRAIN_COMMENTARY_DIRS, TESSDATA_DIR, TESSDATA_BEST_DIR, POGRETRA_COMMENTARY_DIRS, POGRETRA_DATA_DIR
from ajmc.ocr.run.run_tesseract import get_fig_name, test_ocr, train, get_fig_idxs, show_fig, batch_ocr, train, clean_data_tesstrain, check_missing_gt, clean_gt_folder, test_ocr_raw, evaluate_model, check_dataset_size, get_commentary_dir
from ajmc.ocr.preprocessing import image_preprocessing

mode="pogretra"

for ite in [5,6,7,8,9,10]:
    for valid_ratio in [0.05]:
        print(f"Current ite: {ite}, valid_ratio: {valid_ratio}")
        finetune_suffix = "clean-threshold-1.0-resize-70"
        train(
            f"fi-pogretra-{finetune_suffix}-epoch{ite}-valid{valid_ratio}-lr0.0001-new", 
            [f"{finetune_suffix}-merge"], 
            mode, 
            TESSDATA_BEST_DIR, 
            config_file=os.path.join(PROJECT_DIR, "ocr", "tesstrain_configs", f"finetune_epoch{ite}_valid{valid_ratio}_lr0.0001.yaml"), 
            cleaned_suffix="")

        custom_ckpt = f"fi-pogretra-{finetune_suffix}-epoch{ite}-valid{valid_ratio}-lr0.0001-new"

        custom_lang = {
            "cu31924087948174": [
                # replace grc with our ckpt
                custom_ckpt,
                f"{custom_ckpt}+eng",
                f"{custom_ckpt}+GT4HistOCR_50000000.997_191951",
                f"{custom_ckpt}+eng+GT4HistOCR_50000000.997_191951",
                # append our ckpt
                f"grc+{custom_ckpt}",
                f"grc+eng+{custom_ckpt}",
                f"grc+GT4HistOCR_50000000.997_191951+{custom_ckpt}",
                f"grc+eng+GT4HistOCR_50000000.997_191951+{custom_ckpt}"
            ],
            "sophokle1v3soph": [
                # replace grc with our ckpt
                custom_ckpt,
                f"{custom_ckpt}+deu",
                f"{custom_ckpt}+GT4HistOCR_50000000.997_191951",
                f"{custom_ckpt}+deu+GT4HistOCR_50000000.997_191951",
                # append our ckpt
                f"grc+{custom_ckpt}",
                f"grc+deu+{custom_ckpt}",
                f"grc+GT4HistOCR_50000000.997_191951+{custom_ckpt}",
                f"grc+deu+GT4HistOCR_50000000.997_191951+{custom_ckpt}"
            ],
        }
        evaluate_model(["cu31924087948174", "sophokle1v3soph"], 
            TESSDATA_BEST_DIR, 
            custom_ckpt,
            custom_lang=custom_lang)


from pathlib import Path

from ajmc.commons import variables as vs
from ajmc.commons.file_management import get_62_based_datecode
from ajmc.ocr.tesseract import models
from ajmc.ocr.variables import COMM_IDS_TO_TESS_LANGS

for comm_id in vs.ALL_COMM_IDS:
    tess_langs = COMM_IDS_TO_TESS_LANGS[comm_id].split('+')
    tess_langs = ['eng_eng-mr-eng_ajmc_lat_eng_train_aug_8' if l == 'eng' else 'grc_grc-br_ajmc-pog_grc_train_aug_40' if l == 'grc' else l for l in
                  tess_langs]
    tess_langs = '+'.join(tess_langs)

    output_dir = vs.get_comm_ocr_runs_dir(comm_id) / (get_62_based_datecode() + '_tess_retrained') / 'outputs'

    models.run(img_dir=vs.get_comm_img_dir(comm_id),
               output_dir=output_dir,
               langs=tess_langs,
               config={'tessedit_create_hocr': '1'},
               tessdata_prefix=Path('/scratch/sven/ocr_exp/custom_tessdata'),
               )

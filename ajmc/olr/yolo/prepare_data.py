import os

import yaml
from ajmc.text_processing import ocr_classes, canonical_classes
from ajmc.commons import variables
import json

base_dir = '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentaries/commentaries_data'
# You get the commentary to canonical
commentary_id = 'cu31924087948174'
ocr_run = None
canonical_path = os.path.join(base_dir, commentary_id, variables.PATHS['canonical'], ocr_run)

try:
    with open(canonical_path, 'r') as f:
        comm = canonical_classes.CanonicalCommentary.from_json(json_path=canonical_path)

except FileNotFoundError:
    ocr_dir = os.path.join(base_dir, commentary_id, variables.PATHS['ocr'], ocr_run, 'outputs')
    comm = ocr_classes.OcrCommentary.from_ajmc_structure(ocr_dir=ocr_dir).to_canonical()
    comm.to_json(canonical_path)






#%%
# you get the splits
# You output the regions


yolo_yaml = {
    'path': '/scratch/sven/',
    'train': 'images/train',
    'val': 'images/val',
    'nc': None,  # todo ⚠️ : find nc
    'names': None, # todo ⚠️ : find nc
}

with open('config.yaml', 'w') as f:
    docs = yaml.dump(yolo_yaml, f)






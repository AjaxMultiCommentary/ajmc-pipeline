import os

from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.text_processing.canonical_classes import CanonicalCommentary


can = OcrCommentary.from_ajmc_structure('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/sophokle1v3soph/ocr/runs/13p0bP_lace_retrained/outputs').to_canonical()
can.to_json()
#%%
import os

base = '/Users/sven/packages/ajmc/data/yolo/datasets/multiclass'
for conf_dir in next(os.walk(base))[1]:
    config_path = os.path.join(base, conf_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        conf = f.read()
    conf = conf.replace('path: ../datasets/', 'path: ../datasets/multiclass/')
    with open(config_path, 'w') as f:
        f.write(conf)
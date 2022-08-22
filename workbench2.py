import os

from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.text_processing.canonical_classes import CanonicalCommentary


can = OcrCommentary.from_ajmc_structure('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/sophokle1v3soph/ocr/runs/13p0bP_lace_retrained/outputs').to_canonical()
can.to_json()

rom = CanonicalCommentary.from_json('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/17g08V_kraken.json')

page = [p for p in rom.children['page'] if p.id == 'DeRomilly1976_0081'][0]

'82' in page.text
#%%
import os

base = '/Users/sven/packages/ajmc/data/yolo/datasets/binary_class'

for conf_dir in next(os.walk(base))[1]:
    config_path = os.path.join(base, conf_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        conf = f.read()
    conf = conf.replace('names:\n- others\n- commentary\n- primary_text\n- paratext\n- numbers\n- app_crit\n- O', 'names:\n- region')
    conf = conf.replace('nc: 7', 'nc: 1')
    with open(config_path, 'w') as f:
        f.write(conf)
import json
import os
from ajmc.text_processing import ocr_classes
from ajmc.commons import variables
import json
import os
from ajmc.text_processing import ocr_classes
from ajmc.commons import variables

configs_dir = 'data/layoutlm/configs'
yolo_configs_dir = '/data/configs_/configs_'
output_dir = 'data/layoutlm/configs_'

for fname in os.listdir(configs_dir):
    if fname.endswith('.json'):
        with open(os.path.join(configs_dir, fname), "r") as file:
            config = json.loads(file.read())




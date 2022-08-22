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

        temp = {}
        for set_, path_sets in config['data_dirs_and_sets'].items():
            temp[set_] = []
            for path, splits in config['data_dirs_and_sets'][set_].items():
                for split in splits:
                    dict_ = {'id': path.split('/')[0],
                             'run': path.split('/')[-1][:-5],
                             'split': split}
                    temp[set_].append(dict_)

        config['data'] = temp
        del config['data_dirs_and_sets']
        config['excluded_region_types'] = ['line_number_commentary', 'handwritten_marginalia', 'undefined', 'line_region']

        with open(os.path.join(output_dir, fname), "w") as outfile:
            json.dump(config, outfile, indent=4, ensure_ascii=False, sort_keys=True)



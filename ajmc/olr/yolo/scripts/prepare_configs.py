import json
import os
from ajmc.text_processing import ocr_classes
from ajmc.commons import variables

configs_dir = '/Users/sven/packages/ajmc/data/configs/olr/layoutlm'

for fname in os.listdir(configs_dir):
    if fname.endswith('.json'):
        with open(os.path.join(configs_dir, fname), "r") as file:
            config = json.loads(file.read())

        for set_, path_sets in config['data_dirs_and_sets'].items():
            for path in config['data_dirs_and_sets'][set_].keys():
                path = path.strip('/')
                comm_id = path.split('/')[0]
                ocr_run = path.split('/')[-2]

                path = os.path.join(variables.PATHS['base_dir'], path)

                comm = ocr_classes.OcrCommentary.from_ajmc_structure(path)

                can_path = os.path.join(variables.PATHS['base_dir'], comm_id, 'canonical/v2', ocr_run + '.json')

                if not os.path.exists(can_path):
                    can = comm.to_canonical()
                    can.to_json(can_path)

# %%
import json
import os
from ajmc.text_processing import ocr_classes
from ajmc.commons import variables

configs_dir = '/Users/sven/packages/ajmc/data/configs/olr/layoutlm'
yolo_configs_dir = '/data/yolo/yolo'

for fname in os.listdir(configs_dir):
    if fname.endswith('.json'):
        with open(os.path.join(configs_dir, fname), "r") as file:
            config = json.loads(file.read())

        temp = {}
        for set_, path_sets in config['data_dirs_and_sets'].items():
            temp[set_] = {}
            for path, splits in config['data_dirs_and_sets'][set_].items():
                path = path.strip('/')
                comm_id = path.split('/')[0]
                ocr_run = path.split('/')[-2]

                path = os.path.join(variables.PATHS['base_dir'], path)

                can_path = os.path.join(comm_id, 'canonical/v2', ocr_run + '.json')
                temp[set_][can_path] = splits

        config['data_dirs_and_sets'] = temp

        with open(os.path.join(yolo_configs_dir, fname), "w") as outfile:
            json.dump(config, outfile, indent=4, ensure_ascii=False, sort_keys=True)

# %%
from ajmc.text_processing import ocr_classes
import logging

logging.basicConfig(level=logging.ERROR)

paths = [
    'sophoclesplaysa05campgoog/ocr/runs/15o09Y_lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27/outputs',
    "Kamerbeek1953/ocr/runs/17u09o_kraken/outputs",
    "sophoclesplaysa05campgoog/ocr/runs/15o0dN_lace_retrained_sophoclesplaysa05campgoog-2021-05-24-08-15-12-porson-with-sophoclesplaysa05campgoog-2021-05-23-22-17-38/outputs",
    "sophoclesplaysa05campgoog/ocr/runs/15o09Y_lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27/outputs",
    "sophoclesplaysa05campgoog/ocr/runs/15o09Y_lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27/outputs",
    "Paduano1982/ocr/runs/17v0fZ_kraken/outputs",
    "Wecklein1894/ocr/runs/13p0am_lace_base/outputs"
]

base_dir =  '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data'
for path in path:
    print(f'PROCESSING {path}')
    comm_id = path.split('/')[0]
    ocr_run = path.split('/')[-2]
    path = os.path.join(variables.PATHS['base_dir'], path)
    can_path = os.path.join(comm_id, 'canonical/v2', ocr_run + '.json')

    comm = ocr_classes.OcrCommentary.from_ajmc_structure(path).to_canonical()

    comm.to_json(can_path)



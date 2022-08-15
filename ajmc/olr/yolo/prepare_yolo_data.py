import os
import random

import numpy as np

from ajmc.commons.image import Image
from ajmc.olr.layout_lm.config import rois, regions_to_coarse_labels, coarse_labels_to_ids, ids_to_coarse_labels
import yaml
from ajmc.olr.utils import get_olr_split_page_ids
from ajmc.text_processing import canonical_classes
import json


base_data_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data'
base_xp_dir = '/Users/sven/packages/ajmc/data/yolo'
configs_dir = os.path.join(base_xp_dir, 'configs')

excluded_configs = ['1C_jebb_blank_tokens.json']

for config_name in os.listdir(configs_dir):
    if config_name.endswith('.json') and config_name not in excluded_configs:
        print(f'******** Processing {config_name} *********')
        with open(os.path.join(configs_dir, config_name), "r") as file:
            config = json.loads(file.read())

        config_dir = os.path.join(base_xp_dir, 'datasets/binary_class', config_name[:-5])
        # Create folders
        abs_paths = {'images': {'train': os.path.join(config_dir, 'images/train'),
                                'eval': os.path.join(config_dir, 'images/eval')},
                     'labels': {'train': os.path.join(config_dir, 'labels/train'),
                                'eval': os.path.join(config_dir, 'labels/eval')}}
        for k, d in abs_paths.items():
            for k_, path in d.items():
                os.makedirs(path, exist_ok=True)

        # Write yaml dataset config :
        yolo_yaml = {
            'path': f'../datasets/{config_name[:-5]}',
            'train': 'images/train',
            'val': 'images/eval',
            'nc': len(coarse_labels_to_ids.keys()),
            'names': [it[0] for it in sorted([it_ for it_ in coarse_labels_to_ids.items()], key=lambda x: x[1])]
        }

        with open(os.path.join(config_dir, 'config.yaml'), 'w') as file:
            documents = yaml.dump(yolo_yaml, file)


        for set_ in config['data_dirs_and_sets'].keys():
            for path, splits in config['data_dirs_and_sets'][set_].items():

                # You get the commentary to canonical
                print(f'import {path}')
                comm = canonical_classes.CanonicalCommentary.from_json(
                    os.path.join(base_data_dir, path)
                )

                p_ids = get_olr_split_page_ids(comm.id, splits)
                pages = [p for p in comm.children['page'] if p.id in p_ids]

                try:
                    random.seed(42)
                    pages = random.sample(pages, k=int(config['sampling'][set_]*len(pages)))
                except KeyError:
                    pass

                for p in pages :
                    # write page image
                    img_name = p.image.path.split('/')[-1]
                    p.image.write(os.path.join(abs_paths['images'][set_], img_name))
                    # get page labels
                    yolo_labels = []
                    for r in p.children['region']:
                        if r.info['region_type'] in rois:
                            r_coarse_label = regions_to_coarse_labels[r.info['region_type']]
                            r_label_id = coarse_labels_to_ids[r_coarse_label]
                            r_label_id = 0
                            r_width = r.bbox.width / p.image.width
                            r_height = r.bbox.height/ p.image.height
                            r_center_x = r.bbox.center[0]/p.image.width
                            r_center_y = r.bbox.center[1]/p.image.height
                            yolo_labels.append(f'{r_label_id} {r_center_x} {r_center_y} {r_width} {r_height}')

                    # write page labesl
                    with open(os.path.join(abs_paths['labels'][set_], p.image.id + '.txt'), 'w') as f:
                        f.write('\n'.join(yolo_labels))


        # todo 👁️ add noisy pages ?
        # write blank images
        num_blank_pages = 2
        for i in range(num_blank_pages):
            blank_img = Image(matrix=np.ones(p.image.matrix.shape))
            blank_img.write(os.path.join(abs_paths['images']['train'], f'blank_{i}.png'))
            # write page labels
            with open(os.path.join(abs_paths['labels']['train'], f'blank_{i}.txt'), 'w') as f:
                f.write('')




import json
import os

import pandas as pd

from ajmc.commons.geometry import is_bbox_within_bbox_with_threshold
from ajmc.nlp.token_classification.evaluation import seqeval_evaluation, seqeval_to_df
from ajmc.olr.layoutlm.layoutlm import create_olr_config
from ajmc.olr.yolo.utils import parse_yolo_txt_line
from ajmc.text_processing.canonical_classes import CanonicalCommentary

WORD_INCLUSION_THRESH: float = 0.9
base_data_dir = '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentary/commentaries_data'
yolo_path = '/scratch/sven/yolo'
runs_path = os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/train')

results = pd.DataFrame()
commentaries = {}

for xp_name in next(os.walk(runs_path))[1]:
    xp_path = os.path.join(runs_path, xp_name)
    print(f'Processing {xp_name}')

    gt_words_labels = []
    pred_words_labels = []

    # Open config file to get commentary
    config_path = os.path.join(yolo_path, 'configs', xp_name + '.json')
    config = create_olr_config(config_path, prefix=base_data_dir)
    with open(config_path, 'r') as file:
        config = json.loads(file.read())

    # Create the commentary
    for dict_ in config['data']['eval']:
        if not dict_['id'] in commentaries.keys():
            commentaries[dict_['id']] = CanonicalCommentary.from_json(dict_['path'])

    # Find the prediction
    preds_dir = os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/detect', xp_name, 'labels')

    # For each prediction .txt file
    regions = []
    for txt in os.listdir(preds_dir):
        if txt.endswith('.txt'):
            txt_path = os.path.join(preds_dir, txt)
            comm_id = txt.split('_')[0]
            page_id = txt.replace('.txt', '')
            page = [p for p in commentaries[comm_id].children.pages if p.id == page_id][0]

            with open(txt_path, 'r') as f:
                lines = f.read().split('\n')
                lines = [l for l in lines if l]  # drop empty lines

            print('   Line loop')
            for line in lines:
                regions.append(parse_yolo_txt_line(line=line,
                                                   ids_to_label=config['ids_to_labels'],
                                                   image_width=page.image.width,
                                                   image_height=page.image.height,
                                                   is_groundtruth=False))

            print('   Word loop')
            for r in page.children.regions:
                for w in r.children.words:
                    # find the word's gt label
                    gt_rtype = r.info['region_type']
                    gt_rtype = gt_rtype if gt_rtype != 'line_region' else 'other'
                    gt_label = config['region_types_to_labels'][gt_rtype]
                    gt_words_labels.append(gt_label)

                    # find the word's pred label
                    pred_regions = [r for r in regions
                                    if is_bbox_within_bbox_with_threshold(w.bbox.bbox,
                                                                          r['bbox'].bbox,
                                                                          WORD_INCLUSION_THRESH)]
                    if pred_regions:
                        pred_rtype = pred_regions[0]['label']
                    else:
                        pred_rtype = 'O'
                    pred_words_labels.append(pred_rtype)

    result = seqeval_evaluation(predictions=[pred_words_labels],
                                groundtruth=[gt_words_labels],
                                nerify_labels=True)

    result = seqeval_to_df(result)
    result.insert(0, ('config', 'exp'), [xp_name])
    results = pd.concat([results, result], axis=0)

results.to_csv(path_or_buf=os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/detect/results.tsv'), sep='\t')

import json
import os
import pandas as pd
from ajmc.commons.geometry import Shape, is_rectangle_within_rectangle_with_threshold
from ajmc.nlp.token_classification.evaluation import seqeval_evaluation, seqeval_to_df
from ajmc.olr.layout_lm.config import rois, regions_to_coarse_labels, coarse_labels_to_ids, ids_to_coarse_labels, ids_to_ner_labels

from ajmc.text_processing.canonical_classes import CanonicalCommentary, CanonicalSinglePageTextContainer

WORD_INCLUSION_THRESH: float = 0.9
base_data_dir = '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentaries/commentaries_data'
yolo_path = '/scratch/sven/yolo'
runs_path = os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/train')

results = pd.DataFrame()
comms = {}

for xp_name in os.listdir(runs_path):
    xp_path = os.path.join(runs_path, xp_name)
    if os.path.isdir(xp_path):
        print(f'Processing {xp_name}')
        gt_words_labels = []
        pred_words_labels = []

        # Open config file to get commentary
        config_path = os.path.join(yolo_path, 'configs', xp_name + '.json')
        with open(config_path, 'r') as file:
            config = json.loads(file.read())

        # Create the commentaries
        for can_path in config['data_dirs_and_sets']['eval'].keys():
            comm_id_ = can_path.split('/')[-4]
            if not comm_id_ in comms.keys():
                can_path = os.path.join(base_data_dir, can_path)
                print(f'    Loading {can_path}')
                comm = CanonicalCommentary.from_json(can_path)
                print(f'    Loaded {can_path}')
                comms[comm.id] = comm

        # Find the prediction
        preds_dir = os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/detect', xp_name, 'labels')

        # For each prediction .txt file
        regions = []
        for txt in os.listdir(preds_dir):
            if txt.endswith('.txt'):
                txt_path = os.path.join(preds_dir, txt)
                comm_id = txt.split('_')[0]
                page_id = txt.replace('.txt', '')
                page = [p for p in comms[comm_id].children['page'] if p.id == page_id][0]

                with open(txt_path, 'r') as f:
                    lines = f.read().split('\n')
                    lines = [l for l in lines if l]  # drop empty lines

                print('   Line loop')
                for line in lines:

                    label, center_x, center_y, width, height = line.split(' ')
                    label = int(label)
                    label = ids_to_ner_labels[label]

                    center_x = float(center_x)*page.image.width
                    center_y = float(center_y)*page.image.height
                    width = float(width)*page.image.width
                    height = float(height)*page.image.height
                    region = {'label': label,
                              'bbox': Shape.from_center_w_h(center_x, center_y, width, height)}
                    regions.append(region)

                print('   Word loop')
                for r in page.children['region']:
                    for w in r.children['word']:
                        # find the word's gt label
                        gt_rtype = r.info['region_type']
                        gt_rtype = gt_rtype if gt_rtype != 'line_region' else 'other'
                        gt_ner_label = ids_to_ner_labels[coarse_labels_to_ids[regions_to_coarse_labels[gt_rtype]]]
                        gt_words_labels.append(gt_ner_label)

                        # find the word's pred label
                        pred_regions = [r for r in regions
                                        if is_rectangle_within_rectangle_with_threshold(w.bbox.bbox,
                                                                                        r['bbox'].bbox,
                                                                                        WORD_INCLUSION_THRESH)]
                        if pred_regions:
                            pred_rtype = pred_regions[0]['label']
                        else:
                            pred_rtype = 'O'
                        pred_words_labels.append(pred_rtype)

        result = seqeval_evaluation(predictions=[pred_words_labels],
                                    groundtruth=[gt_words_labels])

        result = seqeval_to_df(result)
        result.insert(0, ('config', 'exp'), [xp_name])
        results = pd.concat([results, result], axis=0)


results.to_csv(path_or_buf=os.path.join(yolo_path, 'runs/yolov5m_1280_ep300/detect/results.tsv'), sep='\t')
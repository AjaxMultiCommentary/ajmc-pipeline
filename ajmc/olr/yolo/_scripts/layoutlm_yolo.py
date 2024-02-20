import os
from collections import Counter

import numpy as np
import pandas as pd
from mean_average_precision import MetricBuilder
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast, \
    RobertaForTokenClassification, RobertaTokenizerFast

from ajmc.commons.file_management import walk_dirs
from ajmc.commons.geometry import is_bbox_within_bbox_with_threshold
from ajmc.olr.evaluation import initialize_general_results, update_general_results
from ajmc.olr.layoutlm.layoutlm import align_predicted_page, create_olr_config
from ajmc.olr.utils import get_olr_splits_page_ids
from ajmc.olr.yolo.utils import parse_yolo_txt
# Constants
from ajmc.text_processing.canonical_classes import CanonicalCommentary

BASE_DATA_DIR = '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentary/commentaries_data'
LAYOUTLM_XP_DIR = '/scratch/sven/layoutlm/experiments'
YOLO_XP_DIR = '/scratch/sven/yolo/runs/binary_class'
WORD_INCLUSION_THRESHOLD = 0.6

LAYOUTLM_TO_YOLO_XP = {
    '0A_jebb_base': '0A_jebb_base',
    '0B_kamerbeek_base': '0B_kamerbeek_base',
    '1A_jebb_half_trainset': '1A_jebb_half_trainset',
    '1C_jebb_blank_tokens': '0A_jebb_base',
    '1D_jebb_kamerbeek': '1D_jebb_kamerbeek',
    '1E_jebb_text_only': '0A_jebb_base',
    '2A_campbell_transfer_jebb': '2A_campbell_transfer_jebb',
    '2B_kamerbeek_transfer_jebb': '2B_kamerbeek_transfer_jebb',
    '2C_garvie_transfer_jebb': '2C_garvie_transfer_jebb',
    '3A_paduano_base': '3A_paduano_base',
    '3B_wecklein_base': '3B_wecklein_base',
    '4A_omnibus_base': '4A_omnibus_base',
    '4B_omnibus_external': '4B_omnibus_external',
    '4C_omnibus_transfer':'4C_omnibus_transfer'
}


# Read the experiments config
for i, xp_name in enumerate(walk_dirs(LAYOUTLM_XP_DIR)):

    # Create the config
    config = create_olr_config(os.path.join(LAYOUTLM_XP_DIR, xp_name, 'config.json'), prefix=PATHS['cluster_base_dir'])

    # Initialize mAP computation
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=len(config['ids_to_labels']))

    # Initialize general_results
    if i == 0:
        general_results = initialize_general_results(config['ids_to_labels'])

    # Create the LayoutLM model and its tokenizer
    if not config['text_only']:
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(config['model_name_or_path'])
        model = LayoutLMv3ForTokenClassification.from_pretrained(os.path.join(LAYOUTLM_XP_DIR, xp_name, 'model'))
        feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(config['model_name_or_path'], apply_ocr=False)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name_or_path'], add_prefix_space=True)
        model = RobertaForTokenClassification.from_pretrained(os.path.join(LAYOUTLM_XP_DIR, xp_name, 'model'))
        feature_extractor = None

    # Retrieve the eval pages
    pages = []
    for dict_ in config['data']['eval']:
        commentary = CanonicalCommentary.from_json(os.path.join(PATHS['cluster_base_dir'], dict_['id'],
                                                                PATHS['canonical'], dict_['run'] + '.json'))
        page_ids = get_olr_splits_page_ids(commentary.id, [dict_['split']])
        pages += [p for p in commentary.children.pages
                  if p.id in page_ids]

    # Get the predictions
    for page in pages:
        words, labels = align_predicted_page(page=page,
                                             labels_to_ids=config['labels_to_ids'],
                                             ids_to_labels=config['ids_to_labels'],
                                             rois=config['rois'],
                                             regions_to_coarse_labels=config['region_types_to_labels'],
                                             tokenizer=tokenizer,
                                             feature_extractor=feature_extractor,
                                             unknownify_tokens=config['unknownify_tokens'],
                                             model=model,
                                             text_only=config['text_only'])

        for w, l in zip(words, labels):
            w.layout_lm_label = l

        # get YOLO's predictions
        yolo_preds_path = os.path.join(YOLO_XP_DIR, LAYOUTLM_TO_YOLO_XP[xp_name], 'detect/labels')

        try:
            txt_name = [p for p in os.listdir(yolo_preds_path) if p.startswith(page.id)][0]
            txt_path = os.path.join(yolo_preds_path, txt_name)
            pred_regions = parse_yolo_txt(path=txt_path,
                                          ids_to_label={0: 'region'},
                                          image_width=page.image.width,
                                          image_height=page.image.height,
                                          is_groundtruth=False)
        except IndexError:
            pred_regions = []

        # find the words in each region
        for r in pred_regions:
            r['words'] = [w for w in words
                          if is_bbox_within_bbox_with_threshold(contained=w.bbox.bbox,
                                                                container=r['bbox'].bbox,
                                                                threshold=WORD_INCLUSION_THRESHOLD)]
        # Delete empty regions:
        pred_regions = [r for r in pred_regions if r['words']]

        # Get each region's label
        for r in pred_regions:
            r_labels = [w.layout_lm_label for w in r['words']]
            r['label'] = max(Counter(r_labels))

        # todo 👁️ resize regions

        # Do the evaluation using mAP
        # Get the preds
        preds = []
        for r in pred_regions:
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            r_pred = r['bbox'].xyxy + [config['labels_to_ids'][r['label']]] + [1]
            preds.append(r_pred)
        preds_array = np.array(preds)

        # get the gt
        groundtruth = []
        for r in page.children.regions:
            # [xmin, ymin, xmax, ymax, class_id, confidence, difficult, crowd]
            r_gt = r.bbox.xyxy + \
                   [config['labels_to_ids'][config['region_types_to_labels'][r.info['region_type']]]] + \
                   [0, 0]
            groundtruth.append(r_gt)
        gt_array = np.array(groundtruth)

        metric_fn.add(preds_array, gt_array)


    metrics = metric_fn.value(iou_thresholds=(0.5))

    # Do the counts
    for k, dict_ in metrics[0.5].items():
        dict_['count'] = len([r for p in pages
                              for r in p.children.regions
                              if config['labels_to_ids'][config['region_types_to_labels'][r.info['region_type']]] == k])

    general_results = update_general_results(general_results=general_results,
                                             metrics=metrics, xp_name=xp_name,
                                             ids_to_labels=config['ids_to_labels'])

    df = pd.DataFrame.from_dict(general_results)

tsv_path = os.path.join(LAYOUTLM_XP_DIR, 'general_results_with_yolo_binary.tsv')
df.to_csv(tsv_path, sep='\t', index=False)
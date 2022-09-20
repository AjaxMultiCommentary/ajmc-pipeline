import os

import numpy as np
import pandas as pd
import PIL
from mean_average_precision import MetricBuilder

from ajmc.commons.geometry import Shape, is_bbox_within_bbox
from ajmc.commons.miscellaneous import walk_dirs, get_olr_splits_page_ids
from ajmc.olr.layout_lm.draw import draw_page_labels, draw_caption
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor, \
    RobertaTokenizerFast, RobertaForTokenClassification
from ajmc.olr.layout_lm.config import create_olr_config
from ajmc.olr.layout_lm.layoutlm import align_predicted_page
from ajmc.commons.variables import PATHS, COLORS
from ajmc.olr.map_utils import metrics_abbrev, initialize_general_results, update_general_results

RUNS_DIR = '/scratch/sven/layoutlm/experiments'

results = pd.DataFrame()

# Walk over experiments
for i, xp_name in enumerate(walk_dirs(RUNS_DIR)):
    # Create the config
    config = create_olr_config(os.path.join(RUNS_DIR, xp_name, 'config.json'),
                               prefix=PATHS['cluster_base_dir'])

    # Initialize mAP computation
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False,
                                                      num_classes=len(config['ids_to_labels']))

    if i == 0:
        general_results = initialize_general_results(config['ids_to_labels'])

    # Retrieve the eval pages
    pages = []
    for dict_ in config['data']['eval']:
        commentary = CanonicalCommentary.from_json(os.path.join(PATHS['cluster_base_dir'], dict_['id'],
                                                                PATHS['canonical'], dict_['run'] + '.json'))
        page_ids = get_olr_splits_page_ids(commentary.id, [dict_['split']])
        pages += [p for p in commentary.children['page']
                  if p.id in page_ids]

    # Create the model
    if not config['text_only']:
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(config['model_name_or_path'])
        model = LayoutLMv3ForTokenClassification.from_pretrained(os.path.join(RUNS_DIR, xp_name, 'model'))
        feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(config['model_name_or_path'], apply_ocr=False)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name_or_path'], add_prefix_space=True)
        model = RobertaForTokenClassification.from_pretrained(os.path.join(RUNS_DIR, xp_name, 'model'))
        feature_extractor = None

    # Walk over pages
    for page in pages:

        # predict the page
        words, labels = align_predicted_page(page,
                                             rois=config['rois'],
                                             labels_to_ids=config['labels_to_ids'],
                                             ids_to_labels=config['ids_to_labels'],
                                             regions_to_coarse_labels=config['region_types_to_labels'],
                                             tokenizer=tokenizer,
                                             feature_extractor=feature_extractor,
                                             model=model,
                                             unknownify_tokens=config['unknownify_tokens'],
                                             text_only=config['text_only'])

        # We now create the predicted regions
        pred_regions = []
        for i in range(len(words)):
            if i == 0:
                region = {'words': [words[i]]}
            else:
                if labels[i] != labels[i - 1] or words[i].bbox.xyxy[-1] < (
                        words[i - 1].bbox.xyxy[1] - 50):  # for double col
                    region['label'] = labels[i - 1]
                    pred_regions.append(region)
                    region = {'words': [words[i]]}

                else:
                    region['words'].append(words[i])
        # Append the last region
        region['label'] = labels[-1]
        pred_regions.append(region)

        # Create the region's bbox
        for region in pred_regions:
            region['bbox'] = Shape([p for w in region['words'] for p in w.bbox.bbox])

        # Swallow included regions
        for region in pred_regions:
            if any([is_bbox_within_bbox(region['bbox'].bbox, r['bbox'].bbox)
                    for r in pred_regions if r != region]):
                region['is_included'] = True
            else:
                region['is_included'] = False
        pred_regions = [r for r in pred_regions if not r['is_included']]

        # Draw page words and regions to control
        labels_to_colors = {l: c + tuple([127]) for l, c in
                            zip(config['labels_to_ids'].keys(),
                                list(COLORS['distinct'].values()) + list(COLORS['hues'].values()))}
        img = PIL.Image.open(page.image.path)
        img = draw_page_labels(img=img,
                               words=words,
                               labels=labels,
                               labels_to_colors=labels_to_colors)

        draw = PIL.ImageDraw.Draw(img, 'RGBA')
        for region in pred_regions:
            draw.rounded_rectangle(xy=region['bbox'].bbox,
                                   outline=labels_to_colors[region['label']], radius=4,
                                   width=10)

        img = draw_caption(img, labels_to_colors=labels_to_colors,
                           font_dir='/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/dissemination/media/fonts')

        img_pred_dir = os.path.join(RUNS_DIR, xp_name, 'predictions/images')
        os.makedirs(img_pred_dir, exist_ok=True)
        img.save(os.path.join(img_pred_dir, page.id + '.png'))

        # Compute the mAP

        # Get the preds
        preds = []
        for r in pred_regions:
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            r_pred = r['bbox'].xyxy + [config['labels_to_ids'][r['label']]] + [1]
            preds.append(r_pred)
        preds_array = np.array(preds)

        # get the gt
        groundtruth = []
        for r in page.children['region']:
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
                              for r in p.children['region']
                              if config['labels_to_ids'][config['region_types_to_labels'][r.info['region_type']]] == k])

    general_results = update_general_results(general_results=general_results,
                                             metrics=metrics, xp_name=xp_name,
                                             ids_to_labels=config['ids_to_labels'])

    df = pd.DataFrame.from_dict(general_results)

tsv_path = os.path.join(RUNS_DIR, 'general_results.tsv')
df.to_csv(tsv_path, sep='\t', index=False)

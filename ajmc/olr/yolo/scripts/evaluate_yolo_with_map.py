import os

import pandas as pd

from ajmc.commons.image import Image
import glob
import yaml
import numpy as np
from mean_average_precision import MetricBuilder

from ajmc.commons.file_management.utils import walk_dirs
from ajmc.olr.map_utils import initialize_general_results, update_general_results
from ajmc.olr.yolo.utils import parse_yolo_txt


def do_map_for_yolo(images_dir: str,
                    gt_dir: str,
                    preds_dir: str,
                    ids_to_labels,
                    image_format='png',
                    ):
    # Create the Metric builder
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=len(ids_to_labels.keys()))
    counts = {l: 0 for l in ids_to_labels.values()}

    # Walk over each prediction txt in `preds_dir`
    for pred_path in glob.glob(os.path.join(preds_dir, '*.txt')):
        pred_name = pred_path.split('/')[-1]

        # Get the image to un-normalize
        image_name = pred_name.replace('.txt', '.' + image_format)
        image = Image(path=os.path.join(images_dir, image_name))

        # Parse preds
        preds = parse_yolo_txt(path=pred_path,
                               ids_to_label=ids_to_labels,
                               image_width=image.width,
                               image_height=image.height,
                               is_groundtruth=False)
        preds_array = np.array([list(l['bbox'].xyxy) + [l['label_id'], l['conf']] for l in preds])

        # Parse Groundtruth
        gt = parse_yolo_txt(path=os.path.join(gt_dir, pred_name),
                            ids_to_label=ids_to_labels,
                            image_width=image.width,
                            image_height=image.height,
                            is_groundtruth=True)

        gt_array = np.array([list(l['bbox'].xyxy) + [l['label_id']] + [0, 0] for l in gt])

        metric_fn.add(preds_array, gt_array)

        # Do the counts:
        for l in gt:
            counts[ids_to_labels[l['label_id']]] += 1

    metrics = metric_fn.value(iou_thresholds=(0.5))
    for k, dict_ in metrics[0.5].items():
        dict_['count'] = counts[ids_to_labels[k]]

    return metrics

#%%

DATASETS_DIR = '/scratch/sven/yolo/datasets/'
RUNS_DIR = '/scratch/sven/yolo/runs/'
for xp in ['4D_segmonto_fine', '4E_segmonto_coarse']:
    with open(os.path.join(DATASETS_DIR, 'multiclass', f'{xp}', 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ids_to_labels = {i: l for i, l in enumerate(config['names'])}
    metrics = do_map_for_yolo(images_dir=f'/scratch/sven/yolo/datasets/multiclass/{xp}/images/eval',
                              gt_dir=f'/scratch/sven/yolo/datasets/multiclass/{xp}/labels/eval',
                              preds_dir=f'/scratch/sven/yolo/runs/multiclass/{xp}/detect/exp/labels',
                              ids_to_labels=ids_to_labels,)
    general_results = initialize_general_results(ids_to_labels)
    general_results = update_general_results(general_results, metrics, f'{xp}', ids_to_labels)

    df = pd.DataFrame.from_dict(general_results)
    tsv_path = os.path.join(RUNS_DIR, 'multiclass', f'{xp}', 'general_results.tsv')
    df.to_csv(tsv_path, sep='\t', index=False)

# #%%
# xp_series = ['binary_class',
#              'multiclass']
#
#
# for xp_serie in xp_series:
#
#     for i, xp_name in enumerate(walk_dirs(os.path.join(DATASETS_DIR, xp_serie))):
#         images_dir = os.path.join(DATASETS_DIR, xp_serie, xp_name, 'images/eval')
#         gt_dir = os.path.join(DATASETS_DIR, xp_serie, xp_name, 'labels/eval')
#         preds_dir = os.path.join(RUNS_DIR, xp_serie, xp_name, 'detect/labels', )
#
#         # get labels_map
#         with open(os.path.join(DATASETS_DIR, xp_serie, xp_name, 'config.yaml')) as f:
#             config = yaml.load(f, Loader=yaml.FullLoader)
#         ids_to_labels = {i: l for i, l in enumerate(config['names'])}
#
#         # Create the general_results.
#         if i == 0:
#             general_results = initialize_general_results(ids_to_labels=ids_to_labels)
#
#         metrics = do_map_for_yolo(images_dir=images_dir,
#                                   gt_dir=gt_dir,
#                                   preds_dir=preds_dir,
#                                   ids_to_labels=ids_to_labels,
#                                   )
#
#         general_results = update_general_results(general_results=general_results, metrics=metrics,
#                                                  xp_name=xp_name, ids_to_labels=ids_to_labels)
#
#     df = pd.DataFrame.from_dict(general_results)
#     tsv_path = os.path.join(RUNS_DIR, xp_serie, 'general_results.tsv')
#     df.to_csv(tsv_path, sep='\t', index=False)

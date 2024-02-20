#%% Imports and functions declarations
import json
import shutil
from pathlib import Path
from typing import List, Dict

import cv2

from ajmc.commons import geometry as geom, variables as vs
from ajmc.commons.arithmetic import safe_divide
from ajmc.commons.image import draw_box
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.olr.evaluation import compute_shapes_confusion_matrix, compute_mean_iou
from ajmc.olr.line_detection import models
from ajmc.olr.line_detection.data_processing import get_split_page_ids, get_pages_lines


ROOT_LOGGER.setLevel('INFO')
logger = get_ajmc_logger(__name__)


def compute_micro_averaged_iou(predictions: Dict[str, List[geom.Shape]], groundtruth: Dict[str, List[geom.Shape]]) -> float:
    """Compute the micro-averaged IoU score for the given predictions and groundtruth.
    
    Note:
        The micro-averaged IoU score is the mean IoU score for each page.
    
    Args:
        predictions: The predictions for each page, as ``{"page_id": [line1, line2, ...]}``.
        groundtruth: The groundtruth for each page, as ``{"page_id": [line1, line2, ...]}``.
    """

    scores = [compute_mean_iou(lines, groundtruth[page_id]) for page_id, lines in predictions.items()]

    return sum(scores) / len(scores)


def compute_micro_averaged_metrics(predictions: Dict[str, List[geom.Shape]], groundtruth: Dict[str, List[geom.Shape]]) -> Dict[str, float]:
    # Create the confusion matrix for each model

    scores = {k: [] for k in ['F1', 'Recall', 'Precision']}

    for page_id, lines in predictions.items():
        conf_mtrx = compute_shapes_confusion_matrix(lines, groundtruth[page_id])
        scores['F1'].append(2 * conf_mtrx['TP'] / (2 * conf_mtrx['TP'] + conf_mtrx['FP'] + conf_mtrx['FN']))
        scores['Recall'].append(conf_mtrx['TP'] / (conf_mtrx['TP'] + conf_mtrx['FN']))
        scores['Precision'].append(safe_divide(conf_mtrx['TP'], (conf_mtrx['TP'] + conf_mtrx['FP'])))

    return {k: sum(v) / len(v) for k, v in scores.items()}


def import_predictions(page_ids: List[str]):
    predictions = {'blla': {}, 'legacy': {}, 'blla_adjusted': {}, 'legacy_adjusted': {}}

    for comm_id in vs.ALL_COMM_IDS:
        comm_olr_dir = vs.get_comm_olr_lines_dir(comm_id)
        comm_page_ids = [page_id for page_id in page_ids if page_id.startswith(comm_id)]

        for page_id in comm_page_ids:
            for model_name in predictions.keys():
                page_path = comm_olr_dir / model_name / (page_id + '.json')
                page_preds = json.loads(page_path.read_text())
                predictions[model_name][page_id] = [geom.Shape(points) for points in page_preds]

    return predictions


#%% Set directories and paths

EXP_DIR = Path('/Users/sven/Desktop/line_detection_experiments')
EXP_DIR.mkdir(exist_ok=True)
LINES_VIA_PATH = Path('/Users/sven/drive/lines_annotation.json')

#%% Import the ground truth lines and the predictions
via_dict = json.loads(LINES_VIA_PATH.read_text(encoding='utf-8'))
full_groundtruth = get_pages_lines(get_split_page_ids('test') + get_split_page_ids('train'), via_dict)

# Balance commentaries
import random

random.seed(0)

balanced_groundtruth = {}
for comm_id in vs.ALL_COMM_IDS:
    comm_gt_keys = [k for k in full_groundtruth.keys() if k.startswith(comm_id)]
    sampled_keys = random.sample(comm_gt_keys, k=min(40, len(comm_gt_keys)))
    if len(sampled_keys) < 5:
        print(f'Comm {comm_id} has less than 20 pages')
        continue
    balanced_groundtruth.update({k: full_groundtruth[k] for k in sampled_keys})

#%%
predictions = import_predictions(list(balanced_groundtruth.keys()))

#%% Get the results of the baselines

results = {}

for model_name in ['blla', 'legacy', 'blla_adjusted', 'legacy_adjusted']:
    # Declare the model's results dict, setting all the parameters to None (as baselines do not have parameters)
    results[model_name] = {k: None for k in ['split_lines', 'double_line_threshold', 'minimal_height_factor', 'line_inclusion_threshold']}
    results[model_name]['iou'] = compute_micro_averaged_iou(predictions[model_name], balanced_groundtruth)
    scores = compute_micro_averaged_metrics(predictions[model_name], balanced_groundtruth)
    for metric, score in scores.items():
        results[model_name][metric] = score

#%%

for split_lines in [True]:
    for double_line_threshold in [1.4, 1.6]:
        for minimal_height_factor in [0.4, 0.35, 0.5]:
            for line_inclusion_threshold in [0.6, 0.65]:

                combined_model = models.CombinedModel(adjusted_legacy_model=None,
                                                      adjusted_blla_model=None,
                                                      line_inclusion_threshold=line_inclusion_threshold,
                                                      minimal_height_factor=minimal_height_factor,
                                                      double_line_threshold=double_line_threshold,
                                                      split_lines=split_lines)

                model_name = f'combined_dl{double_line_threshold}_mh{minimal_height_factor}_it{line_inclusion_threshold}_sp{int(split_lines)}'
                if model_name in predictions:
                    continue
                print(f'Running {model_name}...')

                # COMPUTE THE COMBINED LINES
                predictions[model_name] = {}
                for page_id in balanced_groundtruth.keys():
                    predictions[model_name][page_id] = combined_model.predict(None,
                                                                              legacy_predictions=predictions['legacy_adjusted'][page_id],
                                                                              blla_predictions=predictions['blla_adjusted'][page_id])

                # Evaluate the predictions
                results[model_name] = {'split_lines': split_lines,
                                       'double_line_threshold': double_line_threshold,
                                       'minimal_height_factor': minimal_height_factor,
                                       'line_inclusion_threshold': line_inclusion_threshold}

                results[model_name]['iou'] = compute_micro_averaged_iou(predictions[model_name], balanced_groundtruth)
                scores = compute_micro_averaged_metrics(predictions[model_name], balanced_groundtruth)
                for metric, score in scores.items():
                    results[model_name][metric] = score

                # Draw the bounding boxes

                # Create the output directory
                output_dir = EXP_DIR / f'outputs/{model_name}'
                shutil.rmtree(output_dir, ignore_errors=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Prepare the esthetics
                colors = {'legacy_adjusted': vs.COLORS['distinct']['blue'],
                          'blla_adjusted': vs.COLORS['distinct']['green'],
                          model_name: vs.COLORS['distinct']['red']}

                thicknesses = {'legacy_adjusted': 1,
                               'blla_adjusted': 1,
                               model_name: 2}

                # We now draw the bounding boxes of the predictions and the ground truth
                for page_id, lines in predictions[model_name].items():
                    img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')
                    img = cv2.imread(str(img_path))
                    output_path = output_dir / (page_id + '.png')

                    for model_name_ in ['legacy_adjusted', 'blla_adjusted', model_name]:
                        for line in predictions[model_name_][page_id]:
                            img = draw_box(line.bbox, img, stroke_color=colors[model_name_],
                                           stroke_thickness=thicknesses[model_name_])

                    cv2.imwrite(str(output_path), img)

                # Write files
                conf = {'double_line_threshold': double_line_threshold,
                        'minimal_height_factor': minimal_height_factor,
                        'line_inclusion_threshold': line_inclusion_threshold,
                        'split_lines': split_lines}

                (output_dir / 'predictions.json').write_text(
                        json.dumps([p.points for preds in predictions[model_name].values() for p in preds], indent=2))
                (output_dir / 'config.json').write_text(json.dumps(conf, indent=2))
                (output_dir / 'results.json').write_text(json.dumps(results[model_name], indent=2))
                (output_dir.parent / 'all_results.json').write_text(json.dumps(results, indent=2))

#%% Inspect the results
import pandas as pd

try:
    df = pd.DataFrame.from_dict(results, orient='index')
except NameError:
    results = json.loads((EXP_DIR / 'outputs/all_results.json').read_text(encoding='utf-8'))
    df = pd.DataFrame.from_dict(results, orient='index')

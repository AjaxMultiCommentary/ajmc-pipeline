#%% Imports and functions declarations
import json
from pathlib import Path
from typing import List, Dict

import cv2
from tqdm import tqdm

from ajmc.commons import geometry as geom, variables as vs
from ajmc.commons.geometry import adjust_bbox_to_included_contours
from ajmc.commons.image import find_contours, remove_artifacts_from_contours
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.olr.line_segmentation.data_processing import get_split_page_ids, get_pages_lines
from ajmc.olr.line_segmentation.models import combine_kraken_predictions

ROOT_LOGGER.setLevel('INFO')
logger = get_ajmc_logger(__name__)


def compute_page_confusion_matrix(pred_lines: List[geom.Shape], gt_lines: List[geom.Shape], iou_threshold: float = 0.8):
    """Computes the confusion matrix of the given lines.
    """
    FalseNegatives = 0
    TruePositives = 0
    FalsePositives = 0
    TrueNegatives = 0

    for pred_line in pred_lines:
        iou = match_shape_to_gt(pred_line, gt_lines)
        if iou > iou_threshold:
            TruePositives += 1
        else:
            FalsePositives += 1

    for gt_line in gt_lines:
        iou = match_shape_to_gt(gt_line, pred_lines)
        if iou < iou_threshold:
            FalseNegatives += 1

    return {'TP': TruePositives, 'FP': FalsePositives, 'FN': FalseNegatives, 'TN': TrueNegatives}


def compute_shapes_iou(shape1: geom.Shape, shape2: geom.Shape) -> float:
    """Computes the intersection over union of two bboxes.
    """
    intersection = geom.compute_bbox_overlap_area(shape1.bbox, shape2.bbox)
    union = shape1.area + shape2.area - intersection
    return intersection / union


def match_shape_to_gt(pred_shape: geom.Shape, gt_shapes: List[geom.Shape]) -> float:
    """Finds the gt shape with the highest iou with the given shape."""
    try:
        return max([compute_shapes_iou(pred_shape, gt_shape) for gt_shape in gt_shapes])
    except ValueError:
        return 0


def compute_mean_iou(pred_shapes: List[geom.Shape], gt_shapes: List[geom.Shape]) -> float:
    """Computes the mean iou of the given shapes with the gt shapes.
    """
    return sum([match_shape_to_gt(shape, gt_shapes) for shape in pred_shapes]) / max(len(pred_shapes), len(gt_shapes))


def compute_all_ious(preds: dict):
    scores = {}
    for model_name, model_preds in preds.items():
        scores[model_name] = {}
        for page_id, lgcy_adj_lines in model_preds.items():
            scores[model_name][page_id] = compute_mean_iou(lgcy_adj_lines, groundtruth[page_id])

    for model_name, model_scores in scores.items():
        print(f'{model_name}: {sum(model_scores.values()) / len(model_scores)}')

    return scores


def get_f1_scores(preds: dict):
    # Create the confusion matrix for each model
    confusion_matrices = {}

    for model_name, model_preds in preds.items():
        confusion_matrices[model_name] = {k: 0 for k in ['TP', 'FP', 'FN', 'TN']}
        for page_id, lgcy_adj_lines in model_preds.items():
            matrix = compute_page_confusion_matrix(lgcy_adj_lines, groundtruth[page_id])
            for k in matrix.keys():
                confusion_matrices[model_name][k] += matrix[k]

    for model_name, matrix in confusion_matrices.items():
        print(f'{model_name}: {matrix}')
        f1_score = 2 * matrix['TP'] / (2 * matrix['TP'] + matrix['FP'] + matrix['FN'])
        print(f'F1 score: {f1_score}')
        recall = matrix['TP'] / (matrix['TP'] + matrix['FN'])
        print(f'Recall: {recall}')
        precision = matrix['TP'] / (matrix['TP'] + matrix['FP'])
        print(f'Precision: {precision}')

    return confusion_matrices


def import_raw_predictions(page_ids: List[str]):
    # Create a dictionary {'model_name': {'page_id': [line1, line2, ...], ...}, ...}
    preds = {'blla': {}, 'legacy': {}}
    for comm_id in vs.ALL_COMM_IDS:
        comm_olr_dir = vs.get_comm_base_dir(comm_id) / 'olr/lines'
        for json_path in (comm_olr_dir / 'legacy').glob('*.json'):
            if json_path.stem in page_ids:
                page_dict = json.loads(json_path.read_text())
                preds['legacy'][json_path.stem] = [geom.Shape.from_xyxy(*box) for box in page_dict['boxes']]
        for json_path in (comm_olr_dir / 'blla').glob('*.json'):
            if json_path.stem in page_ids:
                page_dict = json.loads(json_path.read_text())
                preds['blla'][json_path.stem] = [geom.Shape(l['boundary']) for l in page_dict['lines']]
    return preds


def optimize_page_predictions(page_preds: Dict[List[geom.Shape]],
                              page_id: str,
                              artifact_size_threshold: float = 0.003):
    """Optimizes the predictions for a single page.

    Args:
        page_preds: A dictionary of the form {'model_name': [line1, line2, ...], ...}
        page_id: The id of the page to optimize.
    """
    page_img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')
    page_image = cv2.imread(str(page_img_path))
    page_contours = find_contours(page_image)
    artifact_perimeter_threshold = int(page_image.shape[1] * artifact_size_threshold)
    page_contours = remove_artifacts_from_contours(page_contours, artifact_perimeter_threshold)

    for k in page_preds.keys():
        page_preds[k] = [adjust_bbox_to_included_contours(l.bbox, page_contours) for l in page_preds[k]]

    return page_preds


def optimize_predictions(preds: dict):
    preds.update({'blla_adjusted': {}, 'legacy_adjusted': {}})

    for page_id in tqdm(preds['legacy'].keys()):
        page_img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')
        page_image = cv2.imread(str(page_img_path))
        page_contours = find_contours(page_image)
        artifact_perimeter_threshold = int(page_image.shape[1] * ARTIFACT_SIZE_THRESHOLD)
        page_contours = remove_artifacts_from_contours(page_contours, artifact_perimeter_threshold)

        for model_name in ['blla', 'legacy']:
            preds[model_name + '_adjusted'][page_id] = [adjust_bbox_to_included_contours(l.bbox, page_contours) for l in preds[model_name][page_id]]

    return preds


def get_combined_predictions(preds: dict):
    # by default, we take the legacy line
    # if a line is taller than 1.5 * the average line height, and it contains blla lines which are at least 90% overlapping, we take the blla lines
    # if a blla line is not overlapping with any legacy line, we add it

    preds['combined'] = {}

    for page_id, lgcy_adj_lines in preds['legacy_adjusted'].items():
        preds['combined'][page_id] = combine_kraken_predictions(legacy_preds=preds['legacy_adjusted'][page_id],
                                                                blla_preds=preds['blla_adjusted'][page_id],
                                                                line_inclusion_threshold=LINE_INCLUSION_THRESHOLD,
                                                                double_line_height_threshold=DOUBLE_LINE_HEIGHT_THRESHOLD,
                                                                minimal_height_factor=MINIMAL_HEIGHT_FACTOR)

    SAVE_PATH.write_text(json.dumps(preds, default=lambda x: x.bbox, sort_keys=True), encoding='utf-8')
    return preds


#%% CONSTANTS

EXP_DIR = Path('/Users/sven/Desktop/line_detection_experiments')
EXP_DIR.mkdir(exist_ok=True)
LINES_VIA_PATH = Path('/Users/sven/drive/lines_annotation.json')
SAVE_PATH = EXP_DIR / 'predictions.json'

FORCE_RECOMPUTE = False
ARTIFACT_SIZE_THRESHOLD = 0.003
DOUBLE_LINE_HEIGHT_THRESHOLD = 1.6
MINIMAL_HEIGHT_FACTOR = 0.35
LINE_INCLUSION_THRESHOLD = 0.8

#%% Import the ground truth lines and the predictions
via_dict = json.loads(LINES_VIA_PATH.read_text(encoding='utf-8'))
groundtruth = get_pages_lines(get_split_page_ids('test') + get_split_page_ids('train'), via_dict)

if SAVE_PATH.exists() and not FORCE_RECOMPUTE:
    preds = json.loads(SAVE_PATH.read_text(encoding='utf-8'))
    preds = {k: {page_id: [geom.Shape(box) for box in boxes] for page_id, boxes in v.items()} for k, v in preds.items()}

else:
    preds = import_raw_predictions(get_split_page_ids('test') + get_split_page_ids('train'))
    preds = optimize_predictions(preds)
    SAVE_PATH.write_text(json.dumps(preds, default=lambda x: x.bbox, sort_keys=True), encoding='utf-8')

#%% COMPUTE THE COMBINED LINES
preds = get_combined_predictions(preds)

#%% Evaluate the predictions
compute_all_ious(preds)
get_f1_scores(preds)

#%% DRAWING THE BOUNDING BOXES
import shutil

output_dir = Path(f'/Users/sven/Desktop/line_detection_experiments/outputs/combined/')
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

# We now draw the bounding boxes of the predictions and the ground truth
for test_page_id, test_page_lines in preds['combined'].items():
    # if not test_page_id == 'cu31924087948174_0022':
    #     continue
    page_img_path = vs.get_comm_img_dir(test_page_id.split('_')[0]) / (test_page_id + '.png')
    page_image = cv2.imread(str(page_img_path))

    for model in ['legacy_adjusted', 'blla_adjusted', 'combined']:
        for line in preds[model][test_page_id]:
            cv2.rectangle(page_image, (int(line.bbox[0][0]), int(line.bbox[0][1])),
                          (int(line.bbox[1][0]), int(line.bbox[1][1])),
                          (0, 0, 255) if model == 'combined' else (255, 0, 0) if model == 'legacy_adjusted' else (0, 255, 0),
                          2 if model == 'combined' else 1)

    cv2.imwrite(str(output_dir / (test_page_id + '.png')), page_image)

#%% WORKBENCH
# cu31924087948174_0010 Included line. remove them
# cu31924087948174_0022 Overlapping lines. remove them

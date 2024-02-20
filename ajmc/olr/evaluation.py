from typing import List

import numpy as np

from ajmc.commons import geometry as geom


def compute_shapes_confusion_matrix(pred_shapes: List[geom.Shape],
                                    gt_shapes: List[geom.Shape],
                                    iou_threshold: float = 0.8):
    """Computes the confusion matrix of predicted and groundtruth shapes at a given IoU threshold.

    Args:
        pred_shapes: The predicted shapes.
        gt_shapes: The groundtruth shapes.
        iou_threshold: The IoU threshold at which to consider a shape as a true positive.
    """
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0

    for pred_shape in pred_shapes:
        iou = match_shape_to_gt(pred_shape, gt_shapes)
        if iou > iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

    for gt_line in gt_shapes:
        iou = match_shape_to_gt(gt_line, pred_shapes)
        if iou < iou_threshold:
            false_negatives += 1

    return {'TP': true_positives, 'FP': false_positives, 'FN': false_negatives, 'TN': true_negatives}


def compute_shapes_iou(shape1: geom.Shape, shape2: geom.Shape) -> float:
    """Computes the intersection over union of two ``geometry.Shape``s."""
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
    """Computes the mean IoU of the given shapes with the gt shapes."""
    return sum([match_shape_to_gt(shape, gt_shapes) for shape in pred_shapes]) / max(len(pred_shapes), len(gt_shapes))


# ========== mAP utilities ==========
metrics_abbrev = {'ap': 'AP', 'precision': 'P', 'recall': 'R', 'number': 'N'}


def initialize_general_results(ids_to_labels):
    general_results = {('info', 'exp'): [], ('all', 'mAP'): []}
    general_results.update({(n, m): [] for n in ids_to_labels.values()
                            for m in metrics_abbrev.values()})
    return general_results


def update_general_results(general_results, metrics, xp_name, ids_to_labels):
    general_results[('info', 'exp')].append(xp_name)
    general_results[('all', 'mAP')].append(float(metrics['mAP']))

    for l_id, dict_ in metrics[0.5].items():
        for m, score in dict_.items():
            if m == 'count':
                general_results[(ids_to_labels[l_id], 'N')].append(dict_[m])
            else:
                if dict_['count'] == 0:
                    general_results[(ids_to_labels[l_id], metrics_abbrev[m])].append(np.nan)
                else:
                    general_results[(ids_to_labels[l_id], metrics_abbrev[m])].append(float(score) if m == 'ap' else float(score.mean()))

    return general_results

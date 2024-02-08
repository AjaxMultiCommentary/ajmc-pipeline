import json
from typing import Dict, List

import cv2

from ajmc.commons import variables as vs
from ajmc.commons.geometry import Shape, adjust_bbox_to_included_contours
from ajmc.commons.image import find_contours, remove_artifacts_from_contours


def optimize_page_predictions(page_preds: Dict[str, List[Shape]],
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


for comm_dir in vs.COMMS_DATA_DIR.iterdir():
    if not comm_dir.is_dir():
        continue

    lines_dir = comm_dir / 'olr/lines'
    for img_path in vs.get_comm_img_dir(comm_dir.name).glob('*.png'):
        preds = {}

        blla_preds = json.loads((lines_dir / 'blla' / (img_path.stem + '.json')).read_text())
        preds['blla_adjusted'] = [Shape(l['boundary']) for l in blla_preds['lines']]

        legacy_preds = json.loads((lines_dir / 'legacy' / (img_path.stem + '.json')).read_text())
        preds['legacy_adjusted'] = [Shape.from_xyxy(*box) for box in legacy_preds['boxes']]

        preds = optimize_page_predictions(preds, img_path.stem)

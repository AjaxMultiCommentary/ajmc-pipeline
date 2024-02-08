from pathlib import Path
from typing import List

import cv2
import easyocr
from kraken import blla, pageseg
from kraken.lib import vgsl
from PIL import Image

from ajmc.commons import geometry as geom
from ajmc.commons.geometry import adjust_bbox_to_included_contours, is_bbox_within_bbox_with_threshold, are_bboxes_overlapping_with_threshold
from ajmc.commons.image import draw_box, find_contours, remove_artifacts_from_contours


class LineDetectionModel:

    def __init__(self):
        pass

    def predict(self, img_path: Path) -> List[geom.Shape]:
        pass

    def draw_predictions(self, img_path: Path, out_path: Path):
        img = cv2.imread(str(img_path))
        for shape in self.predict(img_path):
            img = draw_box(shape.bbox, img)
        cv2.imwrite(str(out_path), img)


    def evaluate(self, image, gt):
        pass


class EasyOCRModel(LineDetectionModel):

    def __init__(self, languages=['en']):
        super().__init__()
        self.reader = easyocr.Reader(languages)

    def predict(self, img_path: Path):
        return [geom.Shape.from_xxyy(*pred) for pred in self.reader.detect(str(img_path))[0][0]]


class KrakenLegacyModel(LineDetectionModel):
    def __init__(self):
        super().__init__()

    def predict(self, img_path: Path):
        img = Image.open(img_path)
        # img = binarization.nlbin(img)
        return [geom.Shape.from_xyxy(*box) for box in pageseg.segment(img)['boxes']]


class KrakenBllaModel(LineDetectionModel):

    def __init__(self, model_path: Path):
        super().__init__()
        self.model = vgsl.TorchVGSLModel.load_model(model_path)

    def predict(self, img_path: Path):
        img = Image.open(img_path)
        # img = binarization.nlbin(img)
        return [geom.Shape(l['boundary']) for l in blla.segment(img, model=self.model)['lines']]


def adjust_predictions(img_path: Path, predictions: List[List[geom.Shape]], artifact_size_threshold: float = 0.003):
    img = cv2.imread(str(img_path))
    page_contours = find_contours(img)
    artifact_perimeter_threshold = int(img.shape[1] * artifact_size_threshold)
    page_contours = remove_artifacts_from_contours(page_contours, artifact_perimeter_threshold)

    return [[adjust_bbox_to_included_contours(l.bbox, page_contours) for l in preds] for preds in predictions]


def combine_kraken_predictions(legacy_preds: List[geom.Shape],
                               blla_preds: List[geom.Shape],
                               line_inclusion_threshold: float = 0.8,
                               minimal_height_factor: float = 0.35,
                               double_line_height_threshold: float = 1.6) -> List[geom.Shape]:
    """Combines the predictions of the legacy and blla models for a single page"""

    combined_preds = []

    # We first get the average line height of the legacy model
    line_heights = sorted([l.height for l in legacy_preds])
    # We remove the 20% smallest and tallest lines to avoid the diacritics lines
    if len(line_heights) >= 5:
        line_heights = line_heights[len(line_heights) // 5: - len(line_heights) // 5]
    try:
        avg_line_height = sum(line_heights) / len(line_heights)
    except ZeroDivisionError:
        avg_line_height = 0

    # Discard small legacy and blla lines
    blla_preds = [l for l in blla_preds if l.height > minimal_height_factor * avg_line_height]
    legacy_preds = [l for l in legacy_preds if l.height > minimal_height_factor * avg_line_height]

    # We first try to match the legacy lines with the blla lines
    for lgcy_line in legacy_preds:
        overlapping_blla_lines = [l for l in blla_preds
                                  if is_bbox_within_bbox_with_threshold(l.bbox, lgcy_line.bbox, line_inclusion_threshold)]

        if lgcy_line.height > double_line_height_threshold * avg_line_height:
            if len(overlapping_blla_lines) == 1:
                if overlapping_blla_lines[0].height > double_line_height_threshold * avg_line_height:
                    combined_preds.append(lgcy_line)
            elif len(overlapping_blla_lines) > 1:
                combined_preds.extend(overlapping_blla_lines)
        else:
            if len(overlapping_blla_lines) == 2:
                left_line, right_line = sorted(overlapping_blla_lines, key=lambda l: l.bbox[0][0])
                combined_preds.append(geom.Shape([lgcy_line.bbox[0], (left_line.bbox[1][0], lgcy_line.bbox[1][1])]))

                # Compute the new left x value of the right line
                new_left_x = left_line.bbox[1][0] + int(2 * (right_line.bbox[0][0] - left_line.bbox[1][0]) / 3)
                combined_preds.append(geom.Shape([(new_left_x, lgcy_line.bbox[0][1]), lgcy_line.bbox[1]]))

            else:
                combined_preds.append(lgcy_line)

    for line in blla_preds:
        if not any([are_bboxes_overlapping_with_threshold(line.bbox, l.bbox, 0.35) for l in combined_preds]):
            combined_preds.append(line)

    combined_preds = [l for l in combined_preds if
                      not any([is_bbox_within_bbox_with_threshold(l.bbox, l2.bbox, 0.9) for l2 in combined_preds if l2 != l])]

    return combined_preds

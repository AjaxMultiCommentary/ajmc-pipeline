from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from kraken import blla, pageseg, binarization
from kraken.lib import vgsl
from PIL import Image

from ajmc.commons import geometry as geom
from ajmc.commons.geometry import adjust_bbox_to_included_contours, is_bbox_within_bbox_with_threshold, are_bboxes_overlapping_with_threshold
from ajmc.commons.image import draw_box, find_contours, remove_artifacts_from_contours


class LineDetectionModel:
    """Abstract class for all line detection models."""

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, img_path: Path, **kwargs) -> List[geom.Shape]:
        """Predicts the lines in the image at img_path, returning a list of Shape objects."""
        pass

    def draw_predictions(self, img_path: Path,
                         output_path: Path,
                         predictions: Optional[List[geom.Shape]] = None,
                         color: Tuple[int, int, int] = (0, 0, 255),
                         thickness: int = 1):
        """Draws the model's predictions on the image at img_path and saves the result to out_path.

        Args:
            img_path: The path to the input image.
            output_path: The path to save the output image.
            predictions: The predictions to draw. If None, the predictions are computed.
            color: The desired RGB color of the boxes.
            thickness: The thickness of the box's stroke.
        """

        img = cv2.imread(str(img_path))
        if predictions is None:
            predictions = self.predict(img_path)

        for shape in predictions:
            img = draw_box(shape.bbox, img, stroke_color=color, stroke_thickness=thickness)

        cv2.imwrite(str(output_path), img)


class EasyOCRModel(LineDetectionModel):

    def __init__(self, languages=['en']):
        import easyocr
        super().__init__()
        self.reader = easyocr.Reader(languages)

    def predict(self, img_path: Path):
        return [geom.Shape.from_xxyy(*pred) for pred in self.reader.detect(str(img_path))[0][0]]


class KrakenLegacyModel(LineDetectionModel):
    """A simple wrapper around Kraken's legacy line segmentation model."""

    def __init__(self):
        super().__init__()

    def predict(self, img_path: Path):
        img = Image.open(img_path)
        try:
            img = binarization.nlbin(img)
            return [geom.Shape.from_xyxy(*box) for box in pageseg.segment(img)['boxes']]
        except:
            return []


class KrakenBllaModel(LineDetectionModel):
    """A simple wrapper around Kraken's BL/LLA line segmentation model."""

    def __init__(self, model_path: Path):
        """Initializes the model with the given model_path.

        Args:
            model_path: The path to the BL/LLA model, usually in ``kraken/blla.mlmodel``, e.g.
            ``anaconda3/envs/kraken/lib/python3.10/site-packages/kraken/blla.mlmodel`` if you are using Anaconda.
        """
        super().__init__()
        self.model = vgsl.TorchVGSLModel.load_model(model_path)

    def predict(self, img_path: Path) -> List[geom.Shape]:
        img = Image.open(img_path)
        try:
            img = binarization.nlbin(img)
            return [geom.Shape(l['boundary']) for l in blla.segment(img, model=self.model)['lines']]
        except:
            return []


class AdjustedModel(LineDetectionModel):
    """A model that adjusts each prediction of a base model to the minimal bounding rectangle containing the predictions contours"""

    def __init__(self,
                 base_model: LineDetectionModel,
                 artifact_size_threshold: float = 0.003,
                 remove_side_margins: float = 0):
        """Initializes the model with the given base_model.

        Args:
            base_model: The base model to adjust.
            artifact_size_threshold: The threshold for the size of artifacts to remove from the contours, see
            ``ajmc.commons.image.remove_artifacts_from_contours`` for more details.
            remove_side_margins: The margin to remove from the sides of the image, see ``adjust_img_predictions`` for more details.
        """

        super().__init__()
        self.base_model = base_model
        self.artifact_size_threshold = artifact_size_threshold
        self.remove_side_margins = remove_side_margins

    def predict(self, img_path: Path,
                base_model_predictions: Optional[List[geom.Shape]] = None) -> List[geom.Shape]:
        """Adjusts the base model's predictions for the image at img_path and returns the adjusted predictions.

        Args:
            img_path: The path to the input image.
            base_model_predictions: The predictions of the base model. If None, the predictions are computed.
        """

        if base_model_predictions is None:
            base_model_predictions = self.base_model.predict(img_path)

        return adjust_predictions(img_path, base_model_predictions, self.artifact_size_threshold, self.remove_side_margins)


class CombinedModel(LineDetectionModel):
    """A model that combines the (potentially adjusted) predictions of Kraken's BL/LA and legacy models"""

    def __init__(self,
                 adjusted_legacy_model: Optional[LineDetectionModel] = None,
                 adjusted_blla_model: Optional[LineDetectionModel] = None,
                 line_inclusion_threshold: float = 0.8,
                 minimal_height_factor: float = 0.35,
                 double_line_threshold: float = 1.6,
                 split_lines: bool = False):
        """Initializes the model with the given adjusted adjusted_legacy_model and adjusted_blla_model.

        Args:
            adjusted_legacy_model: The legacy model to use. If ``None``, ``legacy_predictions`` must be provided in ``self.predict()``.
            adjusted_blla_model: The BL/LA model to use. If ``None``, ``legacy_predictions`` must be provided in ``self.predict()``.
            line_inclusion_threshold: The threshold for the inclusion of a line in the merging process, see ``combine_kraken_predictions`` for more details.
            minimal_height_factor: The minimal height factor for the inclusion of a line in the merging process, see ``combine_kraken_predictions`` for more details.
            double_line_threshold: The threshold for the inclusion of a line in the merging process, see ``combine_kraken_predictions`` for more details.
            split_lines: Whether to split the lines in the merging process, see ``combine_kraken_predictions`` for more details.
        """

        super().__init__()
        self.adjusted_legacy_model = adjusted_legacy_model
        self.adjusted_blla_model = adjusted_blla_model
        self.line_inclusion_threshold = line_inclusion_threshold
        self.minimal_height_factor = minimal_height_factor
        self.double_line_threshold = double_line_threshold
        self.split_lines = split_lines

    def predict(self, img_path: Optional[Path],
                legacy_predictions: Optional[List[geom.Shape]] = None,
                blla_predictions: Optional[List[geom.Shape]] = None) -> List[geom.Shape]:
        """Predicts the lines for the image at img_path using the legacy and BL/LLA models and returns the combined predictions.

        Args:
            img_path: The path to the input image. Can be ``None`` if ``legacy_predictions`` and ``blla_predictions`` are given.
            legacy_predictions: The predictions of the legacy model. If None, the predictions are computed.
            blla_predictions: The predictions of the BL/LLA model. If None, the predictions are computed.
        """

        if legacy_predictions is None:
            legacy_predictions = self.adjusted_legacy_model.predict(img_path)

        if blla_predictions is None:
            blla_predictions = self.adjusted_blla_model.predict(img_path)

        return combine_kraken_predictions(legacy_predictions, blla_predictions,
                                          line_inclusion_threshold=self.line_inclusion_threshold,
                                          minimal_height_factor=self.minimal_height_factor,
                                          double_line_threshold=self.double_line_threshold,
                                          split_lines=self.split_lines)


def adjust_predictions(img_path: Path,
                       predictions: List[geom.Shape],
                       artifact_size_threshold: float = 0.003,
                       remove_side_margins: float = 0) -> List[geom.Shape]:
    """Adjusts the predictions for the image at img_path and returns the adjusted predictions.

    Args:
        img_path: The path to the input image.
        predictions: The predictions to adjust.
        artifact_size_threshold: The threshold for the size of artifacts to remove from the contours, see
        ``ajmc.commons.image.remove_artifacts_from_contours`` for more details.
        remove_side_margins: The percentage of side margins to remove the contours from. For instance, setting ``remove_side_margins=0.05`` with an
         image of with 100px will lead to the exclusion of the contours with $x_{max} < 5$ or $x_{min} > 95$.
    """
    img = cv2.imread(str(img_path))
    page_contours = find_contours(img)

    artifact_perimeter_threshold = int(img.shape[1] * artifact_size_threshold)
    page_contours = remove_artifacts_from_contours(page_contours, artifact_perimeter_threshold)
    adjusted_predictions = [adjust_bbox_to_included_contours(l.bbox, page_contours) for l in predictions]

    if remove_side_margins > 0:
        left_margin = int(img.shape[1] * remove_side_margins)
        right_margin = int(img.shape[1] * (1 - remove_side_margins))
        adjusted_predictions = [p for p in adjusted_predictions if not (p.bbox[1][0] < left_margin or p.bbox[0][0] > right_margin)]

    return adjusted_predictions


def combine_kraken_predictions(legacy_preds: List[geom.Shape],
                               blla_preds: List[geom.Shape],
                               line_inclusion_threshold: float = 0.8,
                               minimal_height_factor: float = 0.35,
                               double_line_threshold: float = 1.6,
                               split_lines: bool = False) -> List[geom.Shape]:
    """Combines the predictions of the legacy and blla models for a single page

    Args:
        legacy_preds: The predictions of the legacy model
        blla_preds: The predictions of the blla model
        line_inclusion_threshold: The overlapping threshold for the inclusion of a line in the other model's prediction
        minimal_height_factor: The minimal height factor for a line to be considered as a line, with respect to the average line height of the page
        double_line_threshold: The threshold for a line to be considered as a double line
        split_lines: Whether to split the lines that are divided (horizontally) in two by the blla model
    """

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

    # We now try to match the legacy lines with the blla lines
    for lgcy_line in legacy_preds:
        overlapping_blla_lines = [l for l in blla_preds
                                  if is_bbox_within_bbox_with_threshold(l.bbox, lgcy_line.bbox, line_inclusion_threshold)]
        # Why isn't this using are_bboxes_overlapping_with_threshold?
        # This means we do not grab the blla lines that much bigger than the legacy line

        if lgcy_line.height > double_line_threshold * avg_line_height:
            if len(overlapping_blla_lines) == 1:
                if overlapping_blla_lines[0].height > double_line_threshold * avg_line_height:
                    combined_preds.append(lgcy_line)
            elif len(overlapping_blla_lines) > 1:
                combined_preds.extend(overlapping_blla_lines)
        else:
            if len(overlapping_blla_lines) == 2 and split_lines:
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

    non_overlapping_preds = []
    for l in combined_preds:
        if not any([is_bbox_within_bbox_with_threshold(l.bbox, l2.bbox, 0.7) for l2 in non_overlapping_preds]):
            non_overlapping_preds.append(l)

    return non_overlapping_preds

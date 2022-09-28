import cv2
from typing import List, Tuple, Optional, Union
import numpy as np
from ajmc.commons.geometry import Shape
from ajmc.commons.miscellaneous import lazy_property, get_custom_logger, lazy_init, BoxType

logger = get_custom_logger(__name__)


class Image:
    """Default class for AJMC images.

    Note:
          The center of `Image`-coordinates is the upper left corner, consistantly with cv2 and numpy. This implies
          that Y-coordinates are ascending towards the bottom of the image.
    """

    @lazy_init
    def __init__(self,
                 id: Optional[str] = None,
                 path: Optional[str] = None,
                 matrix: Optional[np.ndarray] = None,
                 word_range: Optional[Tuple[int, int]] = None):
        pass

    @lazy_property
    def matrix(self) -> np.ndarray:
        """np.ndarray of the image image matrix. Its shape is (height, width, channels)."""
        return cv2.imread(self.path)

    @lazy_property
    def height(self) -> int:
        return self.matrix.shape[0]

    @lazy_property
    def width(self) -> int:
        return self.matrix.shape[1]

    @lazy_property
    def contours(self):
        return find_contours(self.matrix)

    def crop(self,
             box: BoxType,
             margin: int = 0) -> 'Image':
        """Gets the slice of `self.matrix` corresponding to `box`.

        Args:
            box: The bbox delimiting the desired crop
            margin: The extra margin desired around `box`

        Returns:
             A new `Image` containing the desired crop.
        """
        cropped = self.matrix[box[0][1] - margin:box[1][1] + margin, box[0][0] - margin:box[1][0] + margin, :]

        return Image(matrix=cropped)

    def write(self, output_path: str):
        cv2.imwrite(output_path, self.matrix)


def binarize(img_matrix: np.ndarray,
             inverted: bool = False):
    """Binarizes a cv2 `image`"""
    binarization_type = (cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) if inverted else (cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, type=binarization_type)[1]


def find_contours(img_matrix: np.ndarray,
                  binarize: bool = True) -> List[Shape]:
    """Binarizes `img_matrix` and finds contours using `cv2.findContours`."""

    # This has to be done in cv2. Using cv2.THRESH_BINARY_INV to avoid looking for the white background as a contour
    if binarize:
        gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    else:
        thresh = img_matrix

    # alternative: CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Discard single-point contours
    contours = [Shape.from_numpy_array(c) for c in contours if len(c) > 1]

    return contours


def draw_boxes(boxes: List[BoxType],
               matrix: np.ndarray,
               color: Tuple[int, int, int] = (0, 0, 255),
               thickness: int = 2,
               output_path: str = None,
               show: bool = False) -> np.ndarray:
    """Draws a list of bboxes on an image matrix.

    Args:
        boxes: A list of bboxes.
        matrix: An image matrix to draw on.
        color: A tuple of BGR-color, e.g. (255,109, 118)
        thickness: An integer, see cv2, e.g. 2
        output_path: A path to write the image to.
        show: Whether to display the image.

    Returns:
        np.ndarray: The modified `matrix`

    """

    for box in boxes:
        matrix = cv2.rectangle(matrix, pt1=box[0], pt2=box[1], color=color, thickness=thickness)

    if output_path:
        cv2.imwrite(output_path, matrix)

    if show:
        cv2.imshow('image', matrix)

    return matrix


def draw_page_regions_lines_words(matrix: np.ndarray,
                                  page: Union['OcrPage', 'CanonicalPage'],
                                  output_path: Optional[str] = None,
                                  region_elements: bool = False):
    matrix = draw_boxes(boxes=[r.bbox.bbox for r in page.children.regions],
                        matrix=matrix,
                        color=(255, 0, 0),
                        thickness=3)
    if region_elements:
        matrix = draw_boxes([r.bbox.bbox for region in page.children.regions for r in region.children.lines],
                            matrix,
                            (0, 255, 0), thickness=2)
        matrix = draw_boxes([r.bbox.bbox for region in page.children.regions for r in region.children.words],
                            matrix,
                            thickness=1)
    else:
        matrix = draw_boxes([r.bbox.bbox for r in page.children.lines], matrix, (0, 255, 0), thickness=2)
        matrix = draw_boxes([r.bbox.bbox for r in page.children.words], matrix, thickness=1)

    if output_path:
        cv2.imwrite(output_path, matrix)

    return matrix


def draw_reading_order(matrix: np.ndarray,
                       page: Union['OcrPage', 'CanonicalPage'],
                       output_path: Optional[str] = None):
    # Compute word centers
    w_centers = [w.bbox.center for w in page.children.words]
    matrix = cv2.polylines(img=matrix,
                           pts=[np.array(w_centers, np.int32).reshape((-1, 1, 2))],
                           isClosed=False,
                           color=(255, 0, 0),
                           thickness=4)
    if output_path:
        cv2.imwrite(output_path, matrix)

    return matrix


def remove_artifacts_from_contours(contours: List[Shape],
                                   artifact_perimeter_threshold: float) -> List[Shape]:
    """Removes contours if the perimeter of their bounding box is inferior to `artifact_perimeter_threshold`"""

    contours_ = [c for c in contours if (2 * (c.width + c.height)) > artifact_perimeter_threshold]
    logger.info(f"""Removed {len(contours) - len(contours_)} artifacts""")

    return contours_


def resize_image(img: np.ndarray,
                 target_height) -> np.ndarray:
    """Resize image to target height while maintaining aspect ratio."""

    scale_percent = target_height / img.shape[0]  # percent of original size
    target_width = int(img.shape[1] * scale_percent)
    dim = target_width, target_height

    return cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_AREA)
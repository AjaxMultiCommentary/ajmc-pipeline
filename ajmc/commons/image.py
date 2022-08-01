import cv2
from typing import List, Tuple, Optional
import numpy as np
from ajmc.commons.geometry import Shape
from ajmc.commons.miscellaneous import lazy_property, RectangleType, get_custom_logger, lazy_init

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

    def crop(self, rect: RectangleType, margin: int = 0) -> 'Image':
        """Gets the slice of `self.matrix` corresponding to `rect`.

        Args:
            rect: The rectangle delimiting the desired crop
            margin: The extra margin desired around `rect`

        Returns:
             A new matrix containing the desired crop.
        """
        cropped = self.matrix[rect[0][1] - margin:rect[2][1] + margin, rect[0][0] - margin:rect[2][0] + margin, :]

        return Image(matrix=cropped)

    def write(self, output_path: str):
        cv2.imwrite(output_path, self.matrix)


def binarize(img_matrix: np.ndarray):
    """Binarizes a cv2 `image`"""
    gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def find_contours(img_matrix: np.ndarray, do_binarize: bool = True, remove_artifacts=True) -> List[Shape]:
    """Binarizes `img_matrix` and finds contours using `cv2.findContours`."""

    temp = binarize(img_matrix) if do_binarize else img_matrix
    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # alternative: CHAIN_APPROX_NONE
    contours = [Shape.from_numpy_array(c) for c in contours if len(c) > 1]

    if remove_artifacts:
        contours = remove_artifacts_from_contours(contours, 0.002 * img_matrix.shape[0])

    return contours


def draw_rectangles(rectangles: List[RectangleType], matrix: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255),
                    thickness: int = 2, output_path: str = None, show: bool = False):
    """Draws a list of rectangles on an image matrix.

    Args:
        rectangles: A list of rectangles.
        matrix: An image matrix to draw on.
        color: A tuple of BGR-color, e.g. (255,109, 118)
        thickness: An integer, see cv2, e.g. 2
        output_path: A path to write the image to.
        show: Whether to display the image.

    Returns:
        np.ndarray: The modified `matrix`

    """

    for rectangle in rectangles:
        matrix = cv2.rectangle(matrix, pt1=tuple(rectangle[0]), pt2=tuple(rectangle[2]), color=color,
                               thickness=thickness)
    if output_path:
        cv2.imwrite(output_path, matrix)

    if show:
        cv2.imshow('image', matrix)

    return matrix


def draw_page_regions_lines_words(matrix: np.ndarray,
                                  page: 'OcrPage',
                                  output_path: Optional[str] = None,
                                  region_elements: bool = False):
    matrix = draw_rectangles(rectangles=[r.bbox.bbox for r in page.children['region']],
                             matrix=matrix,
                             color=(255, 0, 0),
                             thickness=3)
    if region_elements:
        matrix = draw_rectangles([r.bbox.bbox for region in page.children['region'] for r in region.children['line']], matrix,
                                 (0, 255, 0), thickness=2)
        matrix = draw_rectangles([r.bbox.bbox for region in page.children['region'] for r in region.children['word']], matrix,
                                 thickness=1)
    else:
        matrix = draw_rectangles([r.bbox.bbox for r in page.children['line']], matrix, (0, 255, 0), thickness=2)
        matrix = draw_rectangles([r.bbox.bbox for r in page.children['word']], matrix, thickness=1)

    if output_path:
        cv2.imwrite(output_path, matrix)

    return matrix


def draw_reading_order(matrix: np.ndarray,
                       page: 'OcrPage',
                       output_path: Optional[str] = None):
    # Compute word centers
    w_centers = [w.bbox.center for w in page.children['word']]
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
    """Removes contours if the perimeter of their bounding rectangle is inferior to `artifact_perimeter_threshold`"""

    contours_ = [c for c in contours if (2 * (c.width + c.height)) > artifact_perimeter_threshold]
    logger.info(f"""Removed {len(contours) - len(contours_)} artifacts""")

    return contours_

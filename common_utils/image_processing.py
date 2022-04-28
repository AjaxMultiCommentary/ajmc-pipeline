import cv2
from typing import List, Tuple, Optional
import numpy as np
from common_utils.geometry import Shape
import os
from common_utils.general_utils import lazy_property, RectangleType
from common_utils.variables import PATHS


def binarize(img_matrix: np.ndarray):
    """Binarizes a cv2 `image`"""
    gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def find_contours(img_matrix: np.ndarray, do_binarize: bool = True, remove_artifacts=True) -> List[Shape]:
    """Binarizes `img_matrix` and finds contours using `cv2.findContours`."""

    temp = binarize(img_matrix) if do_binarize else img_matrix
    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # alternative: CHAIN_APPROX_NONE
    contours = [Shape.from_numpy_array(c) for c in contours if len(c)>1]

    if remove_artifacts:
        contours = remove_artifacts_from_contours(contours, 0.002*img_matrix.shape[0])

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


def draw_page_regions_lines_words(page, output_path: str,
                                  region_elements:bool=False):
    matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.regions],
                             page.image.matrix.copy(), (255, 0, 0), 3)
    if region_elements:
        matrix = draw_rectangles([r.coords.bounding_rectangle for region in page.regions for r in region.lines], matrix, (0, 255, 0), thickness=2)
        matrix = draw_rectangles([r.coords.bounding_rectangle for region in page.regions for r in region.words], matrix, thickness=1)
    else:
        matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.lines], matrix, (0, 255, 0), thickness=2)
        matrix = draw_rectangles([r.coords.bounding_rectangle for r in page.words], matrix,thickness=1)
    cv2.imwrite(output_path, matrix)


def remove_artifacts_from_contours(contours: List[Shape],
                                   artifact_size_threshold: float) -> List[Shape]:
    """Removes contours if the perimeter of their bounding rectangle is inferior to `artifact_size_threshold`"""

    contours_ = [c for c in contours if (2 * (c.xywh[2] + c.xywh[3])) > artifact_size_threshold]
    print(f"""Removed {len(contours)-len(contours_)} artifacts""")

    return contours_


class Image:
    """Default class for images

    Note:
          The center of `Image`-coordinates is the upper left corner, consistantly with cv2 and numpy. This implies
          that Y-coordinates are ascending towards the bottom of the image.
    """

    def __init__(self, page_id: str, format_: str = 'png', matrix: np.ndarray = None, path: Optional[str] = None):
        self.id = page_id
        self.filename = page_id + '.' + format_
        if matrix is not None:
            self._matrix = matrix
        self.path = path if path else os.path.join(PATHS['base_dir'], self.id.split('_')[0], PATHS['png'], self.filename)

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

        return Image(self.id, matrix=cropped)

    def write(self, output_path: str):
        cv2.imwrite(output_path, self.matrix())



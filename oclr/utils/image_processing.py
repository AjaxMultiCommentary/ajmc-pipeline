import cv2
from typing import List, Tuple
import numpy as np
from oclr.utils.geometry import Shape
import os
from commons.utils import lazy_property
from commons.variables import PATHS
from commons.types import RectangleType, PathType


def binarize(image: np.ndarray):
    """Binarizes a cv2 `image`"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def find_contours(image: np.ndarray, do_binarize: bool = True) -> List[np.ndarray]:
    """Binarizes `image` and finds contours using `cv2.findContours`."""

    temp = binarize(image) if do_binarize else image
    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return [Shape.from_numpy_array(c) for c in contours]


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


class Image:  # todo
    """Default class for images"""

    def __init__(self, page_id: str, format_: str = '.png', matrix: np.ndarray = None):
        self.id = page_id
        self.filename = page_id + format_
        if matrix is not None:
            self._matrix = matrix

    @lazy_property
    def matrix(self) -> np.ndarray:
        """np.ndarray of the image image matrix. Its shape is (height, width, channels)."""
        return cv2.imread(os.path.join(PATHS['base_dir'], self.id.split('_')[0], PATHS['png'], self.filename))


    @lazy_property
    def contours(self):
        return find_contours(self.matrix)

    @lazy_property
    def overlap_matrix(self):  # todo is this still requested.

        overlap_matrix = np.ndarray((4, self.matrix.shape[0], self.matrix.shape[1]), dtype=object)
        overlap_matrix[:, :, :] = ''
        overlap_matrix[3, :, :] = -1

        # Add contours points to overlap_matrix
        for i, contour in enumerate(self.contours):
            for point in contour:
                self.overlap_matrix[3, point[0][1], point[0][0]] = i

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


    def write(self, output_path: PathType):
        cv2.imwrite(output_path, self.matrix())




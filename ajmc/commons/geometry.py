from typing import List, Union, Iterable
import numpy as np

from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.miscellaneous import lazy_property, RectangleType
from ajmc.commons.docstrings import docstring_formatter, docstrings


class Shape:
    """The basic class for contours, bounding rectangles and coordinates.

    `Shape` objects can be instanciated directly or via `Shape.from_points()`, `Shape.from_numpy_array()`
    or `Shape.from_xywh()`. Notice that default constructor expects a list of 4 lists of x-y points, like
    `[[x:int,y:int], ...]`.

    Attributes:

        points (List[List[int]]): a list of list of x,y-points `[[x:int,y:int], ...]`

    """

    def __init__(self, points: Iterable[Iterable[int]] = None):
        """Default constructor.
        
        Args:
            points: an iterable of iterable of points such as `[[x:int,y:int], ...]` or `[(x,y), ...]`.
        """
        self.points = points

    @classmethod
    def from_numpy_array(cls, points: np.ndarray):
        """Creates a Shape from a numpy array of points.

        Args:
             points: Array which can be coherced to a shape (N,2) where N is the number of points
        """

        points = points.squeeze()
        assert points.shape[-1] == 2 and points.shape[
            0] > 1, """The array's shape must be (N,2) or squeezable to (N,2)"""
        return cls(points.tolist())

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int):
        """Creates a Shape from `x`, `y`, `w`, `h`, where `x` and `y` are the coordinates of the upper left corner,
        while `w` and `h` represent width and height respectively."""
        return cls([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    @classmethod
    def from_center_w_h(cls, center_x: int, center_y: int, w: int, h:int ):
        x = center_x - int(w/2)
        y = center_y - int(h/2)
        return cls.from_xywh(x, y, w, h)


    @lazy_property
    @docstring_formatter(**docstrings)
    def bbox(self) -> RectangleType:
        """{rectangle} This format is used internally (preferably to `bbox_2` in order to
        perform operations such as `any([is_point_within_rectangle(p,r) for p in points])`. """
        return get_bbox_from_points(self.points)

    @lazy_property
    def bbox_2(self) -> RectangleType:
        """Bounding rectangle represented by `[[Up-left xy], [bottom-right xy]]`,
        mainly used for memory efficient storage."""
        return self.bbox[::2]

    @lazy_property
    def xywh(self) -> List[int]:
        """Gets the bounding rectangle in `[x,y,w,h]` format, where `x` and `y` are the coordinates of the upper-left
        corner."""
        return [self.bbox[0][0], self.bbox[0][1], self.width, self.height]

    @lazy_property
    def width(self) -> int:
        return self.bbox[2][0] - self.bbox[0][0] + 1

    @lazy_property
    def height(self) -> int:
        return self.bbox[2][1] - self.bbox[0][1] + 1

    @lazy_property
    def center(self) -> List[int]:
        return [int(self.xywh[0] + self.width / 2), int(self.xywh[1] + self.height / 2)]

    @lazy_property
    def area(self) -> int:
        return max(self.width * self.height, 0)


@docstring_formatter(**docstrings)
def get_bbox_from_points(points: Union[np.ndarray, Iterable[Iterable[int]]]) -> RectangleType:
    """Gets the bounding box (i.e. the minimal rectangle containing all points) from a sequence of x-y points.

    Args:
        points: {points}
    
    Returns:
        A list of four lists of x-y-points representing the four points of the rectangle from the upper
        left corner clockwise."""

    if type(points) == np.ndarray:
        points = points.squeeze()
        assert len(points.shape) == 2 and points.shape[-1] == 2, """Points-array must be in 
        the shape (number_of_points, 2)"""
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    else:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def compute_rectangle_area(rectangle: RectangleType) -> int:
    # Todo 👁️ Shouldn't this work directly with a shape object.
    return (rectangle[2][0] - rectangle[0][0] + 1) * (rectangle[2][1] - rectangle[0][1] + 1)


@docstring_formatter(**docstrings)
def is_point_within_rectangle(point: Union[Iterable[int], np.ndarray], rectangle: RectangleType) -> bool:
    """Checks wheter a `point` is contained within a `rectangle`.

    Note:
        Included means included or equal, not strictly included.

    Args:
         point: {point}
         rectangle: {rectangle}
    """
    return all([point[0] >= rectangle[0][0],
                point[1] >= rectangle[0][1],
                point[0] <= rectangle[2][0],
                point[1] <= rectangle[2][1]])


@docstring_formatter(**docstrings)
def is_rectangle_within_rectangle(contained: RectangleType, container: RectangleType) -> bool:
    """Checks whether the `contained` rectangle is entirely contained within the `container` rectangle.

    Note:
        Included means included or equal, not strictly included. For any rectangle `r`, we have
        `is_rectangle_within_rectangle(r, r) == True`.

    Args:
        contained: {rectangle}
        container: {rectangle}
    """

    return all([contained[0][0] >= container[0][0],
                contained[0][1] >= container[0][1],
                contained[2][0] <= container[2][0],
                contained[2][1] <= container[2][1]])


@docstring_formatter(**docstrings)
def compute_overlap_area(r1: RectangleType, r2: RectangleType) -> int:
    """Measures the area of intersection between two rectangles.

    Args:
        r1: {rectangle}
        r2: {rectangle}

    Returns:
        int: The area of intersection
    """
    inter_width = compute_interval_overlap((r1[0][0], r1[2][0]), (r2[0][0], r2[2][0]))
    inter_height = compute_interval_overlap((r1[0][1], r1[2][1]), (r2[0][1], r2[2][1]))

    return inter_width * inter_height


@docstring_formatter(**docstrings)
def are_rectangles_overlapping(r1: RectangleType, r2: RectangleType) -> bool:
    """Checks whether rectangles are overlapping with each other.

    Args:
        r1: {rectangle}
        r2: {rectangle}
    """
    return bool(compute_overlap_area(r1, r2))


@docstring_formatter(**docstrings)
def is_rectangle_within_rectangle_with_threshold(contained: RectangleType, container: RectangleType,
                                                 threshold: float) -> bool:
    """Asserts more than `threshold` of `contained`'s area is within `container`. Is not merged with
    `are_rectangles_overlapping` for effisciency purposes.

    Args:
        contained: {rectangle}
        container: {rectangle}
        threshold: The minimal proportional of `contained` which should be included in `container`.
    """
    contained_area = compute_rectangle_area(contained)
    return compute_overlap_area(contained, container) > threshold * contained_area


@docstring_formatter(**docstrings)
def are_rectangles_overlapping_with_threshold(r1: RectangleType, r2: RectangleType, threshold: float) -> bool:
    """Checks whether the overlapping (intersection) area of two rectangles is higher than `threshold`* union area

    Args:
        r1: {rectangle}
        r2: {rectangle}
        threshold: The minimal proportion of the union area to be included in the intersection area.
    """
    inter_area = compute_overlap_area(r1, r2)
    union_area = compute_rectangle_area(r1) + compute_rectangle_area(r2) - inter_area
    return inter_area >= threshold * union_area


@docstring_formatter(**docstrings)
def adjust_to_included_contours(r: RectangleType,
                                contours: List[Shape]) -> Shape:
    """Finds the contours included in `rectangle` and returns a shape objects that minimally contains them.

    Note:
        This function is mainly used to resize word-boxes. It therefore discards the contours that would make
        `rectangle` taller (i.e. longer on the Y-axis). This is helpful to avoid line-overlapping word-boxes.

    Args:
        r: {rectangle}
        contours: A list of included contours
    """

    included_contours = [c for c in contours
                         if are_rectangles_overlapping(c.bbox, r)
                         and not (c.bbox[0][1] < r[0][1]
                                  or c.bbox[2][1] > r[2][1])]

    if included_contours:  # If we find included contours, readjust the bounding box
        return Shape([xy for c in included_contours for xy in c.bbox])
    else:
        return Shape(r)  # Leave the box untouched # Todo why ?

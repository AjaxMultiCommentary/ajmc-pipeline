from typing import List, Union, Iterable, Optional
import numpy as np
from ajmc.commons.miscellaneous import lazy_property, RectangleType


class Shape:
    """The basic class for contours, rectangles and coordinates.

    `Shape` objects can be instanciated directly or via `Shape.from_points()`, `Shape.from_numpy_array()`
    or `Shape.from_xywh()`. Notice that default constructor expects a list of 4 lists of x-y points, like
    `[[x:int,y:int], ...]`.

    Attributes:
        bounding_rectangle : a list of 4 list of x-y points `[[x:int,y:int], ...]`, representing the four points
        of the rectangle from the upper left corner clockwise.
        points (List[List[int]]): a list of list of x,y-points `[[x:int,y:int], ...]`

    """

    def __init__(self, bounding_rectangle: RectangleType, points: Optional[List[List[int]]] = None):
        """Default constructor.
        
        Args:
            bounding_rectangle : a list of 4 list of x-y points `[[x:int,y:int], ...]`, representing the four points
            of the rectangle from the upper left corner clockwise.
            points: a list of list of points `[[x:int,y:int], ...]`
        """
        self.bounding_rectangle: RectangleType = bounding_rectangle
        self.points = points if points else bounding_rectangle

    @classmethod
    def from_points(cls, points: Iterable[Iterable[int]]):
        return cls(get_bounding_rectangle_from_points(points), points)

    @classmethod
    def from_numpy_array(cls, points: np.ndarray):
        """Creates a Shape from a numpy array of points.

        Args:
             points: Array which can be coherced to a shape (N,2) where N is the number of points
        """

        points = points.squeeze()
        assert points.shape[-1] == 2 and points.shape[
            0] > 1, """The array's shape must be (N,2) or squeezable to (N,2)"""
        points = points.tolist()
        return cls(get_bounding_rectangle_from_points(points), points)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int):
        """Creates a Shape from `x`, `y`, `w`, `h`, where `x` and `y` are the coordinates of the upper left corner,
        while `w` and `h` represent width and height respectively."""
        return cls([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    @lazy_property
    def width(self):
        return self.bounding_rectangle[2][0] - self.bounding_rectangle[0][0]

    @lazy_property
    def height(self):
        return self.bounding_rectangle[2][1] - self.bounding_rectangle[0][1]

    @lazy_property
    def xywh(self) -> List[int]:
        """Gets the bounding rectangle in `[x,y,w,h]` format, where `x` and `y` are the coordinates of the upper-left
        corner."""
        return [self.bounding_rectangle[0][0], self.bounding_rectangle[0][1], self.width, self.height]

    @lazy_property
    def area(self) -> int:
        return max(self.width * self.height, 0)


def get_bounding_rectangle_from_points(points: Union[np.ndarray, Iterable[Iterable[int]]]) -> RectangleType:
    """Gets the bounding rectangle (i.e. the minimal rectangle containing all points) from a sequence of x-y points.

    Args:
        points: A sequence of (x, y) points, e.g. `[(1,2), (1,5), (3,5)]`
    
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
    return (rectangle[2][0] - rectangle[0][0]) * (rectangle[2][1] - rectangle[0][1])


def is_point_within_rectangle(point: Union[Iterable[int], np.ndarray], rectangle: RectangleType) -> bool:
    """Checks wheter a `point` is contained within a `rectangle`."""
    return all([point[0] >= rectangle[0][0],
                point[1] >= rectangle[0][1],
                point[0] <= rectangle[2][0],
                point[1] <= rectangle[2][1]])


def is_rectangle_within_rectangle(contained: RectangleType, container: RectangleType) -> bool:
    """Checks whether a rectangle is entirely contained within another."""

    return all([contained[0][0] >= container[0][0],
                contained[0][1] >= container[0][1],
                contained[2][0] <= container[2][0],
                contained[2][1] <= container[2][1]])


def are_rectangles_overlapping(r1: RectangleType, r2: RectangleType) -> bool:
    """Checks whether rectangles are overlapping with each other."""

    return any([is_point_within_rectangle(p, r2) for p in r1])


def compute_overlap_area(r1: RectangleType, r2: RectangleType) -> int:
    """Measures the area of intersection between two rectangles.

    Args:
        r1: A list of four lists of x,y points.
        r2: A list of four lists of x,y points.

    Returns:
        int: The area of intersection
    """

    inter_width = max(min(r1[2][0], r2[2][0]) - max(r1[0][0], r2[0][0]), 0)
    inter_height = max(min(r1[2][1], r2[2][1]) - max(r1[0][1], r2[0][1]), 0)

    return inter_width * inter_height


def is_rectangle_within_rectangle_with_threshold(contained: RectangleType, container: RectangleType,
                                                 threshold: float) -> bool:
    """Asserts more than `threshold` of `contained`'s area is within `container`. Is not merged with
    `are_rectangles_overlapping` for effisciency purposes. """
    contained_area = compute_rectangle_area(contained)
    return compute_overlap_area(contained, container) > threshold * contained_area


def are_rectangles_overlapping_with_threshold(r1: RectangleType, r2: RectangleType, threshold: float) -> bool:
    """Checks whether the overlapping (intersection) area of two rectangles is higher than `threshold`* union area """
    inter_area = compute_overlap_area(r1, r2)
    union_area = compute_rectangle_area(r1) + compute_rectangle_area(r2) - inter_area
    return inter_area >= threshold * union_area


def adjust_to_included_contours(rectangle: RectangleType,
                                contours: List[Shape]) -> Shape:
    """Finds the contours included in `rectangle` and returns a shape objects that minimally contains them.

    Note:
        This function is mainly used to resize word-boxes. It thereforediscards the contours that would make
        `rectangle` taller (i.e. longer on the Y-axis). This is helpful to avoid line-overlapping word-boxes.
    """

    included_contours = [c for c in contours
                         if are_rectangles_overlapping(c.bounding_rectangle, rectangle)
                         and not (c.bounding_rectangle[0][1] < rectangle[0][1]
                                  or c.bounding_rectangle[2][1] > rectangle[2][1])]

    if included_contours:
        return Shape.from_points([xy for c in included_contours for xy in c.bounding_rectangle])
    else:
        return Shape.from_points(rectangle)

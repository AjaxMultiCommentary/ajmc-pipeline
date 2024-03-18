"""Basic geometrical operations and objects."""

from typing import Iterable, List, Tuple, Union

import numpy as np
from lazy_objects.lazy_objects import lazy_property

from ajmc.commons import variables
from ajmc.commons.arithmetic import compute_interval_overlap
from ajmc.commons.docstrings import docstring_formatter, docstrings


class Shape:
    """The basic class for contours, bounding boxes and coordinates.

    ``Shape`` objects can be instanciated directly from points. Other constructors are ``Shape.from_numpy_array()``,
    ``Shape.from_center_w_h()``and ``Shape.from_xywh()``.
    """

    @docstring_formatter(**docstrings)
    def __init__(self, points: Iterable[Iterable[int]] = None):
        """Default constructor.
        
        Args:
            points: {points}
        """
        self.points = points

    @classmethod
    @docstring_formatter(**docstrings)
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
        """Creates a Shape from ``x``, ``y``, ``w``, ``h``, where ``x`` and ``y`` are the coordinates of the upper left corner,
        while ``w`` and ``h`` represent width and height respectively."""
        return cls([(x, y), (x + w, y + h)])

    @classmethod
    def from_xxyy(cls, x1: int, x2: int, y1: int, y2: int):
        """Creates a Shape from ``x1``, ``x2``, ``y1``, ``y2``, where ``x1`` and ``y1`` are the coordinates of the upper left corner,
        while ``x2`` and ``y2`` represent the coordinates of the lower right corner respectively."""
        return cls([(x1, y1), (x2, y2)])

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int):
        """Creates a Shape from ``x1``, ``y1``, ``x2``, ``y2``, where ``x1`` and ``y1`` are the coordinates of the upper left corner,
        while ``x2`` and ``y2`` represent the coordinates of the lower right corner respectively."""
        return cls([(x1, y1), (x2, y2)])

    @classmethod
    def from_center_w_h(cls, center_x: int, center_y: int, w: int, h: int):
        """Creates a Shape from ``center_x``, ``center_y``, ``w``, ``h``, where ``center_x`` and ``center_y`` are the coordinates
        of the center and ``w`` and ``h`` represent width and height respectively."""
        x = center_x - int(w / 2)
        y = center_y - int(h / 2)
        return cls.from_xywh(x, y, w, h)

    @classmethod
    def from_via(cls, via_region_dict: dict):
        """Creates a Shape from a VIA dictionary."""
        return cls.from_xywh(via_region_dict['shape_attributes']['x'],
                             via_region_dict['shape_attributes']['y'],
                             via_region_dict['shape_attributes']['width'],
                             via_region_dict['shape_attributes']['height'])


    @lazy_property
    @docstring_formatter(**docstrings)
    def bbox(self) -> variables.BoxType:
        """{bbox}"""
        return get_bbox_from_points(self.points)

    @lazy_property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Returns the coordinates of the upper left corner and the lower right corner of the bounding box."""
        return self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1]

    @lazy_property
    def xmin(self) -> int:
        """Returns the x-coordinate of the upper left corner."""
        return self.bbox[0][0]

    @lazy_property
    def ymin(self) -> int:
        """Returns the y-coordinate of the upper left corner."""
        return self.bbox[0][1]

    @lazy_property
    def xmax(self) -> int:
        """Returns the x-coordinate of the lower right corner."""
        return self.bbox[1][0]

    @lazy_property
    def ymax(self) -> int:
        """Returns the y-coordinate of the lower right corner."""
        return self.bbox[1][1]


    @lazy_property
    def xywh(self) -> Tuple[int, int, int, int]:
        """Gets the bounding box in ``[x,y,w,h]`` format, where ``x`` and ``y`` are the coordinates of the upper-left
        corner."""
        return self.bbox[0][0], self.bbox[0][1], self.width, self.height

    @lazy_property
    def width(self) -> int:
        return self.bbox[1][0] - self.bbox[0][0] + 1

    @lazy_property
    def height(self) -> int:
        return self.bbox[1][1] - self.bbox[0][1] + 1

    @lazy_property
    def center(self) -> Tuple[int, int]:
        return int(self.bbox[0][0] + self.width / 2), int(self.bbox[0][1] + self.height / 2)

    @lazy_property
    def area(self) -> int:
        return max(self.width * self.height, 0)


@docstring_formatter(**docstrings)
def get_bbox_from_points(points: Union[np.ndarray, Iterable[Iterable[int]]]) -> variables.BoxType:
    """Gets the bounding box (i.e. the minimal rectangle containing all points) from a sequence of x-y points.

    Args:
        points: {points}
    
    Returns:
        {bbox}"""

    if type(points) == np.ndarray:
        points = points.squeeze()
        assert len(points.shape) == 2 and points.shape[-1] == 2, """Points-array must be in 
        the shape (number_of_points, 2)"""
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        return (x_min, y_min), (x_max, y_max)

    else:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (x_min, y_min), (x_max, y_max)


def compute_bbox_area(bbox: variables.BoxType) -> int:
    return (bbox[1][0] - bbox[0][0] + 1) * (bbox[1][1] - bbox[0][1] + 1)


@docstring_formatter(**docstrings)
def is_point_within_bbox(point: Union[Iterable[int], np.ndarray],
                         bbox: variables.BoxType) -> bool:
    """Checks wheter a ``point`` is contained within a ``bbox``.

    Note:
        Included means included or equal, not strictly included.

    Args:
         point: {point}
         bbox: {bbox}
    """
    return all([point[0] >= bbox[0][0],
                point[1] >= bbox[0][1],
                point[0] <= bbox[1][0],
                point[1] <= bbox[1][1]])


@docstring_formatter(**docstrings)
def is_bbox_within_bbox(contained: variables.BoxType,
                        container: variables.BoxType) -> bool:
    """Checks whether the ``contained`` bbox is entirely contained within the ``container`` bbox.

    Note:
        Included means included or equal, not strictly included. For any bbox ``r``, we have
        ``is_bbox_within_bbox(r, r) == True``.

    Args:
        contained: {bbox}
        container: {bbox}
    """

    return all([contained[0][0] >= container[0][0],
                contained[0][1] >= container[0][1],
                contained[1][0] <= container[1][0],
                contained[1][1] <= container[1][1]])


@docstring_formatter(**docstrings)
def compute_bbox_overlap_area(bbox1: variables.BoxType,
                              bbox2: variables.BoxType) -> int:
    """Measures the area of intersection between two bboxes.

    Args:
        bbox1: {bbox}
        bbox2: {bbox}

    Returns:
        int: The area of intersection
    """
    inter_width = compute_interval_overlap((bbox1[0][0], bbox1[1][0]), (bbox2[0][0], bbox2[1][0]))
    inter_height = compute_interval_overlap((bbox1[0][1], bbox1[1][1]), (bbox2[0][1], bbox2[1][1]))

    return inter_width * inter_height


@docstring_formatter(**docstrings)
def are_bboxes_overlapping(bbox1: variables.BoxType,
                           bbox2: variables.BoxType) -> bool:
    """Checks whether bboxes are overlapping with each other.

    Args:
        bbox1: {bbox}
        bbox2: {bbox}
    """
    return bool(compute_bbox_overlap_area(bbox1, bbox2))


@docstring_formatter(**docstrings)
def is_bbox_within_bbox_with_threshold(contained: variables.BoxType,
                                       container: variables.BoxType,
                                       threshold: float) -> bool:
    """Asserts more than ``threshold`` of ``contained``'s area is within ``container``. Is not merged with
    ``are_bboxes_overlapping`` for effisciency purposes.

    Args:
        contained: {bbox}
        container: {bbox}
        threshold: The minimal proportional of ``contained`` which should be included in ``container``.
    """
    contained_area = compute_bbox_area(contained)
    return compute_bbox_overlap_area(contained, container) > threshold * contained_area


@docstring_formatter(**docstrings)
def are_bboxes_overlapping_with_threshold(bbox1: variables.BoxType,
                                          bbox2: variables.BoxType,
                                          threshold: float) -> bool:
    """Checks whether the overlapping (intersection) area of two bboxes is higher than ``threshold`` x union area.

    Args:
        bbox1: {bbox}
        bbox2: {bbox}
        threshold: The minimal proportion of the union area to be included in the intersection area.
    """
    inter_area = compute_bbox_overlap_area(bbox1, bbox2)
    union_area = compute_bbox_area(bbox1) + compute_bbox_area(bbox2) - inter_area
    return inter_area >= threshold * union_area


@docstring_formatter(**docstrings)
def adjust_bbox_to_included_contours(bbox: variables.BoxType,
                                     contours: List[Shape]) -> Shape:
    """Finds the contours included in ``bbox`` and returns a shape objects that minimally contains them.

    Note:
        This function is mainly used to resize word-boxes. It therefore discards the contours that would make
        ``bbox`` taller (i.e. longer on the Y-axis). This is helpful to avoid line-overlapping word-boxes.

    Args:
        bbox: {bbox}
        contours: A list of included contours
    """

    included_contours = [c for c in contours
                         if are_bboxes_overlapping(c.bbox, bbox)
                         and not (c.bbox[0][1] < bbox[0][1]
                                  or c.bbox[1][1] > bbox[1][1])]

    if included_contours:  # If we find included contours, readjust the bounding box
        return Shape([xy for c in included_contours for xy in c.bbox])
    else:
        return Shape(bbox)  # Leave the box untouched

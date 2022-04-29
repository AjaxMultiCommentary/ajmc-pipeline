import numpy as np
import pytest

import common_utils.geometry as geo
from text_importation.classes import Page


@pytest.fixture
def points():
    return {'base': [(0, 2), (2, 2), (1, 1), (2, 0), (0, 0)],
            'included': [(0, 1), (1, 1), (1, 1), (1, 0), (0, 0)],
            'overlapping': [[1, 3], [3, 3], [2, 2], [3, 1], [1, 1]],
            'line': [(0, 0), (1, 1), (2, 2)],
            'non_overlapping': [(5, 7), (7, 7), (6, 6), (7, 5), (5, 5)]
            }


@pytest.fixture
def rectangles(points):
    return {k: geo.get_bounding_rectangle_from_points(v) for k, v in points.items()}


# points_np = np.array(points_1)
# shape = geo.Shape.from_points(points_1)
# rect = np.array([[1, 1], [5, 7]])
# point = np.array([3, 3])
# page = Page("sophoclesplaysa05campgoog_0146")


def test_shape(points):
    shape = geo.Shape.from_points(points['base'])
    # Test `width` and `height` attributes
    assert shape.width == shape.height == 2
    # Test `xywh` and `area` attributes
    assert shape.xywh[2] * shape.xywh[3] == shape.area == 4


def test_get_bounding_rectangle_from_points(points, rectangles):
    # Assert the rectangle is actually what we want it to be
    assert rectangles['base'] == [points['base'][0], points['base'][1], points['base'][3], points['base'][4]]
    # Test with numpy array of points
    assert rectangles['base'] == geo.get_bounding_rectangle_from_points(np.array(points['base']))
    # Test a non-rectangular sequence of points
    assert geo.get_bounding_rectangle_from_points(points['line']) == rectangles['base']


def test_is_point_within_rectangle(rectangles):
    # Test with an included and a non-included point
    assert geo.is_point_within_rectangle(point=(1, 2), rectangle=rectangles['base'])
    assert not geo.is_point_within_rectangle(point=(4, 5), rectangle=rectangles['base'])


def test_is_rectangle_within_rectangle(rectangles):
    # Test with an included and a non-included rectangle
    assert geo.is_rectangle_within_rectangle(container=rectangles['base'], contained=rectangles['included'])
    assert not geo.is_rectangle_within_rectangle(container=rectangles['base'], contained=rectangles['overlapping'])


def test_are_rectangles_overlapping(rectangles):
    # Test with an overlapping and a non-overlapping rectangle
    assert geo.are_rectangles_overlapping(rectangles['base'], rectangles['overlapping'])
    assert not geo.are_rectangles_overlapping(rectangles['base'], rectangles['non_overlapping'])


def test_measure_overlap_area(rectangles):
    # Test with an overlapping and a non-overlapping rectangle
    assert geo.measure_overlap_area(rectangles['base'], rectangles['included']) == 1
    assert geo.measure_overlap_area(rectangles['base'], rectangles['base']) == 4
    assert geo.measure_overlap_area(rectangles['base'], rectangles['non_overlapping']) == 0


def test_are_rectangles_overlapping_with_threshold(rectangles):
    # Test with two different thresholds with overlapping rects
    assert geo.are_rectangles_overlapping(rectangles['base'], rectangles['overlapping'], 0.20)
    assert not geo.are_rectangles_overlapping(rectangles['base'], rectangles['overlapping'], 0.25)
    # Test with non overlapping rectangles
    assert not geo.are_rectangles_overlapping(rectangles['base'], rectangles['non_overlapping'], 0.1)


def test_shrink_to_included_contours():
    # todo
    pass
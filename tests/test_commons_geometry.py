import numpy as np
import pytest
import commons.geometry as geo



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


def test_compute_rectangle_area(rectangles):
    assert geo.compute_rectangle_area(rectangles['base'])==4

def test_measure_overlap_area(rectangles):
    # Test with an overlapping and a non-overlapping rectangle
    assert geo.compute_overlap_area(rectangles['base'], rectangles['included']) == 1
    assert geo.compute_overlap_area(rectangles['base'], rectangles['base']) == 4
    assert geo.compute_overlap_area(rectangles['base'], rectangles['non_overlapping']) == 0


def test_is_rectangle_within_rectangle_with_threshold(rectangles):
    # Test with two different thresholds with overlapping rects
    assert geo.is_rectangle_within_rectangle_with_threshold(rectangles['base'], rectangles['overlapping'], 0.20)
    assert not geo.is_rectangle_within_rectangle_with_threshold(rectangles['base'], rectangles['overlapping'], 0.25)
    # Test with non overlapping rectangles
    assert not geo.is_rectangle_within_rectangle_with_threshold(rectangles['base'], rectangles['non_overlapping'], 0.1)

def test_are_rectangles_overlapping_with_threshold(rectangles):
    assert geo.are_rectangles_overlapping_with_threshold(rectangles['base'], rectangles['overlapping'], 1/7)
    assert not geo.are_rectangles_overlapping_with_threshold(rectangles['base'], rectangles['overlapping'], 1/4)
    assert not geo.are_rectangles_overlapping_with_threshold(rectangles['base'], rectangles['non_overlapping'], 1 / 4)

def test_shrink_to_included_contours(points, rectangles):
    # Make sure it takes only horizontally overlappping shapes
    contours_1 = [geo.Shape.from_points(points[k]) for k in ['base', 'overlapping', 'horizontally_overlapping']]
    contours_2 = [geo.Shape.from_points(points[k]) for k in ['base', 'horizontally_overlapping']]
    assert geo.shrink_to_included_contours(rectangles['base'], contours_1).bounding_rectangle == \
           geo.shrink_to_included_contours(rectangles['base'], contours_2).bounding_rectangle

import numpy as np
import ajmc.commons.geometry as geo
import tests.sample_objects as so

points = so.sample_points
bboxes = so.sample_bboxes

def test_shape():
    shape = geo.Shape(points['base'])
    # Test `width` and `height` attributes
    assert shape.width == shape.height == 3
    # Test `xywh` and `area` attributes
    assert shape.xywh[2] * shape.xywh[3] == shape.area == 9


def test_get_bbox_from_points():
    # Assert the bbox is actually what we want it to be
    assert bboxes['base'] == [points['base'][0], points['base'][1], points['base'][3], points['base'][4]]
    # Test with numpy array of points
    assert bboxes['base'] == geo.get_bbox_from_points(np.array(points['base']))
    # Test a non-rectangular sequence of points
    assert geo.get_bbox_from_points(points['line']) == bboxes['base']


def test_is_point_within_bbox():
    # Test with an included and a non-included point
    assert geo.is_point_within_bbox(point=(1, 2), bbox=bboxes['base'])
    assert not geo.is_point_within_bbox(point=(4, 5), bbox=bboxes['base'])


def test_is_bbox_within_bbox():
    # Test with an included and a non-included bbox
    assert geo.is_bbox_within_bbox(container=bboxes['base'], contained=bboxes['included'])
    assert not geo.is_bbox_within_bbox(container=bboxes['base'], contained=bboxes['overlapping'])


def test_are_bboxes_overlapping():
    # Test with an overlapping and a non-overlapping bbox
    assert geo.are_bboxes_overlapping(bboxes['base'], bboxes['overlapping'])
    assert not geo.are_bboxes_overlapping(bboxes['base'], bboxes['non_overlapping'])


def test_compute_bbox_area():
    assert geo.compute_bbox_area(bboxes['base']) == 9

def test_measure_overlap_area():
    # Test with an overlapping and a non-overlapping bbox
    assert geo.compute_bbox_overlap_area(bboxes['base'], bboxes['included']) == 4
    assert geo.compute_bbox_overlap_area(bboxes['base'], bboxes['base']) == 9
    assert geo.compute_bbox_overlap_area(bboxes['base'], bboxes['non_overlapping']) == 0


def test_is_bbox_within_bbox_with_threshold():
    # Test with two different thresholds with overlapping rects
    assert geo.is_bbox_within_bbox_with_threshold(bboxes['base'], bboxes['overlapping'], 0.20)
    assert not geo.is_bbox_within_bbox_with_threshold(bboxes['base'], bboxes['overlapping'], 5 / 9)
    # Test with non overlapping bboxes
    assert not geo.is_bbox_within_bbox_with_threshold(bboxes['base'], bboxes['non_overlapping'], 0.1)

def test_are_bboxes_overlapping_with_threshold():
    assert geo.are_bboxes_overlapping_with_threshold(bboxes['base'], bboxes['overlapping'], 4 / 14)
    assert not geo.are_bboxes_overlapping_with_threshold(bboxes['base'], bboxes['overlapping'], 5 / 14)
    assert not geo.are_bboxes_overlapping_with_threshold(bboxes['base'], bboxes['non_overlapping'], 0.1)

def test_shrink_to_included_contours():
    # Make sure it takes only horizontally overlappping shapes
    contours_1 = [geo.Shape(points[k]) for k in ['base', 'overlapping', 'horizontally_overlapping']]
    contours_2 = [geo.Shape(points[k]) for k in ['base', 'horizontally_overlapping']]
    assert geo.adjust_bbox_to_included_contours(bboxes['base'], contours_1).bbox == \
           geo.adjust_bbox_to_included_contours(bboxes['base'], contours_2).bbox

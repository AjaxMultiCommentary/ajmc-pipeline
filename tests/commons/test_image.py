import numpy as np
import pytest
from tests import sample_objects as so
from ajmc.commons import image as img
from ajmc.commons.geometry import Shape


def test_find_contours():
    contours = img.find_contours(so.sample_image.matrix, remove_artifacts=False)
    assert type(contours) == list
    assert all([isinstance(x, Shape) for x in contours])


@pytest.mark.parametrize('art_size', [0.1, 0.01, 0.001])
def test_remove_artifacts_from_contours(art_size):
    artifact_size = art_size * so.sample_image.matrix.shape[0]
    contours = img.find_contours(so.sample_image.matrix, remove_artifacts=False)
    contours_ = img.remove_artifacts_from_contours(contours, artifact_size)
    assert len(contours_) <= len(contours)


def test_image():
    assert isinstance(so.sample_image.matrix, np.ndarray)
    assert isinstance(so.sample_image.crop(so.sample_bboxes['base']), img.Image)


def test_draw_bboxes():
    matrix = so.sample_image.matrix.copy()
    assert so.sample_image.matrix.shape == img.draw_bboxes([r for _, r in so.sample_bboxes.items()],
                                                        matrix).shape

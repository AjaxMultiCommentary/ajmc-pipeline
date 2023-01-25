import numpy as np
import pytest

from ajmc.commons import image as img
from ajmc.commons.geometry import Shape
from tests import sample_objects as so


def test_find_contours():
    contours = img.find_contours(so.sample_img.matrix)
    assert type(contours) == list
    assert all([isinstance(x, Shape) for x in contours])


@pytest.mark.parametrize('art_size', [0.1, 0.01, 0.001])
def test_remove_artifacts_from_contours(art_size):
    artifact_size = art_size * so.sample_img.matrix.shape[0]
    contours = img.find_contours(so.sample_img.matrix)
    contours_ = img.remove_artifacts_from_contours(contours, artifact_size)
    assert len(contours_) <= len(contours)


def test_image():
    assert isinstance(so.sample_img.matrix, np.ndarray)
    assert isinstance(so.sample_img.crop(so.sample_bboxes['base']), img.AjmcImage)

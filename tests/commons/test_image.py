import numpy as np
import pytest

from ajmc.commons import image as img
from ajmc.commons.geometry import Shape
import cv2


@pytest.fixture
def img_matrix(sample_image_path):
    return cv2.imread(sample_image_path)


def test_find_contours(img_matrix):
    contours = img.find_contours(img_matrix, remove_artifacts=False)
    assert type(contours) == list
    assert all([isinstance(x, Shape) for x in contours])


@pytest.mark.parametrize('art_size', [0.1, 0.01, 0.001])
def test_remove_artifacts_from_contours(img_matrix, art_size):
    artifact_size = art_size * img_matrix.shape[0]
    contours = img.find_contours(img_matrix, remove_artifacts=False)
    contours_ = img.remove_artifacts_from_contours(contours, artifact_size)
    assert len(contours_) <= len(contours)


def test_image(sample_image_path, rectangles):
    image = img.Image(path=sample_image_path)
    assert isinstance(image.matrix, np.ndarray)
    assert isinstance(image.crop(rectangles['base']), img.Image)

def test_draw_rectangles(img_matrix, rectangles):
    assert img_matrix.shape == img.draw_rectangles([r for _,r in rectangles.items()],
                                                   img_matrix).shape


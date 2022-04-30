import os

import pytest
from commons import geometry as geo


@pytest.fixture(scope="session")
def points():
    return {'base': [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)],
            'included': [(0, 0), (1, 0), (1, 1), (0, 1)],
            'overlapping': [[1, 1], [3, 1], [2, 2], [3, 3], [1, 3]],
            'non_overlapping': [(5, 5), (7, 5), (6, 6), (7, 7), (5, 7)],
            'line': [(0, 0), (1, 1), (2, 2)],
            'horizontally_overlapping': [(1, 0), (3, 0), (3, 2), (1, 2)],
            }


@pytest.fixture(scope="session")
def rectangles(points):
    return {k: geo.get_bounding_rectangle_from_points(v) for k, v in points.items()}


@pytest.fixture(scope="session")
def img_path():
    return os.path.join('../data/test_image.png')


@pytest.fixture(scope="session")
def sample_tsv_path():
    # return "/Users/matteo/Documents/AjaxMultiCommentary/HIPE2022-corpus/data/release/v2.0/HIPE-2022-v2.0-ajmc-dev-en.tsv"
    return '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/epibau/EpibauCorpus/data/release/v0.3/EpiBau-data-v0.3-test.tsv'


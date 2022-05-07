import os
from commons import variables
import pytest
from commons import geometry as geo


@pytest.fixture()
def test_base_dir():
    return variables.PATHS['base_dir']


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
def sample_tsv_path():
    # return "/Users/matteo/Documents/AjaxMultiCommentary/HIPE2022-corpus/data/release/v2.0/HIPE-2022-v2.0-ajmc-dev-en.tsv"
    return '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/epibau/EpibauCorpus/data/release/v0.3/EpiBau-data-v0.3-test.tsv'

@pytest.fixture()
def sample_commentary_id():
    return 'cu31924087948174'

@pytest.fixture()
def sample_page_id(sample_commentary_id):
    return sample_commentary_id+'_0083'

@pytest.fixture()
def sample_via_path(test_base_dir, sample_commentary_id):
    return os.path.join(test_base_dir, sample_commentary_id, variables.PATHS['via_path'])

@pytest.fixture
def sample_ocr_run():
    return '2480ei_greek-english_porson_sophoclesplaysa05campgoog'

@pytest.fixture()
def sample_ocr_dir(test_base_dir, sample_commentary_id, sample_ocr_run):
    return os.path.join(test_base_dir, sample_commentary_id, variables.PATHS['ocr'], sample_ocr_run, 'outputs' )

@pytest.fixture()
def sample_ocr_path(sample_ocr_dir, sample_page_id):
    return os.path.join(sample_ocr_dir, sample_page_id+'.hocr')

@pytest.fixture()
def sample_groundtruth_dir(test_base_dir, sample_commentary_id):
    return os.path.join(test_base_dir, sample_commentary_id, variables.PATHS['groundtruth'])

@pytest.fixture()
def sample_groundtruth_path(sample_groundtruth_dir, sample_page_id):
    return os.path.join(sample_groundtruth_dir, sample_page_id + '.hmtl')

@pytest.fixture()
def sample_image_dir(test_base_dir, sample_commentary_id):
    return os.path.join(test_base_dir, sample_commentary_id, variables.PATHS['png'])

@pytest.fixture()
def sample_image_path(sample_image_dir, sample_page_id):
    return os.path.join(sample_image_dir, sample_page_id + '.png')
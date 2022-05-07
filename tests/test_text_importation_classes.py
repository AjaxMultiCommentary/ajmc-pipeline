import os

from ajmc.text_importation import classes
import pytest
from ajmc.commons import image


@pytest.fixture()
def commentary_from_paths(sample_commentary_id,
                          sample_ocr_dir,
                          sample_image_dir,
                          sample_via_path,
                          sample_groundtruth_dir):
    return classes.Commentary(commentary_id=sample_commentary_id,
                              ocr_dir=sample_ocr_dir,
                              via_path=sample_via_path,
                              image_dir=sample_image_dir,
                              groundtruth_dir=sample_groundtruth_dir)


@pytest.fixture()
def commentary_from_structure(sample_ocr_dir):
    return classes.Commentary.from_structure(ocr_dir=sample_ocr_dir)


def test_commentary(commentary_from_structure, commentary_from_paths, sample_ocr_dir, sample_groundtruth_dir):
    for comm in [commentary_from_paths, commentary_from_structure]:
        # Test ocr ocr_format
        assert type(comm.ocr_format) == str

        # test Commentary.pages
        assert all([isinstance(p, classes.Page) for p in comm.pages])
        assert len(comm.pages) == len([f for f in os.listdir(sample_ocr_dir) if comm.id in f])

        # Test Commentary.groundtruth_pages
        assert all([isinstance(p, classes.Page) for p in comm.ocr_groundtruth_pages])
        assert len(comm.ocr_groundtruth_pages) == len([f for f in os.listdir(sample_groundtruth_dir) if comm.id in f])

        # Test Commentary.olr_groundtruth_pages
        assert all([isinstance(p, classes.Page) for p in comm.olr_groundtruth_pages])

        # See test_page() for regions, lines, words

        # Test Commentary.via_project
        assert type(comm.via_project) == dict



def test_page(sample_ocr_path,
              sample_page_id,
              sample_groundtruth_path,
              sample_image_path,
              sample_via_path,
              commentary_from_paths):

    page_from_paths = classes.Page(ocr_path=sample_ocr_path,
                                   page_id=sample_page_id,
                                   groundtruth_path=sample_groundtruth_path,
                                   image_path=sample_image_path,
                                   via_path=sample_via_path,
                                   commentary=commentary_from_paths)

    page_from_structure = classes.Page.from_structure(sample_ocr_path, commentary_from_paths)

    for page in [page_from_paths, page_from_structure]:

        assert isinstance(page.ocr_format, str)

        assert isinstance(page.image, image.Image)

        assert all([isinstance(r, classes.Region) for r in page.regions])
        assert all([isinstance(l, classes.TextElement) for l in page.lines])
        assert all([isinstance(w, classes.TextElement) for w in page.words])


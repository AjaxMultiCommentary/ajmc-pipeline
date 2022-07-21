import json
import os
import re

from tests import sample_objects as so
import jsonschema

from ajmc.text_importation import classes
import pytest
from ajmc.commons import image
from ajmc.commons import variables

commentary_from_paths = classes.OcrCommentary(id=so.sample_commentary_id,
                                              ocr_dir=so.sample_ocr_dir,
                                              via_path=so.sample_via_path,
                                              image_dir=so.sample_image_dir,
                                              groundtruth_dir=so.sample_groundtruth_dir)


commentary_from_structure = classes.OcrCommentary.from_ajmc_structure(ocr_dir=so.sample_ocr_dir)


def test_ocrcommentary():
    for comm in [commentary_from_paths, commentary_from_structure]:
        # test OcrCommentary.pages
        assert all([isinstance(p, classes.OcrPage) for p in comm.pages])
        assert len(comm.pages) == len([f for f in os.listdir(so.sample_ocr_dir) if comm.id in f])

        # Test OcrCommentary.groundtruth_pages
        assert all([isinstance(p, classes.OcrPage) for p in comm.ocr_groundtruth_pages])
        assert len(comm.ocr_groundtruth_pages) == len([f for f in os.listdir(so.sample_groundtruth_dir) if comm.id in f])

        # See test_page() for regions, lines, words

        # Test OcrCommentary.via_project
        assert type(comm.via_project) == dict


def test_ocrcommentary_to_canonical():

    for ocr_p, can_p in zip(so.sample_ocrcommentary.pages, so.sample_cancommentary.children['page']):
        # Assert each canonical page has the same number of words, minus those which have been deleted on purpose
        ocr_ws = [w for w in ocr_p.words if re.sub(r'\s+', '', w.text) != '']  # as we are deleting empty words
        assert len(ocr_ws) == len(can_p.words)  # set because reading order might have changed

        # Assert images stay the same
        assert len(can_p.images) == 1
        assert ocr_p.image.id == can_p.images[0].id


def test_ocrpage():
    page = classes.OcrPage(ocr_path=so.sample_ocr_page_path,
                           id=so.sample_page_id,
                           image_path=so.sample_image_path,
                           commentary=commentary_from_paths)

    assert isinstance(page.ocr_format, str)

    assert isinstance(page.image, image.Image)

    assert all([isinstance(r, classes.OlrRegion) for r in page.regions])
    assert all([isinstance(l, classes.OcrLine) for l in page.lines])
    assert all([isinstance(w, classes.OcrWord) for w in page.words])

    # Validate page.json
    with open(os.path.join('../..', variables.PATHS['schema']), 'r') as file:
        schema = json.loads(file.read())

    jsonschema.validate(instance=page.canonical_data, schema=schema)



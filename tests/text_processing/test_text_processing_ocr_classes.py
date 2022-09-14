import json
import os
import re
from ajmc.commons.image import Image
from tests import sample_objects as so
import jsonschema
from ajmc.text_processing import ocr_classes
from ajmc.commons import image
from ajmc.commons import variables

commentary_from_paths = ocr_classes.OcrCommentary(id=so.sample_commentary_id,
                                                  ocr_dir=so.sample_ocr_dir,
                                                  via_path=so.sample_via_path,
                                                  image_dir=so.sample_image_dir,
                                                  groundtruth_dir=so.sample_groundtruth_dir)

commentary_from_structure = ocr_classes.OcrCommentary.from_ajmc_structure(ocr_dir=so.sample_ocr_dir)


def test_ocrcommentary():
    for comm in [commentary_from_paths, commentary_from_structure]:
        # test OcrCommentary.pages
        assert all([isinstance(p, ocr_classes.OcrPage) for p in comm.pages])
        assert len(comm.pages) == len([f for f in os.listdir(so.sample_ocr_dir) if comm.id in f])

        # Test OcrCommentary.groundtruth_pages
        assert all([isinstance(p, ocr_classes.OcrPage) for p in comm.ocr_groundtruth_pages])
        assert len(comm.ocr_groundtruth_pages) == len(
            [f for f in os.listdir(so.sample_groundtruth_dir) if comm.id in f])

        # See test_page() for regions, lines, words

        # Test OcrCommentary.via_project
        assert type(comm.via_project) == dict


def test_ocrcommentary_to_canonical():
    assert len(so.sample_cancommentary.children['page']) == len(so.sample_ocrcommentary.pages)
    for ocr_p, can_p in zip(so.sample_ocrcommentary.pages, so.sample_cancommentary.children['page']):
        # Assert images stay the same
        assert isinstance(can_p.image, Image)
        assert ocr_p.image.id == can_p.image.id

        # Assert each canonical page has the same number of words, minus those which have been deleted on purpose
        ocr_p.optimise()
        assert len(ocr_p.children['region']) == len(can_p.children['region'])
        for ocr_r, can_r in zip(ocr_p.children['region'], can_p.children['region']):

            if ocr_r.bbox.xywh != can_r.bbox.xywh:
                print('ocr_type: ', ocr_r.region_type)
                print('ocr_text: ', ocr_r.text[:50])
                print('can_type: ', can_r.info['region_type'])
                print('can_text: ', can_r.text[:50])

            assert len(ocr_r.children['line']) == len(can_r.children['line'])
            for ocr_l, can_l in zip(ocr_p.children['line'], can_p.children['line']):
                assert ocr_l.bbox.xywh == can_l.bbox.xywh

                assert len(ocr_l.children['word']) == len(can_l.children['word'])
                for ocr_w, can_w in zip(ocr_p.children['word'], can_p.children['word']):
                    assert ocr_w.bbox.xywh == can_w.bbox.xywh
                    assert ocr_w.text == can_w.text

        ocr_p.reset()


def test_ocrpage():
    page = ocr_classes.OcrPage(ocr_path=so.sample_ocr_page_path,
                               id=so.sample_page_id,
                               image_path=so.sample_image_path,
                               commentary=commentary_from_paths)

    assert isinstance(page.ocr_format, str)

    assert isinstance(page.image, image.Image)

    assert all([isinstance(r, ocr_classes.OlrRegion) for r in page.children['region']])
    assert all([isinstance(l, ocr_classes.OcrLine) for l in page.children['line']])
    assert all([isinstance(w, ocr_classes.OcrWord) for w in page.children['word']])

    # Validate page.json
    with open(os.path.join('..', variables.PATHS['schema']), 'r') as file:
        schema = json.loads(file.read())

    jsonschema.validate(instance=page.canonical_data, schema=schema)

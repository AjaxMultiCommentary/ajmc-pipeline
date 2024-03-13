import json
import os

import jsonschema

from ajmc.commons import image, variables
from tests import sample_objects as so


def test_ocrcommentary():
    # test RawCommentary.children
    assert all([isinstance(p, ocr_classes.RawPage) for p in so.sample_ocrcommentary.children.pages])
    assert len(so.sample_ocrcommentary.children.pages) == len(list(so.sample_ocr_run_outputs_dir.glob('*.hocr')))
    assert all([isinstance(r, ocr_classes.RawRegion) for r in so.sample_ocrcommentary.children.regions])
    assert all([isinstance(l, ocr_classes.RawLine) for l in so.sample_ocrcommentary.children.lines])
    assert all([isinstance(w, ocr_classes.RawWord) for w in so.sample_ocrcommentary.children.words])

    # test RawCommentary.images
    assert all([isinstance(i, image.AjmcImage) for i in so.sample_ocrcommentary.images])

    # Test RawCommentary.groundtruth_pages
    assert all([isinstance(p, ocr_classes.RawPage) for p in so.sample_ocrcommentary.ocr_gt_pages])
    assert len(so.sample_ocrcommentary.ocr_gt_pages) == len(
            [f for f in os.listdir(so.sample_ocr_gt_dir) if so.sample_ocrcommentary.id in f])

    # See test_page() for regions, lines, words

    # Test RawCommentary.via_project
    assert type(so.sample_ocrcommentary.via_project) == dict


def test_ocrcommentary_to_canonical():
    assert len(so.sample_cancommentary.children.pages) == len(so.sample_ocrcommentary.children.pages)
    for ocr_p, can_p in zip(so.sample_ocrcommentary.children.pages, so.sample_cancommentary.children.pages):
        # Assert images stay the same
        assert isinstance(can_p.image, image.AjmcImage)
        assert ocr_p.image.id == can_p.image.id

        # Assert each canonical page has the same number of words, minus those which have been deleted on purpose
        ocr_p.optimise()
        assert len(ocr_p.children.regions) == len(can_p.children.regions)
        for ocr_r, can_r in zip(ocr_p.children.regions, can_p.children.regions):

            if ocr_r.bbox.xywh != can_r.bbox.xywh:
                print('ocr_type: ', ocr_r.region_type)
                print('ocr_text: ', ocr_r.text[:50])
                print('can_type: ', can_r.region_type)
                print('can_text: ', can_r.text[:50])

            assert len(ocr_r.children.lines) == len(can_r.children.lines)
            for ocr_l, can_l in zip(ocr_p.children.lines, can_p.children.lines):
                assert ocr_l.bbox.xywh == can_l.bbox.xywh

                assert len(ocr_l.children.words) == len(can_l.children.words)
                for ocr_w, can_w in zip(ocr_p.children.words, can_p.children.words):
                    assert ocr_w.bbox.xywh == can_w.bbox.xywh
                    assert ocr_w.text == can_w.text

        ocr_p.mreset()


# test_ocrcommentary_to_canonical()

def test_ocrpage():
    page = ocr_classes.RawPage(ocr_path=so.sample_ocr_page_path, id=so.sample_page_id,
                               img_path=so.sample_img_path, commentary=so.sample_ocrcommentary)

    assert isinstance(page.ocr_format, str)

    assert isinstance(page.image, image.AjmcImage)

    assert all([isinstance(r, ocr_classes.RawRegion) for r in page.children.regions])
    assert all([isinstance(l, ocr_classes.RawLine) for l in page.children.lines])
    assert all([isinstance(w, ocr_classes.RawWord) for w in page.children.words])

    # Validate page.json
    schema_path = variables.SCHEMA_PATH
    schema = json.loads(schema_path.read_text('utf-8'))
    jsonschema.validate(instance=page.to_inception_dict(), schema=schema)

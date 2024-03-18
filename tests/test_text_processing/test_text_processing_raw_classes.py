import json

import jsonschema

from ajmc.commons import image, variables
from ajmc.text_processing import raw_classes
from tests import sample_objects as so


def test_rawcommentary():
    # test RawCommentary.children
    assert all([isinstance(p, raw_classes.RawPage) for p in so.sample_raw_commentary.children.pages])
    assert len(so.sample_raw_commentary.children.pages) == len(list(so.sample_ocr_run_outputs_dir.glob('*.hocr')))
    assert all([isinstance(r, raw_classes.RawRegion) for r in so.sample_raw_commentary.children.regions])
    assert all([isinstance(l, raw_classes.RawLine) for l in so.sample_raw_commentary.children.lines])
    assert all([isinstance(w, raw_classes.RawWord) for w in so.sample_raw_commentary.children.words])

    # test RawCommentary.images
    assert all([isinstance(i, image.AjmcImage) for i in so.sample_raw_commentary.images])

    # Test RawCommentary.groundtruth_pages
    assert all([isinstance(p, raw_classes.RawPage) for p in so.sample_raw_commentary.ocr_gt_pages])


# test_ocrcommentary_to_canonical()

def test_rawpage():
    page = raw_classes.RawPage(ocr_path=so.sample_ocr_page_path, id=so.sample_page_id,
                               img_path=so.sample_img_path, commentary=so.sample_raw_commentary)

    assert isinstance(page.ocr_format, str)

    assert isinstance(page.image, image.AjmcImage)

    assert all([isinstance(r, raw_classes.RawRegion) for r in page.children.regions])
    assert all([isinstance(l, raw_classes.RawLine) for l in page.children.lines])
    assert all([isinstance(w, raw_classes.RawWord) for w in page.children.words])

    # Validate page.json
    schema_path = variables.SCHEMA_PATH
    schema = json.loads(schema_path.read_text('utf-8'))
    jsonschema.validate(instance=page.to_inception_dict(), schema=schema)


def test_rawpage_optimise():
    for page in so.sample_raw_commentary.children.pages:
        page.optimise()
        for region in page.children.regions:
            assert isinstance(region, raw_classes.RawRegion)
            assert region.parents.page == page
            assert region.children.lines
            assert all([isinstance(l, raw_classes.RawLine) for l in region.children.lines])
            for line in region.children.lines:
                assert line.parents.region == region

        for line in page.children.lines:
            assert isinstance(line, raw_classes.RawLine)
            assert line.parents.page == page
            assert isinstance(line.parents.region, raw_classes.RawRegion)
            assert line.children.words
            assert all([isinstance(w, raw_classes.RawWord) for w in line.children.words])
            for word in line.children.words:
                try:
                    assert word.parents.line == line
                except AssertionError:
                    print()
                    print('*********************************************')
                    print('page', page.id)
                    print('line', line.text)
                    print(line.bbox.bbox)
                    print('word', word.text)
                    print(word.bbox.bbox)
                    print('word parent line:', word.parents.line.text)
                    print(word.parents.line.bbox.bbox)

        for word in page.children.words:
            assert isinstance(word, raw_classes.RawWord)
            assert word.text
            assert word.bbox

        assert all([isinstance(r, raw_classes.RawRegion) for r in page.children.regions])
        assert all([isinstance(l, raw_classes.RawLine) for l in page.children.lines])
        assert all([isinstance(w, raw_classes.RawWord) for w in page.children.words])

        page.reset()

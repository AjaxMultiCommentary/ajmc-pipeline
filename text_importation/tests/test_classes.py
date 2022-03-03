from text_importation.classes import *


def test_commentary():
    c = Commentary('lestragdiesdeso00tourgoog', ocr='tesshocr')
    assert type(c.via_project) == dict
    assert all([isinstance(p, Page) for p in c.pages])


def test_page():
    pass


c = Commentary('lestragdiesdeso00tourgoog', ocr='tesshocr')
p = c.pages[1]
# regions = get_page_region_dicts_from_via(p.id, c.via_project)
# regions = [Region(r, p) for r in regions]
# order_olr_regions(regions)
p.regions

p.canonical_data
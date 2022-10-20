from ajmc.commons.miscellaneous import LazyObject
from tests import sample_objects as so
from ajmc.commons.geometry import Shape
from ajmc.text_processing import canonical_classes as cc
from ajmc.commons.variables import CHILD_TYPES, TEXTCONTAINER_TYPES
import pytest


@pytest.mark.parametrize('tc', [so.sample_cancommentary,
                                so.sample_cancommentary.children.pages[0],
                                so.sample_cancommentary.children.regions[0],
                                so.sample_cancommentary.children.lines[0],
                                so.sample_cancommentary.children.words[0],
                                so.sample_cancommentary_from_json,
                                so.sample_cancommentary_from_json.children.pages[0],
                                so.sample_cancommentary_from_json.children.regions[0],
                                so.sample_cancommentary_from_json.children.lines[0],
                                so.sample_cancommentary_from_json.children.words[0]])
def test_textcontainer(tc):
    # test CanonicalCommentary.children
    assert isinstance(tc.children, LazyObject)
    for tc_type in CHILD_TYPES:
        assert isinstance(getattr(tc.children, tc_type), list)
        assert all([isinstance(tc, cc.get_tc_type_class(tc_type)) for tc in getattr(tc.children, tc_type)])

    # Test CanonicalCommentary.parents
    assert isinstance(tc.parents, LazyObject)
    for tc_type in TEXTCONTAINER_TYPES:
        parent = getattr(tc.parents, tc_type)
        assert isinstance(parent, cc.get_tc_type_class(tc_type)) or parent is None

    assert isinstance(tc.type, str)

    assert isinstance(tc.id, str)

    assert isinstance(tc.text, str)



@pytest.mark.parametrize('commentary', [so.sample_cancommentary,
                                so.sample_cancommentary_from_json])
def test_canonical_commentary(commentary):
    # test CanonicalCommentary.images
    assert all([isinstance(i, cc.Image) for i in commentary.images])
    assert len(commentary.images) == len(commentary.children.pages)

@pytest.mark.parametrize('tc', [so.sample_cancommentary.children.pages[0],
                                so.sample_cancommentary.children.regions[0],
                                so.sample_cancommentary.children.lines[0],
                                so.sample_cancommentary.children.words[0],
                                so.sample_cancommentary_from_json.children.pages[0],
                                so.sample_cancommentary_from_json.children.regions[0],
                                so.sample_cancommentary_from_json.children.lines[0],
                                so.sample_cancommentary_from_json.children.words[0]])
def test_canonical_textcontainer(tc):
    assert tc in getattr(tc.parents.commentary.children, tc.type+'s')

    if tc.parents.page is not None:
        assert tc in getattr(tc.parents.page.children, tc.type+'s')

    if tc.parents.region is not None:
        assert tc in getattr(tc.parents.region.children, tc.type+'s')

    if tc.parents.line is not None:
        assert tc in getattr(tc.parents.line.children, tc.type+'s')

    assert isinstance(tc.bbox, Shape)





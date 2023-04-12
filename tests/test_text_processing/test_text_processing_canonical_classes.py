import pytest
from lazy_objects.lazy_objects import LazyObject

from ajmc.commons import variables as vs
from ajmc.text_processing import canonical_classes as cc
from tests import sample_objects as so


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
    for tc_type in vs.CHILD_TYPES:
        assert isinstance(getattr(tc.children, tc_type), list)
        assert all([isinstance(tc, cc.get_tc_type_class(tc_type)) for tc in getattr(tc.children, tc_type)])

    # Test CanonicalCommentary.parents
    assert isinstance(tc.parents, LazyObject)
    for tc_type in vs.TEXTCONTAINER_TYPES:
        parent = getattr(tc.parents, tc_type)
        assert isinstance(parent, cc.get_tc_type_class(tc_type)) or parent is None

    assert isinstance(tc.type, str)

    assert isinstance(tc.id, str)

    assert isinstance(tc.text, str)


@pytest.mark.parametrize('commentary', [so.sample_cancommentary,
                                        so.sample_cancommentary_from_json])
def test_canonical_commentary(commentary):
    # test CanonicalCommentary.images
    assert all([isinstance(i, cc.AjmcImage) for i in commentary.images])
    assert len(commentary.images) == len(commentary.children.pages)


@pytest.mark.parametrize('tc', [*[getattr(so.sample_cancommentary.children, tc_type)[0]
                                  for tc_type in vs.CHILD_TYPES
                                  if getattr(so.sample_cancommentary.children, tc_type)]])
def test_canonical_textcontainer(tc):
    assert tc in getattr(tc.parents.commentary.children, vs.TC_TYPES_TO_CHILD_TYPES[tc.type])

    # Test family relationships
    for tc_type in vs.TEXTCONTAINER_TYPES:
        parent = getattr(tc.parents, tc_type)
        if parent is not None:
            if parent.type != 'word':
                assert tc in getattr(parent.children, vs.TC_TYPES_TO_CHILD_TYPES[tc.type])

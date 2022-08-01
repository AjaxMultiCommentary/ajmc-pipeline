from tests import sample_objects as so
from ajmc.commons.geometry import Shape
from ajmc.text_processing import canonical_classes as cc
import pytest


@pytest.mark.parametrize('container', [so.sample_cancommentary,
                                       so.sample_cancommentary.children['page'][0],
                                       so.sample_cancommentary.children['region'][0],
                                       so.sample_cancommentary.children['line'][0],
                                       so.sample_cancommentary.children['word'][0]])
def test_canonicaltextcontainer(container):
    assert isinstance(container.commentary, cc.CanonicalCommentary)

    assert isinstance(container.type, str)

    assert isinstance(container.id, str)

    assert isinstance(container.word_range, tuple)
    assert len(container.word_range) == 2

    assert isinstance(container.children, dict)

    assert isinstance(container.parents, dict)

    assert isinstance(container.bbox, Shape)
    assert len(container.bbox.xywh) == 4

    assert isinstance(container.text, str)

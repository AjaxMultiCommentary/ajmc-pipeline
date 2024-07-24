import pytest
import unicodedata

from ajmc.nlp.lemlink.data_preparation import TEI2TextMapper

@pytest.fixture(scope="module")
def mapper():
    return TEI2TextMapper('https://raw.githubusercontent.com/gregorycrane/Wolf1807/master/ajax-2019/ajax-lj.xml')

class TestTEI2TextMapper():
    def test_init(self, mapper):
        assert isinstance(mapper.text, str)
        assert unicodedata.is_normalized('NFC', mapper.text)

    def test_offsets_to_selector(self, mapper):
        assert mapper.offsets_to_selector([38, 100]) == 'tei-l@n=2[4]:tei-l@n=3[29]'

    def test_selector_to_offsets(self, mapper):
        assert mapper.selector_to_offsets('tei-l@n=2[4]:tei-l@n=3[29]') == [38, 100]
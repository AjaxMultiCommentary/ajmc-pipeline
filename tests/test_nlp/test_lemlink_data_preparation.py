import unicodedata

import pytest

from ajmc.nlp.lemlink.data_preparation import TEI2TextMapper


@pytest.fixture(scope="module")
def mapper():
    return TEI2TextMapper('https://raw.githubusercontent.com/gregorycrane/Wolf1807/master/ajax-2019/ajax-lj.xml')

class TestTEI2TextMapper():
    def test_init(self, mapper):
        assert isinstance(mapper.text, str)
        assert unicodedata.is_normalized('NFC', mapper.text)

    def test_lines_for_offsets(self, mapper):
        offsets = mapper.selector_to_offsets('tei-l@n=9[0]:tei-l@n=24[34]')
        lines = mapper.lines_for_offsets(offsets)
        
        for idx, line in enumerate(lines, start=9):
            assert line.n == str(idx)

        offsets = mapper.selector_to_offsets('tei-l@n=208[0]:tei-l@n=209[15]')
        lines = mapper.lines_for_offsets(offsets)

        assert len(lines) == 2
        
        for idx, line in enumerate(lines, start=208):
            assert line.n == str(idx)

    def test_offsets_to_selector(self, mapper):
        assert mapper.offsets_to_selector([38, 100]) == 'tei-l@n=2[4]:tei-l@n=3[29]'

    def test_selector_to_offsets(self, mapper):
        offsets = mapper.selector_to_offsets('tei-l@n=9[0]:tei-l@n=9[34]')

        assert offsets == [289, 323]
        assert mapper.text[offsets[0]:offsets[1]] == 'ἔνδον γὰρ ἁνὴρ ἄρτι τυγχάνει, κάρα';

from ajmc.commons import unicode_utils as uu


def test_count_chars_by_charset():
    string = 'abdεθ-:123ξ,'
    assert uu.count_chars_by_charset(string, 'latin') == 3
    assert uu.count_chars_by_charset(string, 'greek') == 3
    assert uu.count_chars_by_charset(string, 'numeral') == 3
    assert uu.count_chars_by_charset(string, 'punctuation') == 3

import logging

import pytest
from typing import List
from ajmc.commons import miscellaneous as misc


def test_lazy_property():
    class Foo:
        def __init__(self):
            pass

        @misc.lazy_property
        def bar(self) -> List[int]:
            """A list of ints."""
            return [1, 2, 3]

    a = Foo()

    assert a.bar == a._bar == [1, 2, 3]

    a.bar = [4, 5, 6]
    assert a.bar == [4, 5, 6]
    assert a._bar == [4, 5, 6]

    del a.bar
    assert a.bar == a._bar == [1, 2, 3]
    assert Foo.bar.__doc__.startswith('A list of ints.')


def test_recursive_iterator():
    iterable = [(0, 1), [[[[2]], 3], 4], 5]
    iterator = misc.recursive_iterator(iterable, (list, tuple))
    assert [i for i in iterator] == [0, 1, 2, 3, 4, 5]


def test_get_custom_logger():
    assert isinstance(misc.get_custom_logger('test'), logging.Logger)


@pytest.mark.parametrize('sheet_name', ['sheet_1', 'sheet_2'])
def test_read_google_sheet(sheet_name):
    df = misc.read_google_sheet(sheet_id='1Ao9zSzmvdwvn7OAAtq7gwLbrJQg21TARg0roO1-CoHg',
                                sheet_name=sheet_name)
    assert df['test_int'][0] == 108
    assert df['test_str'][1] == 'coucou'
    assert len(df['test_str']) == 2

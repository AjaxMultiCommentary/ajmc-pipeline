import logging
import pytest
from typing import List
from ajmc.commons import miscellaneous as misc


def test_recursive_iterator():
    iterable = [(0, 1), [[[[2]], 3], 4], 5]
    iterator = misc.recursive_iterator(iterable, (list, tuple))
    assert [i for i in iterator] == [0, 1, 2, 3, 4, 5]


def test_split_list():
    assert misc.split_list([1, 2, 3, 4, 5], 2, pad=None) == [[1, 2], [3, 4], [5, None]]


@pytest.mark.parametrize('sheet_name', ['sheet_1', 'sheet_2'])
def test_read_google_sheet(sheet_name):
    df = misc.read_google_sheet(sheet_id='1qABQgkQeQJPDn9SkJPtyXFwe9bUhgID6xWM6adRS3vc',
                                # sheet_id='1Ao9zSzmvdwvn7OAAtq7gwLbrJQg21TARg0roO1-CoHg',
                                sheet_name=sheet_name)
    assert df['test_int'][0] == 108
    assert df['test_str'][1] == 'coucou'
    assert len(df['test_str']) == 2


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


def test_get_custom_logger():
    assert isinstance(misc.get_custom_logger('test'), logging.Logger)


def test_lazy_init():
    class Foo:
        @misc.lazy_init
        def __init__(self, a: int, b: int = 3, **kwargs):
            pass

    a = Foo(1, b=2, c=4)
    assert a.a == 1
    assert a.b == 2
    assert a.c == 4


def test_lazy_attributer():
    @misc.lazy_attributer(attr_name='bar', func=lambda self: self.a + 4, attr_decorator=property)
    class Foo:
        def __init__(self, a: int):
            self.a = a

    a = Foo(1)
    assert a.bar == 5


def test_lazyobject():
    a = misc.LazyObject(compute_function=lambda x: len(x),
                        constrained_attrs=['a', 'bcd'])
    assert a.bcd == 3

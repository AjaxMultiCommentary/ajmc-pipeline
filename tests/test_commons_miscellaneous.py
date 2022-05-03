import pytest
import logging
from commons import miscellaneous as misc


def test_recursive_iterator():
    iterable = [(0, 1), [[[[2]], 3], 4], 5]
    iterator = misc.recursive_iterator(iterable, (list, tuple))
    assert [i for i in iterator] == [0, 1, 2, 3, 4, 5]


def test_get_custom_logger():
    assert isinstance(misc.get_custom_logger('test'), logging.Logger)

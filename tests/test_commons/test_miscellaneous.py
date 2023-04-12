import logging

import pytest

import ajmc.commons.file_management
from ajmc.commons import miscellaneous as misc


def test_recursive_iterator():
    iterable = [(0, 1), [[[[2]], 3], 4], 5]
    iterator = misc.recursive_iterator(iterable, (list, tuple))
    assert [i for i in iterator] == [0, 1, 2, 3, 4, 5]


def test_split_list():
    assert misc.split_list([1, 2, 3, 4, 5], 2, pad=None) == [[1, 2], [3, 4], [5, None]]


@pytest.mark.parametrize('sheet_name', ['sheet_1', 'sheet_2'])
def test_read_google_sheet(sheet_name):
    df = ajmc.commons.file_management.read_google_sheet(sheet_id='1exFGuU_dDcbbiacvpu1jz13JIvOQ0fTArPP27XUpLNk',
                                                        sheet_name=sheet_name)
    assert df['test_int'][0] == 108
    assert df['test_str'][1] == 'coucou'
    assert len(df['test_str']) == 2


def test_get_custom_logger():
    assert isinstance(misc.get_custom_logger('test'), logging.Logger)

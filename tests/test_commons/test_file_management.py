import time

from ajmc.commons import file_management


def test_int_to_x_based_code():
    assert file_management.int_to_x_based_code(0, 62) == '0'
    assert file_management.int_to_x_based_code(64, 62) == '12'
    assert file_management.int_to_x_based_code(3, base=62, fixed_min_len=3) == '003'


def test_get_62_based_datecode():
    # make the computer wait for 1 second to make sure the datecode is different
    a = file_management.get_62_based_datecode()
    time.sleep(1)
    assert a != file_management.get_62_based_datecode()
    assert len(file_management.get_62_based_datecode()) == 6

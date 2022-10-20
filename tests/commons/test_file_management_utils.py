from ajmc.commons import variables
from ajmc.commons.file_management import utils
import os
from tests import sample_objects as so
import time

def test_int_to_x_based_code():
    assert utils.int_to_x_based_code(0, 62) == '0'
    assert utils.int_to_x_based_code(64, 62) == '12'
    assert utils.int_to_x_based_code(3, base=62, fixed_min_len=3) == '003'

def test_get_62_based_datecode():
    #make the computer wait for 1 second to make sure the datecode is different
    a = utils.get_62_based_datecode()
    time.sleep(1)
    assert a != utils.get_62_based_datecode()
    assert len(utils.get_62_based_datecode()) == 6


def test_verify_path_integrity():
    good_paths = [
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary/commentaries_data/cu31924087948174/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/outputs',
        ]
    wrong_paths = [
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary/commentaries_data/cu31924087948174/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/output',
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary/commentaries_data/coucou/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/outputs'
    ]
    for path in good_paths:
        utils.verify_path_integrity(path=path,
                                    path_pattern=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])

    for path in wrong_paths:
        try:
            utils.verify_path_integrity(path=path,
                                        path_pattern=variables.FOLDER_STRUCTURE_PATHS['ocr_outputs_dir'])
        except AssertionError:
            pass

        else:
            raise """`utils.verify_path_integrity` should raise an error with `wrong_paths`."""


def test_get_path_from_id():
    assert utils.find_file_by_name(so.sample_page_id, so.sample_image_dir) == os.path.join(so.sample_image_dir, so.sample_page_id + '.png')
    assert not utils.find_file_by_name(so.sample_page_id.split('_')[0] + '_9999', so.sample_image_dir)


# todo : path management
def test_parse_ocr_path():
    path = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/outputs'
    base, commentary_id, ocr_run = utils.parse_ocr_path(path)
    assert base == '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data'
    assert commentary_id == 'cu31924087948174'
    assert ocr_run == '2480ei_greek-english_porson_sophoclesplaysa05campgoog'




from commons.file_management import utils
from commons import variables


def test_get_62_based_datecode():
    assert len(utils.get_62_based_datecode()) == 6


def test_verify_path_integrity():
    good_paths = [
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/outputs',
        ]
    wrong_paths = [
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/output',
        '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/coucou/ocr/runs/2480ei_greek-english_porson_sophoclesplaysa05campgoog/outputs'
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

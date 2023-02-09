from pathlib import Path

from ajmc.ocr import evaluation as ocr_eval
from ajmc.text_processing.ocr_classes import OcrCommentary, OcrPage


def test_count_chars_by_charset():
    string = 'abdεθ-:123ξ,'
    assert ocr_eval.count_chars_by_charset(string, 'latin') == 3
    assert ocr_eval.count_chars_by_charset(string, 'greek') == 3
    assert ocr_eval.count_chars_by_charset(string, 'numbers') == 3
    assert ocr_eval.count_chars_by_charset(string, 'punctuation') == 3


def test_count_errors_by_charset():
    gt_string = 'abdεθ-:123ξ,'
    ts_string = 'aaedεx-x1x3ξ,'
    assert ocr_eval.count_errors_by_charset(gt_string, ts_string, 'latin') == 2
    assert ocr_eval.count_errors_by_charset(gt_string, ts_string, 'greek') == 1
    assert ocr_eval.count_errors_by_charset(gt_string, ts_string, 'numbers') == 1
    assert ocr_eval.count_errors_by_charset(gt_string, ts_string, 'punctuation') == 1


def test_bag_of_word_evaluation():
    gt_bag = ['soleil', 'maison', 'je', '122', 'courage']
    pr_bag_1 = ['soleil', 'maeson', 'je', '122.cou']
    pr_bag_2 = ['soleil', 'maeson', 'je', '123', 'courage', 'sole']
    error_counts_1 = ocr_eval.bag_of_word_evaluation(gt_bag, pr_bag_1)
    error_counts_2 = ocr_eval.bag_of_word_evaluation(gt_bag, pr_bag_2)

    assert error_counts_1['distance'] == 12
    assert error_counts_1['ccr'] == 0.5
    assert error_counts_1['precision'] == 0.5
    assert error_counts_1['recall'] == 0.4

    assert error_counts_2['distance'] == 6
    assert error_counts_2['ccr'] == 0.75
    assert error_counts_2['precision'] == 0.5
    assert error_counts_2['recall'] == 0.6


def test_coord_based_page_evaluation():
    base_dir = Path('tests/data/sample_evaluation_data')

    # We first create a commentary because via is accessed via the commentary
    comm = OcrCommentary(via_path=base_dir / 'via_project.json', )

    gt_page = OcrPage(ocr_path=base_dir / 'gt_sophoclesplaysa05campgoog_0146.html',
                      page_id='sophoclesplaysa05campgoog_0146',
                      img_path=base_dir / 'sophoclesplaysa05campgoog_0146.png',
                      commentary=comm)

    test_page = OcrPage(ocr_path=base_dir / 'test_sophoclesplaysa05campgoog_0146.html',
                        page_id='sophoclesplaysa05campgoog_0146',
                        img_path=base_dir / 'sophoclesplaysa05campgoog_0146.png',
                        commentary=comm)

    editops, error_counts, _ = ocr_eval.coord_based_page_evaluation(gt_page=gt_page, pred_page=test_page)

    assert error_counts['global']['words']['total'] == 548
    assert error_counts['global']['words']['false'] == 25
    assert error_counts['global']['words']['cr'] == 1 - 25 / 548

    assert error_counts['global']['chars']['total'] == 2518
    assert error_counts['global']['chars']['false'] == 37
    assert error_counts['global']['chars']['cr'] == 1 - 37 / 2518

    assert error_counts['global']['greek']['false'] == 18  # 21 - 3 insertion of greek chars in latin words ;-)
    assert error_counts['global']['latin']['false'] == 12
    assert error_counts['global']['numbers']['false'] == 6

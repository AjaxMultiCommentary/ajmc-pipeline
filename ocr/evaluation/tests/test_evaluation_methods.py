from text_importation.classes import Page
from ocr.evaluation.evaluation_methods import *


def test_bag_of_word_evaluation():
    gt_bag = ['soleil', 'maison', 'je', '122', 'courage']
    pr_bag_1 = ['soleil', 'maeson', 'je', '122.cou']
    pr_bag_2 = ['soleil', 'maeson', 'je', '123', 'courage', 'sole']
    error_counts_1 = bag_of_word_evaluation(gt_bag, pr_bag_1)
    error_counts_2 = bag_of_word_evaluation(gt_bag, pr_bag_2)

    assert error_counts_1['distance'] == 12
    assert error_counts_1['ccr'] == 0.5
    assert error_counts_1['precision'] == 0.5
    assert error_counts_1['recall'] == 0.4

    assert error_counts_2['distance'] == 6
    assert error_counts_2['ccr'] == 0.75
    assert error_counts_2['precision'] == 0.5
    assert error_counts_2['recall'] == 0.6


def test_coord_based_page_evaluation():
    gt_page = Page('sophoclesplaysa05campgoog_0146',
                   ocr_path='/Users/sven/ajmc/data/ocr/evaluation_test/gt_sophoclesplaysa05campgoog_0146.html')
    test_page = Page('sophoclesplaysa05campgoog_0146',
                     ocr_path='/Users/sven/ajmc/data/ocr/evaluation_test/test_sophoclesplaysa05campgoog_0146.html')

    editops, error_counts, _ = coord_based_page_evaluation(gt_page=gt_page, pred_page=test_page)

    assert error_counts['global']['words']['total'] == 548
    assert error_counts['global']['words']['false'] == 25
    assert error_counts['global']['words']['cr'] == 1 - 25 / 548

    assert error_counts['global']['chars']['total'] == 2518
    assert error_counts['global']['chars']['false'] == 37
    assert error_counts['global']['chars']['cr'] == 1 - 37 / 2518

    assert error_counts['global']['greek']['false'] == 17  # 21 - 3 insertion of greek chars in latin words ;-)
    assert error_counts['global']['latin']['false'] == 12
    assert error_counts['global']['numbers']['false'] == 6


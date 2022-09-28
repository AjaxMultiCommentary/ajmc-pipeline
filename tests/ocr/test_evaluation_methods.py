from ajmc.text_processing.ocr_classes import OcrPage, OcrCommentary
from ajmc.ocr.evaluation.evaluation_methods import bag_of_word_evaluation, coord_based_page_evaluation


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
    base_dir = '/Users/sven/packages/ajmc/'

    # We first create a commentary because via is accessed via the commentary
    comm = OcrCommentary(via_path=base_dir + 'data/ocr/evaluation_test_data/via_project.json')

    gt_page = OcrPage(ocr_path=base_dir + 'data/ocr/evaluation_test_data/gt_sophoclesplaysa05campgoog_0146.html',
                      id='sophoclesplaysa05campgoog_0146',
                      image_path=base_dir + 'data/ocr/evaluation_test_data/sophoclesplaysa05campgoog_0146.png',
                      commentary=comm)

    test_page = OcrPage(ocr_path=base_dir + 'data/ocr/evaluation_test_data/test_sophoclesplaysa05campgoog_0146.html',
                        id='sophoclesplaysa05campgoog_0146',
                        image_path=base_dir + 'data/ocr/evaluation_test_data/sophoclesplaysa05campgoog_0146.png',
                        commentary=comm)

    editops, error_counts, _ = coord_based_page_evaluation(gt_page=gt_page, pred_page=test_page)

    assert error_counts['global']['words']['total'] == 548
    assert error_counts['global']['words']['false'] == 25
    assert error_counts['global']['words']['cr'] == 1 - 25 / 548

    assert error_counts['global']['chars']['total'] == 2518
    assert error_counts['global']['chars']['false'] == 37
    assert error_counts['global']['chars']['cr'] == 1 - 37 / 2518

    assert error_counts['global']['greek']['false'] == 18  # 21 - 3 insertion of greek chars in latin words ;-)
    assert error_counts['global']['latin']['false'] == 12
    assert error_counts['global']['numbers']['false'] == 6

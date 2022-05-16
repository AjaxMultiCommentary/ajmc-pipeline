import csv
import os
from typing import List, Dict, Tuple, Union, Optional
import Levenshtein
from ajmc.commons.variables import ORDERED_OLR_REGION_TYPES
from ajmc.commons.miscellaneous import safe_divide
from ajmc.commons.geometry import is_rectangle_within_rectangle, are_rectangles_overlapping_with_threshold
from ajmc.ocr.evaluation.utils import initialize_soup, count_chars_by_charset, count_errors_by_charset, record_editops, \
    insert_text_in_soup, write_error_counts
from ajmc.text_importation.classes import Page, Commentary
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)
# todo add fuzzy eval

def bag_of_word_evaluation(gt_bag: List[str],
                           pred_bag: List[str],
                           error_counts: Optional[Dict[str, Union[int, float]]] = None,
                           ) -> Dict[str, Union[int, float]]:
    """Performs a bag-of-word evaluation of `pred_bag` against `groundtruth_bag`.

    Args:
        gt_bag: The list of groundtruth words
        pred_bag: The list of ocr-predicted words
        error_counts: A dict with results (pass it if you want to evaluate multiple pages

    Returns:
        float: The caracter-error rate"""

    if not error_counts:
        error_counts = {'gt_words': 0, 'pred_words': 0, 'true_words': 0,
                        'chars': 0, 'distance': 0, }

    pred_bag_ = pred_bag.copy()

    error_counts['gt_words'] += len(gt_bag)
    error_counts['pred_words'] += len(pred_bag_)
    error_counts['chars'] += sum([len(w) for w in gt_bag])

    for gt_word in gt_bag:  # iterate over w in gt_string

        if pred_bag_:  # If there are still words in `ocr_bag_`...

            distances = [Levenshtein.distance(pred_word, gt_word) for pred_word in pred_bag_]

            min_distance = min(distances)

            # Records `min_distance` and removes word
            if min_distance == 0:
                error_counts['true_words'] += 1
            error_counts['distance'] += min_distance
            del pred_bag_[distances.index(min_distance)]

        else:  # If `ocr_bag_` is empty, add the distance of unrecognized gt_words
            error_counts['distance'] += len(gt_word)

    if pred_bag_:  # If `ocr_bag_` still contains words after the end of the loop
        error_counts['distance'] += sum([len(w) for w in pred_bag_])

    error_counts['precision'] = safe_divide(error_counts['true_words'], error_counts['pred_words'])
    error_counts['recall'] = safe_divide(error_counts['true_words'], error_counts['gt_words'])
    error_counts['f1'] = safe_divide((2 * error_counts['precision'] * error_counts['recall']), (
            error_counts['precision'] + error_counts['recall']))
    error_counts['cwr'] = safe_divide(error_counts['true_words'], error_counts['gt_words'])
    error_counts['ccr'] = 1 - safe_divide(error_counts['distance'], error_counts['chars'])

    return error_counts


def simple_coordinates_based_evaluation(gt_words: List['TextElement'],
                                        pred_words: List['TextElement'],
                                        overlap_threshold: float = 0.8) -> float:
    """Computes edit distance between spacially overlapping words and returns the CER.

     Simple means that this method does NOT deal with word-boxes related issues. It only evaluates gt-words which
     overlap to `overlap_threshold` with a predicted word and vice-versa. If no predicted word is found
     (e.g. with crummy groundtruth- or preds-boxes), the word is left out and not counted in the final result.

     Args:
         gt_words: The list of gt words (e.g. `Page.words` or `Region.words`)
         pred_words: The list of ocr words (e.g. `Page.words` or `Region.words`)
         overlap_threshold: The minimal overlap-proportion.

     Returns:
         float: the character error rate
    """

    pred_words_ = pred_words.copy()
    matched_words = 0
    total_characters = 0
    total_edit_distance = 0

    for gt_word in gt_words:

        for i, pred_word in enumerate(pred_words_):
            if are_rectangles_overlapping_with_threshold(pred_word.coords.bounding_rectangle,
                                                         gt_word.coords.bounding_rectangle,
                                                         overlap_threshold):
                total_characters += len(gt_word.text)
                total_edit_distance += Levenshtein.distance(pred_word.text, gt_word.text)
                matched_words += 1
                del pred_words_[i]
                break

    logger.info(f"""Evaluating on {matched_words} words, for a total of {len(gt_words)} words.""")

    return total_edit_distance / total_characters


def coord_based_page_evaluation(gt_page: 'Page',
                                pred_page: 'Page',
                                word_overlap_threshold: Optional[float] = 0.8,
                                error_counts: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
                                editops_record: Optional[Dict[Tuple[str, str, str], int]] = None
                                ) -> Tuple[dict, dict, 'BeautifulSoup']:
    """Performs a regional and coordinates-based evaluation.

    This function returns extremely detailed counts, with word counts, caracter counts by charsets (latin, greek,
    numbers and punctuation) and correct rate (`cr`, corresponding to the normalized levenshtein distance) for
    each of these elements and for each olr region (commentary, primary text...).

    How to read the results? `cr`or `ccr`/`cwr` (correct character/word rate respectively) very straightforward. They
    correspond to the number of correct elements divided by the total number of elements.

    Note:
        Coordinate-based means that evaluation does not process documents in a linear manner, which is prone to
        alignement error when document layouts are complex. Instead, this matches overlapping words in groundtruth and
        ocr-data. More formally, for each groundtruth word :
            - find the predicted word which coordinates overlap the with groundtruth word to `word_overlap_threshold`
                - if found, calculate Levenshtein distance between the two
                - if not found, do not evaluate this word

    Args:
        gt_page: The groundtruth page
        pred_page: The predicted page
        word_overlap_threshold: The threshold to find overlaping words
        error_counts: A dict of counts (pass in if you want to evaluate multiple pages at once)
        editops_record: A dict of edit operations (pass in if you want to evaluate multiple pages at once)

    Returns:
        error_counts, editops_record, soup
    """

    soup = initialize_soup(img_width=gt_page.image.width, img_height=gt_page.image.height)  # Initialize html output
    charsets = ['latin', 'greek', 'punctuation', 'numbers']
    pred_words_ = pred_page.words.copy()

    if not error_counts:
        error_counts = {region:
                            {level:
                                 {count: 0 for count in ['total', 'evaluated', 'false']}
                             for level in ['words', 'chars'] + charsets}
                        for region in ['global'] + ORDERED_OLR_REGION_TYPES}

    if not editops_record:
        editops_record = {}

    for gt_word in gt_page.words:

        # Find `gt_word`'s regions
        gt_word_regions = ['global'] + [r.region_type for r in gt_page.regions if
                                        is_rectangle_within_rectangle(gt_word.coords.bounding_rectangle,
                                                                      r.coords.bounding_rectangle)]

        for region in gt_word_regions:
            error_counts[region]['words']['total'] += 1
            error_counts[region]['chars']['total'] += len(gt_word.text)
            for charset in charsets:
                error_counts[region][charset]['total'] += count_chars_by_charset(gt_word.text, charset)

        # Find the corresponding ocr_word
        for i, pred_word in enumerate(pred_words_):
            if are_rectangles_overlapping_with_threshold(pred_word.coords.bounding_rectangle,
                                                         gt_word.coords.bounding_rectangle, word_overlap_threshold):
                distance = Levenshtein.distance(pred_word.text, gt_word.text)

                for region in gt_word_regions:

                    # Count evaluated words and false words
                    error_counts[region]['words']['evaluated'] += 1
                    error_counts[region]['words']['false'] += min(1, distance)

                    # Count evaluated chars and errors
                    error_counts[region]['chars']['evaluated'] += len(gt_word.text)
                    error_counts[region]['chars']['false'] += distance

                    # Count evaluated chars and errors by charset
                    for charset in charsets:
                        error_counts[region][charset]['evaluated'] += count_chars_by_charset(gt_word.text, charset)
                        error_counts[region][charset]['false'] += count_errors_by_charset(gt_word.text, pred_word.text,
                                                                                          charset)

                # Record edit operations
                editops_record = record_editops(gt_word=gt_word.text,
                                                ocr_word=pred_word.text,
                                                editops=Levenshtein.editops(pred_word.text, gt_word.text),
                                                editops_record=editops_record)

                # Actualize soup
                soup = insert_text_in_soup(soup=soup, word=gt_word, is_gt=True, is_false=bool(distance))
                soup = insert_text_in_soup(soup=soup, word=pred_word, is_gt=False, is_false=bool(distance))

                del pred_words_[i]
                break

    # Compute error rates
    for region in ['global'] + ORDERED_OLR_REGION_TYPES:
        for level in ['words', 'chars'] + charsets:
            error_counts[region][level]['cr'] = 1 - safe_divide(error_counts[region][level]['false'],
                                                                error_counts[region][level]['evaluated'])

    return editops_record, error_counts, soup


def commentary_evaluation(commentary: 'Commentary',
                          write_files: bool = True,
                          output_dir: Optional[str] = None,
                          word_overlap_threshold: float = 0.8):
    """Evaluates all the pages of a `Commentary` that have groundtruth.

    Args:
        commentary: The `Commentary` object to evaluate.
        write_files: Whether to write the files or not
        output_dir: Leave to none if you want to write files to the default dir
        word_overlap_threshold: See `coord_based_page_evaluation`.
    """

    bow_error_counts, coord_error_counts, editops = None, None, None
    soups = []

    for gt_page in commentary.ocr_groundtruth_pages:
        pred_page = [p for p in commentary.pages if p.id == gt_page.id][0]

        bow_error_counts = bag_of_word_evaluation(gt_bag=[w.text for w in gt_page.words],
                                                  pred_bag=[w.text for w in pred_page.words],
                                                  error_counts=bow_error_counts)

        editops, coord_error_counts, soup = coord_based_page_evaluation(gt_page=gt_page,
                                                                        pred_page=pred_page,
                                                                        word_overlap_threshold=word_overlap_threshold,
                                                                        error_counts=coord_error_counts,
                                                                        editops_record=editops)
        soups.append(soup)

    if write_files:
        if not output_dir:
            output_dir = os.path.join(commentary.paths['ocr_dir'], os.pardir, 'evaluation')

        os.makedirs(output_dir, exist_ok=True)

        for i, soup in enumerate(soups):
            with open(os.path.join(output_dir, commentary.ocr_groundtruth_pages[i].id + ".html"), "w") as html_file:
                html_file.write(str(soup))

        # Sort and write editops record
        editops = {k: v for k, v in sorted(editops.items(), key=lambda item: item[1], reverse=True)}
        with open(os.path.join(output_dir, "editops.tsv"), 'w') as csv_file:
            spamwriter = csv.writer(csv_file, delimiter='\t', quotechar='"')
            spamwriter.writerow(['Operation', 'From', 'To', 'Count'])
            for k, v in editops.items():
                spamwriter.writerow([k[0], k[1], k[2], v])

        write_error_counts(bow_error_counts, coord_error_counts, output_dir)

    return bow_error_counts, coord_error_counts, editops


def evaluate_all():
    """Evaluate all commentaries and runs"""
    # a single ocr models output for all commentaries
    # do the evaluate on each of the models outputs
    # weright average to get the general results

    raise NotImplementedError
#
# commentary = Commentary('cu31924087948174',
#                         '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/ocr/runs/tess_eng_grc/outputs')
# commentary_evaluation(commentary=commentary, )
#
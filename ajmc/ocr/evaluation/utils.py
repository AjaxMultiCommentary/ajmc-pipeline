import os
import re
import Levenshtein
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, Tuple
import cv2
from ajmc.commons.variables import CHARSETS


def initialize_soup(img_width: int, img_height: int) -> "BeautifulSoup":
    """Initializes a blank html page for comparing predictions and errors

    Args:
        img_width: Width of the page's image.
        img_height: Height of the page's image

    Returns:
        The initialized BeautifulSoup Object
    """

    divisor = img_width * 2 / 1440

    soup = BeautifulSoup(f"""<!doctype html><html lang="en"><head>
                         <meta charset="utf-8" divisor="{divisor}" half_width="{int(img_width / divisor)}"></head>
                         <body><p style="margin:auto;text-align:center"> 
                         <b>OCR/GROUNDTRUTH COMPARISON </b><br> 
                         Ocr is displayed on the left, groundtruth on the right.<br>
                         Missrecognized words are contoured in red. </p>
                         </body></html>""", features='lxml')

    soup.html.body.append(soup.new_tag(name='div',
                                       attrs={'style': f"""margin:auto; position:relative; 
                                              width:{int(img_width * 2 / divisor)}px; 
                                              height:{int(img_height / (divisor * 0.8))}px"""}))

    return soup


def insert_text_in_soup(soup: "BeautifulSoup", word: 'OcrWord', is_gt: bool, is_false: bool) -> "BeautifulSoup":
    """Adds content to `soup` object.

    This function is used to add single words to the `soup` object initialized by `initialize_soup`, thereby
    reconstructing both the groundtruth and the preds page, with false words in red.

    Args:
        soup: an initilised BeautifulSoup object
        word: The ocr- or groundtruth word to add
        is_gt: Wheter the word is groundtruth or not (determines where to place it).
        is_false: Whether the word was falsely predicted or not.

    Returns
        The modified BeautifulSoup object"""

    divisor = float(soup.html.head.meta['divisor'])

    # Write groundtruth to the right
    x_coord = int(word.coords.bounding_rectangle[3][0] / divisor) + \
              (int(soup.html.head.meta["half_width"]) if is_gt else 0)
    y_coord = int(word.coords.bounding_rectangle[3][1] / divisor)

    new_div = soup.new_tag(name="div",
                           attrs={"style": f"""position:absolute; 
                                             width:{word.coords.width / divisor}px; 
                                             height:{word.coords.height / divisor}px; 
                                             left:{x_coord}px; 
                                             top:{y_coord}px;
                                             font-size: 80%;
                                             """ + ("""color: red;font-weight: bold;""" if is_false else '')})

    new_div.string = word.text
    soup.html.body.div.append(new_div)

    return soup


def actualize_overlap_matrix(args: "ArgumentParser", image: "Image", zonemask: "ZonesMasks", groundtruth: "OcrObject",
                             ocr: "OcrObject") -> "ndarray":
    """Creates the overlap matrix used to match overlaping segments and zones

    :return: an ndarray of shape (4, image.height, image.width) ;
        layer 0 contains zones names
        layer 1 contains groundtruth words ids
        layer 2 contains ocr words ids
        layer 3 contains contours points
    """

    for zone in zonemask.zones:  # For each zone, fill matrix and add error dictionary

        image.overlap_matrix[0, zone.coords[0][1]:zone.coords[2][1],
        zone.coords[0][0]:zone.coords[2][0]] += zone.zone_type + " "  # adds zone.type to matrix, eg. "primary_text"

    for gt_word in groundtruth.words:  # For each gt_word in gt, fill matrix, then find overlapping gt- and ocr-words
        image.overlap_matrix[1, gt_word.coords[0][1]:gt_word.coords[2][1],
        gt_word.coords[0][0]:gt_word.coords[2][0]] = gt_word.id

    for word in ocr.words:  # For each word in ocr, fill matrix
        image.overlap_matrix[2, word.coords[0][1]:word.coords[2][1], word.coords[0][0]:word.coords[2][0]] = word.id

    return image.overlap_matrix


def record_editops(gt_word: str, ocr_word: str, editops: list,
                   editops_record: Dict[Tuple[str, str, str], int]) -> Dict[Tuple[str, str, str], int]:
    """Adds word-level edit operation to the record of all edit operations"""

    for editop in editops:
        if editop[0] == 'delete':
            editop_tuple = ('delete', ocr_word[editop[1]], '')
        elif editop[0] == 'insert':
            editop_tuple = ('insert', gt_word[editop[2]], '')
        else:
            editop_tuple = (editop[0], ocr_word[int(editop[1])], gt_word[int(editop[2])])

        try:
            editops_record[editop_tuple] += 1
        except KeyError:
            editops_record[editop_tuple] = 1

    return editops_record


def actualize_error_counts(error_counts: Dict[str, Dict[str, int]], gt_word: "Segment", distance: int) -> Dict[
    str, Dict[str, int]]:
    """Actualizes error counts at each word step, registering counts at global- and zone-level

    :return: The actualized error counts.
    """

    for key in ["global"] + gt_word.zone_type:
        error_counts[key]["gt_chars"] += len(gt_word.content)
        error_counts[key]["distance"] += distance
        error_counts[key]["gt_words"] += 1
        error_counts[key]["greek_chars"] += len(
            re.findall(r'([\u0373-\u03FF]|[\u1F00-\u1FFF]|\u0300|\u0301|\u0313|\u0314|\u0345|\u0342|\u0308)',
                       gt_word.content, re.UNICODE))
        error_counts[key]["numbers"] += len(re.findall(r"[0-9]", gt_word.content))

        if distance > 0:
            error_counts[key]["false_words"] += 1

    return error_counts


def draw_surrounding_rectangle(image_matrix: "ndarray",
                               segment: "Segment",
                               color: tuple,
                               thickness: int,
                               surrounding_box_type: str = "shrinked") -> "ndarray":
    """Draws the surrounding rectangle of a segment.

    :param image_matrix: ndarray retrieved from cv2.imread()
    :param segment:
    :param color: tuple of BGR-color, e.g. (255,109, 118)
    :param thickness: int, see cv2, e.g. 2
    :param surrounding_box_type: Which surroundings to draw : "shrinked or "raw"
    :return: the modified image_matrix
    """

    if surrounding_box_type == "shrinked":
        _ = cv2.rectangle(image_matrix,
                          (segment.coords[0][0], segment.coords[0][1]),
                          (segment.coords[2][0], segment.coords[2][1]),
                          color, thickness)

    elif surrounding_box_type == "raw":
        _ = cv2.rectangle(image_matrix,
                          (segment.raw_coords[0][0], segment.raw_coords[0][1]),
                          (segment.raw_coords[2][0], segment.raw_coords[2][1]),
                          color, thickness)

    return image_matrix


def compute_confusion_metrics(error_counts: Dict) -> Tuple[float, float, float]:
    """Computes precision, recall and F1-score over words.

    :param error_counts:
    :return: precision: float, recall: float, f1: float
    """
    # todo delete
    TP = error_counts["global"]["gt_words"] - error_counts["global"]["false_words"]  # = nb de mots justement prédits
    FP = error_counts["global"]["ocr_words"] - TP  # mots présents dans l'ocr qui ne sont pas TP
    FN = error_counts["global"]["false_words"]  # nb de mots faux

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def harmonise_unicode(word):
    word = re.sub(r"᾽", "’", word)
    word = re.sub(r"ʼ", "’", word)
    word = re.sub(r"'", "’", word)
    word = re.sub(r"—", "-", word)
    word = re.sub(r"„", '"', word)

    return word


def count_errors_by_charset(gt_string: str, pred_string: str, charset: str) -> int:
    """Counts the number of errors among the character comprised in an unicode character set.

    Example:
        `count_errors_by_charset('Hεll_ World1', 'ηειι- world', 'greek')` returns `3` as among the 4 greek
        chars in `gt`, 3 are misrecognized.

    Args:
        pred_string: prediction/source string
        gt_string: groundtruth/destination string
        charset: should be `'greek'`, `'latin'`, `'numbers'`, `'punctuation'` or a valid `re`-pattern,
                 for instance `r'([\u00F4-\u00FF])'`

    Returns:
        int: the number of errors on selected caracters in `pred_string`
    """

    try:
        pattern = CHARSETS[charset]
    except KeyError:
        pattern = re.compile(charset, re.UNICODE)

    indices = [m.span()[0] for m in re.finditer(pattern, gt_string)]
    editops = Levenshtein.editops(pred_string, gt_string)

    # min() is there to cope with insertion at the end of the string
    return sum([1 for e in editops if min(e[2], len(gt_string) - 1) in indices])


def count_chars_by_charset(string: str, charset: str) -> int:
    """Counts the number of chars by unicode characters set.

    Example:
        `count_chars_by_charset('γεια σας, world', 'greek')` returns `7` as there are 7 greek
        chars in `string`.

    Args:
        string: self explanatory
        charset: should be `'greek'`, `'latin'`, `'numbers'`, `'punctuation'` or a valid `re`-pattern,
                 for instance `r'([\u00F4-\u00FF])'`

    Returns:
        int: the number of charset-matching characters in `string`.
    """
    try:
        pattern = CHARSETS[charset]
    except KeyError:
        pattern = re.compile(charset, flags=re.UNICODE)

    return len(re.findall(pattern, string))


def write_error_counts(bow_error_counts: dict,
                       coord_error_counts: dict,
                       output_dir: str):
    sorted_bow_keys = ['ccr', 'cwr', 'f1', 'precision', 'recall']
    sorted_bow_keys = sorted_bow_keys + [k for k in bow_error_counts.keys() if k not in sorted_bow_keys]

    reformed_bow = {('global', 'bow', k): [bow_error_counts[k]] for k in sorted_bow_keys}

    reformed_coord = {(i, j, k): [coord_error_counts[i][j][k]]
                      for i in coord_error_counts.keys()
                      for j in coord_error_counts[i].keys()
                      for k in ['cr', 'total', 'evaluated', 'false']}

    df_bow = pd.DataFrame.from_dict(reformed_bow, orient='columns')
    df_coord = pd.DataFrame.from_dict(reformed_coord, orient='columns')

    df = pd.concat([df_bow, df_coord], axis=1)
    df.to_csv(os.path.join(output_dir, 'evaluation_results.tsv'), index=False, sep='\t')
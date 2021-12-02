import re

import Levenshtein
from bs4 import BeautifulSoup
import numpy as np
from oclr.utils import file_management
import numpy as np
from typing import List, Dict, Tuple
import cv2


def initialize_soup(image: "Image") -> "BeautifulSoup":
    """Initializes a blank html page for comparisons

    :return: The initialized BeautifulSoup Object
    """

    divisor = image.width * 2 / 1440

    soup = BeautifulSoup("""<!doctype html>
                         <html lang="en">
                         <head><meta charset="utf-8"><title>{}</title></head>
                         <body><p style="margin:auto;text-align:center"> 
                         <b>OCR/GROUNDTRUTH COMPARISON </b><br> 
                         Ocr is displayed on the left, groundtruth on the right.<br>
                         Missrecognized words are contoured in red. </p>
                         </body></html>""".format(image.filename), features="lxml")

    to_append = soup.new_tag(name="div",
                             attrs={"half_width": image.width / divisor,
                                    "style": "margin:auto;"
                                             "position:relative;"
                                             "width:{}px; "
                                             "height:{}px".format(image.width * 2 / divisor,
                                                                  image.height / (divisor*0.8))})

    soup.html.body.append(to_append)

    return soup


def insert_text(soup: "BeautifulSoup", image: "Image", word: "Segment", gt: bool, distance: int) -> "BeautifulSoup":
    """Writes segment text on soup object

    :param soup: an initilised BeautifulSoup object
    :param word: an ocr- or a groundtruth segment
    :param gt: wheter the segment is groundtruth or not
    :param distance: The levenstein distance for the word
    :return: the BeautifulSoup object"""

    divisor = image.width * 2 / 1440

    x_coord = word.raw_coords[3][0] / divisor + soup.html.body.div["half_width"] if gt else word.raw_coords[3][
                                                                                                  0] / divisor
    y_coord = word.raw_coords[3][1] / divisor

    border = "border:1px solid green;"
    color = "color: red;" if distance > 0 else ""
    font_weight = "font-weight: bold;" if distance > 0 else ""
    line = "text-decoration: underline dotted red;" if distance > 0 else ""
    font_size = "font-size: 80%;"

    to_append = soup.new_tag(name="div",
                             attrs={"style": "position:absolute;" +
                                             "width:{}px;".format(word.print_width / divisor) +
                                             "height:{}px;".format(word.print_height / divisor) +
                                             "left:{}px;".format(x_coord) +
                                             "top:{}px;".format(y_coord) +
                                             "{}".format(color) +
                                             "{}".format(font_weight) +
                                             "{}".format(line) +
                                             "{}".format(font_size) #+
                                             #"{}".format(border)

                                    }
                             )

    to_append.string = word.content

    soup.html.body.div.append(to_append)

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

        image.overlap_matrix[0, zone.coords[0][1]:zone.coords[2][1],zone.coords[0][0]:zone.coords[2][0]] += zone.zone_type+" "  # adds zone.type to matrix, eg. "primary_text"

    for gt_word in groundtruth.words:  # For each gt_word in gt, fill matrix, then find overlapping gt- and ocr-words
        image.overlap_matrix[1, gt_word.coords[0][1]:gt_word.coords[2][1],
        gt_word.coords[0][0]:gt_word.coords[2][0]] = gt_word.id

    for word in ocr.words:  # For each word in ocr, fill matrix
        image.overlap_matrix[2, word.coords[0][1]:word.coords[2][1], word.coords[0][0]:word.coords[2][0]] = word.id

    return image.overlap_matrix


def record_editops(gt_word: "Segment", ocr_word: "Segment", editops: list, editops_record: list) -> list:
    """Appends word-level edit operation to the record of all edit operations

    :param editops: edit operations for current word
    :param editops_record: edit operations record
    :return: edit operations record
    """
    for editop in editops:
        if editop[0] == "delete":
            editops_record.append("delete " + ocr_word.content[editop[1]])
        elif editop[0] == "insert":
            editops_record.append("insert " + gt_word.content[editop[2]])
        else:
            editops_record.append(
                "{} {} -> {}".format(editop[0], ocr_word.content[int(editop[1])],
                                     gt_word.content[int(editop[2])]))

    return editops_record


def actualize_error_counts(error_counts: Dict[str, Dict[str, int]], gt_word: "Segment", distance: int) -> Dict[
    str, Dict[str, int]]:
    """Actualizes error counts at each word step, registering counts at global- and zone-level

    :return: The actualized error counts.
    """

    for key in ["global"]+gt_word.zone_type:
        error_counts[key]["gt_chars"] += len(gt_word.content)
        error_counts[key]["distance"] += distance
        error_counts[key]["gt_words"] += 1
        error_counts[key]["greek_chars"] += len(re.findall(r'([\u0373-\u03FF]|[\u1F00-\u1FFF]|\u0300|\u0301|\u0313|\u0314|\u0345|\u0342|\u0308)',gt_word.content, re.UNICODE))
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

    TP = error_counts["global"]["gt_words"] - error_counts["global"]["false_words"]  # = nb de mots justement prédits
    FP = error_counts["global"]["ocr_words"] - TP  # mots présents dans l'ocr qui ne sont pas TP
    FN = error_counts["global"]["false_words"]  # nb de mots faux

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def compute_confusion_metrics2(groundtruth, ocr):

    ocr_bag = [ocr_word for ocr_word in ocr.words] # = list of w in ocr
    false_ocr = 0
    true_ocr = 0
    min_distance = None
    for gt_word in groundtruth.words: # iterate over w in gt
        # finds the word with min distance, break if 0
        for i,ocr_word in enumerate(ocr_bag): # iterate over w in ocr
            ocr_word.distance = Levenshtein.distance(ocr_word.content, gt_word.content)
            if i == 0:
                min_distance = ocr_word.distance
            if ocr_word.distance < min_distance:
                min_distance = ocr_word.distance
            if ocr_word.distance == 0:
                break
        # select the first word with min distance
        for ocr_word in ocr_bag:
            if ocr_word.distance == min_distance:
                ocr_bag.remove(ocr_word)
                break

        if min_distance == 0:
            true_ocr += 1
        else:
            false_ocr += 1

    precision = true_ocr / len(ocr.words)
    recall = true_ocr / len(groundtruth.words)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def compute_confusion_metrics3(groundtruth_bag, ocr_bag):

    ocr_bag_copy = ocr_bag.copy()
    true_ocr = 0

    for gt_word in groundtruth_bag: # iterate over w in gt
        if gt_word in ocr_bag_copy:
            true_ocr += 1
            ocr_bag_copy.remove(gt_word)

    precision = true_ocr / len(ocr_bag)
    recall = true_ocr / len(groundtruth_bag)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def harmonise_unicode(word):

    word = re.sub(r"᾽", "’", word)
    word = re.sub(r"ʼ", "’", word)
    word = re.sub(r"'", "’", word)
    word = re.sub(r"—", "-", word)
    word = re.sub(r"„", '"', word)

    return word

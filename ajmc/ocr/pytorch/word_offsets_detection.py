from pathlib import Path

import cv2
import numpy as np

from ajmc.commons import geometry as geom, variables as vs
from ajmc.commons.geometry import adjust_bbox_to_included_contours
from ajmc.commons.image import find_contours, remove_artifacts_from_contours
from ajmc.commons.miscellaneous import ROOT_LOGGER, get_ajmc_logger
from ajmc.ocr.data_processing.data_generation import draw_textline
from ajmc.ocr.data_processing.font_utils import Font

logger = get_ajmc_logger(__name__)
ROOT_LOGGER.setLevel('INFO')

DATA_DIR = Path('/scratch/sven/ocr_exp/datasets/ajmc')
OUTPUT_DIR = Path('/scratch/sven/ocr_exp/word_detection_exps')


def prepare_line_img_for_word_detection(line_img: np.ndarray,
                                        top_bottom_margin: float = 0.2,
                                        artefact_size_threshold: float = 0.15) -> tuple[np.ndarray, list[geom.Shape]]:
    """Prepares the line image for word detection by removing the top and bottom margins and removing artifacts.

    Args:
        line_img (np.ndarray): The image to prepare, black on white, 8-bit, single channel.
        top_bottom_margin (float): The proportion of the top and bottom margins to remove.
        artefact_size_threshold (float): A proportion of the image height. Countours with a perimeter inferior to
        ``4*line_img.shape[0]*artefact_size_threshold`` are considered artifacts and are removed.

    Returns:
        tuple[np.ndarray, list[geom.Shape]]: The prepared image and the contours.
    """

    work_img = line_img.copy()

    # Remove the top and bottom margins
    margin = int(work_img.shape[0] * top_bottom_margin)
    work_img[0: margin, :] = 255
    work_img[-margin:, :] = 255

    # Get the contours
    contours = find_contours(255 - work_img, binarize=False)

    # Start by removing artifacts aggressively
    threshold = artefact_size_threshold * work_img.shape[0] * 4
    clean_contours = remove_artifacts_from_contours(contours, threshold)
    removed_contours = [c for c in contours if c not in clean_contours]

    # Remove the contours from the image
    for c in removed_contours:
        work_img[c.ymin - 1:c.ymax + 1, c.xmin - 1:c.xmax + 1] = 255

    return work_img, contours


def get_word_boxes_by_dilation(line_img: np.ndarray, text: str) -> list[geom.Shape]:
    """Finds the bounding boxes of the words in the image given a predicted word count.

    Note:
        This works by iteratively dilating the image using cv2.dilate until the number detected contours match the number of words in the string.

    Args:
        line_img (np.ndarray): The image to find the words in, black on white, 8-bit, single channel.
        text (str): The predicted string.

    Returns:
        List[geom.Shape]: The bounding boxes of the words in the image.
    """
    word_count = len(text.split())

    if word_count == 0:
        return []

    work_img, contours = prepare_line_img_for_word_detection(line_img)

    # Appplying dilation on the threshold image
    kernel_size = 2
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilation = cv2.dilate(255 - work_img, rect_kernel, iterations=1)

    dilated_contours = find_contours(dilation, binarize=False)

    while len(dilated_contours) > word_count:
        kernel_size += 1
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilation = cv2.dilate(255 - work_img, rect_kernel, iterations=1)
        dilated_contours = find_contours(dilation, binarize=False)

    # Adjust the bounding boxes to the original image
    dilated_contours = [adjust_bbox_to_included_contours(c.bbox, contours) for c in dilated_contours]

    return dilated_contours


def get_word_boxes_by_spaces(cv2_line: np.ndarray, line_text: str) -> list[geom.Shape]:
    # prepare the line image for word detection
    line_img, contours = prepare_line_img_for_word_detection(cv2_line, top_bottom_margin=0.15, artefact_size_threshold=0.15)
    line_img = 255 - line_img

    # Sum the line image along the y-axis to get a 1D array
    line_sum = line_img.sum(axis=0)

    # Find the groups of consecutive zeros in the 1D array, and get their offsets and their lengths
    zero_groups = []
    zero_group = None
    for i, val in enumerate(line_sum):
        if val == 0:
            if zero_group is None:
                zero_group = [i, 1]
            else:
                zero_group[1] += 1
        else:
            if zero_group is not None:
                zero_groups.append(zero_group)
                zero_group = None

    # Add the last group if it is not None
    if zero_group is not None:
        zero_groups.append(zero_group)

    # Remove the leading and trailing zeros
    if zero_groups:
        if zero_groups[0][0] == 0:
            zero_groups.pop(0)
        if zero_groups[-1][0] + zero_groups[-1][1] == line_img.shape[1]:
            zero_groups.pop(-1)

    # Sort the groups by length
    zero_groups.sort(key=lambda x: x[1], reverse=True)

    # Keep only the zero groups matching the number of spaces in the line text
    zero_groups = zero_groups[:line_text.count(' ')]

    # Reorder the zero groups by their offsets
    zero_groups.sort(key=lambda x: x[0])

    # Get the word boxes
    word_boxes = []
    previous_offset = 0
    for i, (offset, length) in enumerate(zero_groups):
        word_boxes.append(geom.Shape([(previous_offset, 0), (offset, line_img.shape[0])]))
        previous_offset = offset + length
    word_boxes.append(geom.Shape([(previous_offset, 0), (line_img.shape[1], line_img.shape[0])]))

    # Shrink the boxes to included contours
    word_boxes = [adjust_bbox_to_included_contours(w.bbox, contours) for w in word_boxes]

    return word_boxes


def get_word_boxes_by_projection(line_img: np.ndarray, line_text: str) -> list[geom.Shape]:
    """Finds the bounding boxes of the words in  ``line_img`` given a predicted ``line_text``.g

    Note:
        This works by drawing the text and strechting it to fit the line image. This method should be use as a fallback when the other methods fail.

    Args:
        line_img (np.ndarray): The image to find the words in, black on white, 8-bit, single channel.
        line_text (str): The predicted string.
    """

    if not line_text:
        return []

    # We start by preparing the line image for word detection
    line_img, contours = prepare_line_img_for_word_detection(line_img, top_bottom_margin=0.10, artefact_size_threshold=0.15)
    line_img = 255 - line_img

    # We identify the first and last black pixels
    line_img_sum = line_img.sum(axis=0)
    black_pixel_indices = [i for i, val in enumerate(line_img_sum) if val > 0]
    try:
        first_black_pixel = black_pixel_indices[0]
        last_black_pixel = black_pixel_indices[-1]
    except IndexError:
        first_black_pixel = 0
        last_black_pixel = line_img.shape[1]

    # We will now draw the text, starting by creating the fonts
    greek_font = Font(vs.PACKAGE_DIR / 'data/fonts/fonts/Porson-Regular.otf')
    latin_font = Font(vs.PACKAGE_DIR / 'data/fonts/fonts/TimesNewRoman-Regular.ttf')
    fallback_font = Font(vs.PACKAGE_DIR / 'data/fonts/fonts/Cardo-Regular.ttf')
    fonts = {'greek': greek_font, 'latin': latin_font, 'numeral': latin_font, 'punctuation': latin_font}

    # We draw the text
    draw, chars_widths, chars_offsets = draw_textline(line_text,
                                                      fonts=fonts,
                                                      fallback_fonts=[fallback_font],
                                                      font_variants=['regular'] * len(line_text),
                                                      target_height=line_img.shape[0],
                                                      return_chars_offsets=True)

    # We now expand spaces so that the drawn text has the same width as the line image
    number_of_spaces = line_text.count(' ')
    if number_of_spaces > 0:
        missing_width = last_black_pixel - first_black_pixel - draw.width
        space_additional_width = missing_width // number_of_spaces

        for char, width in zip(line_text, chars_widths):
            if char == ' ':
                width += space_additional_width

        # We now resize the offsets and widths
        for i in range(1, len(chars_offsets)):
            chars_offsets[i] += chars_offsets[i - 1] + chars_widths[i - 1]

    # We will now find the word offsets
    word_offsets = []
    previous_offset = first_black_pixel
    word_width = 0
    for char_offset, char_width, char in zip(chars_offsets, chars_widths, line_text):
        if char != ' ':
            word_width += char_width

        else:
            word_offsets.append((previous_offset, word_width))
            previous_offset += word_width + char_width
            word_width = 0

    word_offsets.append((previous_offset, word_width))

    # We now distribute the words along the line horizontally
    if len(word_offsets) > 1:
        total_width = sum(width for _, width in word_offsets)
        space_between_words = (last_black_pixel - first_black_pixel - total_width) // (len(word_offsets) - 1)

        for i in range(1, len(word_offsets)):
            word_offsets[i] = (word_offsets[i - 1][0] + word_offsets[i - 1][1] + space_between_words, word_offsets[i][1])

    # We now draw the word boxes
    word_boxes = []
    for offset, width in word_offsets:
        word_boxes.append(geom.Shape([(offset, 0), (offset + width, line_img.shape[0] - 1)]))

    return word_boxes


def get_word_boxes(line_img: np.ndarray, line_text: str) -> list[geom.Shape]:
    """Finds the bounding boxes of the words in  ``line_img`` given a predicted ``line_text``.

    Args:
        line_img (np.ndarray): The image to find the words in, black on white, 8-bit, single channel.
        line_text (str): The predicted string.
    """
    # We will start by trying to find the word boxes using the projection method
    word_boxes = get_word_boxes_by_dilation(line_img, line_text)

    # If we have found no words, we will try the contour method
    if len(word_boxes) != line_text.count(' ') + 1:
        logger.info(f'Word detection by dilation. Trying fallback method.')
        word_boxes = get_word_boxes_by_projection(line_img, line_text)

    return word_boxes

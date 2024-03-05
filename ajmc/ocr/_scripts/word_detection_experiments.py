from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from ajmc.commons import geometry as geom
from ajmc.commons.geometry import adjust_bbox_to_included_contours
from ajmc.commons.image import find_contours, remove_artifacts_from_contours
from ajmc.commons.miscellaneous import ROOT_LOGGER

ROOT_LOGGER.setLevel('INFO')

DATA_DIR = Path('/scratch/sven/ocr_exp/datasets/ajmc')
OUTPUT_DIR = Path('/scratch/sven/ocr_exp/word_detection_exps')


def prepare_line_img_for_word_detection(line_img: np.ndarray):
    work_img = line_img.copy()

    # Remove the margin
    margin = work_img.shape[0] // 5
    work_img[0: margin, :] = 255
    work_img[-margin:, :] = 255

    # Get the contours
    contours = find_contours(255 - work_img, binarize=False)

    # Start by removing artifacts aggressively
    threshold = 0.15 * work_img.shape[0] * 4
    clean_contours = remove_artifacts_from_contours(contours, threshold)
    removed_contours = [c for c in contours if c not in clean_contours]

    # Remove the contours from the image
    for c in removed_contours:
        work_img[c.ymin - 1:c.ymax + 1, c.xmin - 1:c.xmax + 1] = 255

    return work_img, contours


def get_word_boxes(img: np.ndarray, word_count: int, draw_path: Optional[Union[Path, str]] = None):
    """Finds the bounding boxes of the words in the image given a predicted word count.

    Note:
        This works by iteratively dilating the image using cv2.dilate until the number detected contours match the number of words in the string.

    Args:
        img (np.ndarray): The image to find the words in, black on white, 8-bit, single channel.
        word_count (str): The predicted string.

    Returns:
        List[geom.Shape]: The bounding boxes of the words in the image.
    """

    work_img, contours = prepare_line_img_for_word_detection(img)

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

    if draw_path is not None:
        # Draw the contours
        draw_img = img.copy()
        for c in dilated_contours:
            cv2.rectangle(draw_img, (c.xmin, c.ymin), (c.xmax, c.ymax), (0, 255, 0), 2)
        cv2.imwrite(str(draw_path), draw_img)

    return dilated_contours


def draw_boxes_on_img(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)
    return img


def get_word_boxes_fallback(img, string, draw_path=None):
    work_img, contours = prepare_line_img_for_word_detection(img)
    words = string.split()
    space_length = int(work_img.shape[1] / len(string))
    word_length = [int(len(w) / len(string) * work_img.shape[1]) for w in words]

    word_boxes = []
    last_x = 0
    for i, word in enumerate(words):
        word_boxes.append(geom.Shape.from_xywh(last_x, 0, word_length[i], work_img.shape[0]))
        last_x += word_length[i] + space_length

    word_boxes = [adjust_bbox_to_included_contours(c.bbox, contours) for c in word_boxes]

    if draw_path is not None:
        draw_img = img.copy()
        for c in word_boxes:
            cv2.rectangle(draw_img, (c.xmin, c.ymin), (c.xmax, c.ymax), (0, 255, 0), 2)
        cv2.imwrite(str(draw_path), draw_img)

    return word_boxes


EXP_NAME = 'fallback_word_detection'
EXP_DIR = OUTPUT_DIR / EXP_NAME
EXP_DIR.mkdir(exist_ok=True, parents=True)
errors = 0

i = 0
for img_path in tqdm(DATA_DIR.glob('*.png')):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    string = img_path.with_suffix('.txt').read_text(encoding='utf-8')
    word_count = len(string.split())

    word_boxes = get_word_boxes(img, word_count)
    word_boxes = get_word_boxes_fallback(img, string)

    i += 1

    # if len(word_boxes) != word_count:
    errors += 1
    (EXP_DIR / img_path.name).with_suffix('.txt').write_text(string, encoding='utf-8')
    cv2.imwrite(str(EXP_DIR / img_path.name), draw_boxes_on_img(img, word_boxes))

    if i > 100:
        break
print(f'Exp: {EXP_NAME}')
print(f'Errors: {errors}')

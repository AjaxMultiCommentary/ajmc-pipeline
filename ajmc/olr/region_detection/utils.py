from typing import List

import cv2
import numpy as np
import os
import csv

from ajmc.commons.geometry import Shape, is_rectangle_within_rectangle
from ajmc.commons.image import remove_artifacts_from_contours


def correct_csv_manually(csv_filename: str):
    """manually corrects quoting error in output csv-files"""

    os.system("""sed -ia 's/ //g' """ + csv_filename)
    os.system("""sed -ia "s/'/ /g" """ + csv_filename)
    os.system("""sed -ia 's/ /""/g' """ + csv_filename)
    os.system("""sed -ia 's/},/}",/g' """ + csv_filename)
    os.system("""sed -ia 's/,{/,"{/g' """ + csv_filename)
    os.system("""sed -ia 's/""}/""}"/g' """ + csv_filename)


def write_csv_manually(csv_filename: str, csv_dict: dict, output_dir:str):
    """Writes a dictionnary a csv-file with custom quoting corresponding to via expectations"""

    pwd = os.getcwd()
    os.chdir(output_dir)

    with open(csv_filename, 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(list(csv_dict.keys()))
        for line in range(len(csv_dict["filename"])):
            to_append = []
            for k in list(csv_dict.keys()):
                to_append.append(csv_dict[k][line])
            spamwriter.writerow(to_append)

    correct_csv_manually(csv_filename)
    os.remove(csv_filename + "a")
    os.chdir(pwd)


def detect_regions(img_path: str,
                   output_dir: str,
                   dilation_kernel_size: int,
                   dilation_iterations: int,
                   artifact_size_threshold: int,
                   draw_rectangles: bool,
                   via_csv_dict: dict) -> dict:
    """Automatically detects regions of interest in `img_path`, using a simple dilation process.
    Returns a `'key':[values]`-like dictionnary containing all the generated rectangles for all the images"""

    img_name = img_path.split("/")[-1]

    # Preparing image
    image = cv2.imread(img_path)
    copy = image.copy()
    binarized = image_processing.binarize(image)

    # Appplying dilation on the threshold image
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size))
    dilation = cv2.dilate(binarized, rect_kernel, iterations=dilation_iterations)

    # Finding contours
    contours: List[Shape] = image_processing.find_contours(binarized, do_binarize=False)
    contours = remove_artifacts_from_contours(contours, artifact_size_threshold)

    # Finding dilation contours
    dilated_contours: List[Shape] = image_processing.find_contours(dilation, do_binarize=False)
    dilated_contours = remove_artifacts_from_contours(dilated_contours, artifact_size_threshold)

    dilated_contours_shrinked = []

    for dc in dilated_contours:
        contained_contours = [c for c in contours if
                              is_rectangle_within_rectangle(c.bounding_rectangle, dc.bounding_rectangle)]

        if contained_contours:
            contained_stacked = Shape.from_numpy_array(
                np.concatenate([c.bounding_rectangle for c in contained_contours], axis=0))
            dilated_contours_shrinked.append(contained_stacked)

    for i, c in enumerate(dilated_contours_shrinked):
        via_csv_dict["filename"].append(img_name)
        via_csv_dict["file_size"].append(os.stat(img_path).st_size)
        via_csv_dict["file_attributes"].append("{}")
        via_csv_dict["region_count"].append(len(dilated_contours_shrinked))
        via_csv_dict["region_id"].append(i)
        via_csv_dict["region_shape_attributes"].append({"name": "rect",
                                                        "x": c.xywh[0],
                                                        "y": c.xywh[1],
                                                        "width": c.xywh[2],
                                                        "height": c.xywh[3]})
        via_csv_dict["region_attributes"].append({"text": "undefined"})

    # Draws output rectangles
    if draw_rectangles:
        # for rectangle in dilation_contours_rectangles:
        #     dilation_rectangle = cv2.rectangle(copy, (rectangle[0, 0], rectangle[0, 1]),
        #                          (rectangle[2, 0], rectangle[2, 1]), (0, 0, 255), 4)

        for c in dilated_contours_shrinked:
            shrinked_rectangle = cv2.rectangle(copy,
                                               (c.bounding_rectangle[0][0], c.bounding_rectangle[0][1]),
                                               (c.bounding_rectangle[2][0], c.bounding_rectangle[2][1]),
                                               (0, 0, 255), 4)

        cv2.imwrite(os.path.join(output_dir, img_name), copy)

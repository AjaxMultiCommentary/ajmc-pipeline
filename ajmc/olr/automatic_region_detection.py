"""
`region_detection` automatically detects regions-boxes in image-files and outputs
a csv that can be directly imported to VIA.

**Detecting boxes in image-files** is done using `cv2.dilation`. This dilates recognized letters-contours to recognize
wider structures. The retrieved bboxes are then shrinked back to their original size. This can be seen
when drawing bboxes on images.

**Synergy with VIA**. Then, to **transfer the project to VIA2**, please :

- [Download VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/)
- Open a new VIA2 project and import your images
- In the `project`-menu, chose import file/region attributes and import `default_via_attributes.json` from
the annotation_helper/data directory.
- In the `annotation`-menu, chose import annotations from csv, and import the annotations you want from the
annotation_helper output.
"""

import csv
import os
from typing import List
import cv2
import numpy as np
from ajmc.commons.geometry import Shape, is_bbox_within_bbox
from ajmc.commons.image import binarize, find_contours, remove_artifacts_from_contours
from ajmc.commons.variables import VIA_CSV_DICT_TEMPLATE
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)


@docstring_formatter(**docstrings)
def detect_regions(img_path: str,
                   output_dir: str,
                   dilation_kernel_size: int,
                   dilation_iterations: int,
                   draw_images: bool,
                   via_csv_dict: dict,
                   artifact_size_threshold: float = 0.003) -> dict:
    """Automatically detects regions of interest in an image, using a simple dilation process.
    Returns a `'key':[values]`-like dictionnary containing all the generated bboxes for all the images.

    Args:
        img_path: Absolute page to the image.
        output_dir: Absolute path to the dir in which to write images and csv.
        dilation_kernel_size: {dilation_kernel_size}
        dilation_iterations: {dilation_iterations}
        draw_images: Whether to draw the detected regions and output the image.
        via_csv_dict: The via_dict to complete.
        artifact_size_threshold: {artifact_size_threshold}
    """

    img_name = img_path.split("/")[-1]

    # Preparing image
    img_matrix = cv2.imread(img_path)
    copy = img_matrix.copy()
    binarized = binarize(img_matrix=img_matrix, inverted=True)

    # Setting the artifact perimeter threshold as percent of image
    artifact_perimeter_threshold = int(img_matrix.shape[1] * artifact_size_threshold)

    # Appplying dilation on the threshold image
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size))
    dilation = cv2.dilate(binarized, rect_kernel, iterations=dilation_iterations)
    # cv2.imwrite(os.path.join(output_dir, 'dil_'+img_name), dilation)

    # Finding contours
    contours: List[Shape] = find_contours(binarized, binarize=False)
    contours = remove_artifacts_from_contours(contours, artifact_perimeter_threshold)

    # Finding dilation contours
    dilated_contours: List[Shape] = find_contours(dilation, binarize=False)
    dilated_contours = remove_artifacts_from_contours(dilated_contours, artifact_perimeter_threshold)

    dilated_contours_shrinked = []

    for dc in dilated_contours:
        contained_contours = [c for c in contours if
                              is_bbox_within_bbox(c.bbox, dc.bbox)]

        if contained_contours:
            contained_stacked = Shape.from_numpy_array(
                np.concatenate([c.bbox for c in contained_contours], axis=0))
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

    # Draws output bboxes
    if draw_images:
        # for rectangle in dilation_contours_bboxes:
        #     dilation_rectangle = cv2.rectangle(copy, (rectangle[0, 0], rectangle[0, 1]),
        #                          (rectangle[2, 0], rectangle[2, 1]), (0, 0, 255), 4)

        for c in dilated_contours_shrinked:
            shrinked_bbox = cv2.rectangle(copy,
                                          (c.bbox[0][0], c.bbox[0][1]),
                                          (c.bbox[1][0], c.bbox[1][1]),
                                          (0, 0, 255), 4)

        cv2.imwrite(os.path.join(output_dir, img_name), copy)

    return via_csv_dict


def write_csv_manually(csv_filename: str, csv_dict: dict, output_dir: str):
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


def correct_csv_manually(csv_filename: str):
    """manually corrects quoting error in output csv-files"""

    os.system("""sed -ia 's/ //g' """ + csv_filename)
    os.system("""sed -ia "s/'/ /g" """ + csv_filename)
    os.system("""sed -ia 's/ /""/g' """ + csv_filename)
    os.system("""sed -ia 's/},/}",/g' """ + csv_filename)
    os.system("""sed -ia 's/,{/,"{/g' """ + csv_filename)
    os.system("""sed -ia 's/""}/""}"/g' """ + csv_filename)


@docstring_formatter(**docstrings)
def main(image_dir: str,
         output_dir: str,
         dilation_kernel_size: int,
         dilation_iterations: int,
         draw_images: bool,
         artifact_size_threshold: float = 0.003):
    """Runs `detect_regions` for all images in `img_dir`.

    Args:
        image_dir: {image_dir}
        output_dir: Absolute path to the dir in which to output csv and images (if draw_images).
        dilation_kernel_size: {dilation_kernel_size}
        dilation_iterations: {dilation_iterations}
        draw_images: Whether to draw the detected regions and output the image.
        via_csv_dict: The via_dict to complete.
        artifact_size_threshold: {artifact_size_threshold}
    """

    for filename in sorted(list(os.listdir(image_dir))):
        if filename[-3:] in ["png", "jpg", "tif", "jp2"]:
            logger.info("Processing image " + filename)

            via_csv_dict = detect_regions(img_path=os.path.join(image_dir, filename),
                                          output_dir=output_dir,
                                          dilation_kernel_size=dilation_kernel_size,
                                          dilation_iterations=dilation_iterations,
                                          artifact_size_threshold=artifact_size_threshold,
                                          draw_images=draw_images,
                                          via_csv_dict=VIA_CSV_DICT_TEMPLATE)

    write_csv_manually("detected_annotations.csv", via_csv_dict, output_dir)
    logger.info("{} zones were automatically detected".format(len(via_csv_dict["filename"])))

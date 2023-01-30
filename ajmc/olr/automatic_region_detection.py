"""
`automatic_region_detection` automatically detects regions-boxes in image-files and outputs
a csv that can be directly imported to VIA.
"""

import csv
import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from tqdm import tqdm

from ajmc.commons import variables
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.geometry import is_bbox_within_bbox, Shape
from ajmc.commons.image import binarize, find_contours, remove_artifacts_from_contours
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)


@docstring_formatter(**docstrings)
def detect_regions(img_path: Path, dilation_kernel_size: int, dilation_iterations: int, draw_image: bool,
                   img_output_dir: Path, via_csv_dict: dict, artifact_size_threshold: float = 0.003) -> dict:
    """Automatically detects regions of interest in an image, using a simple dilation process.
    Returns a `'key':[values]`-like dictionnary containing all the generated bboxes for all the images.

    Args:
        img_path: Absolute page to the image.
        dilation_kernel_size: {dilation_kernel_size}
        dilation_iterations: {dilation_iterations}
        draw_image: Whether to draw the detected regions and output the image.
        img_output_dir: Absolute path to the dir in which to write images and csv.
        via_csv_dict: The via_dict to complete.
        artifact_size_threshold: {artifact_size_threshold}
    """

    img_name = img_path.name

    # Preparing image
    img_matrix = cv2.imread(str(img_path))
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
        via_csv_dict["file_size"].append(img_path.stat().st_size)
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
    if draw_image:
        # for rectangle in dilation_contours_bboxes:
        #     dilation_rectangle = cv2.rectangle(copy, (rectangle[0, 0], rectangle[0, 1]),
        #                          (rectangle[2, 0], rectangle[2, 1]), (0, 0, 255), 4)

        for c in dilated_contours_shrinked:
            shrinked_bbox = cv2.rectangle(copy,
                                          (c.bbox[0][0], c.bbox[0][1]),
                                          (c.bbox[1][0], c.bbox[1][1]),
                                          (0, 0, 255), 4)

        cv2.imwrite(str(img_output_dir / img_name), copy)

    return via_csv_dict


def write_csv_manually(csv_filename: str, csv_dict: dict, output_dir: str):
    """Writes a dictionnary a csv-file with custom quoting corresponding to via expectations

    Args:
        csv_filename: The filename of the csv to write, e.g. 'detected_annotations.csv'.
        csv_dict: The dictionnary to write.
        output_dir: Absolute path to the dir in which to write the csv.
    """

    # This sure is an odd set of code, but it relies on the quiet specific way via expects csvs to be
    # Do not change it unless you are absolutely sure of what you are doing
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

    # This sure is an odd set of code, but it relies on the quiet specific way via expects csvs to be
    # Do not change it unless you are absolutely sure of what you are doing
    os.system("""sed -ia 's/ //g' """ + csv_filename)
    os.system("""sed -ia "s/'/ /g" """ + csv_filename)
    os.system("""sed -ia 's/ /""/g' """ + csv_filename)
    os.system("""sed -ia 's/},/}",/g' """ + csv_filename)
    os.system("""sed -ia 's/,{/,"{/g' """ + csv_filename)
    os.system("""sed -ia 's/""}/""}"/g' """ + csv_filename)


@docstring_formatter(**docstrings)
def main(img_dir: Union[str, Path],
         output_dir: Union[str, Path],
         dilation_kernel_size: int,
         dilation_iterations: int,
         draw_images: bool,
         artifact_size_threshold: float = 0.003,
         img_extension: str = variables.DEFAULT_IMG_EXTENSION):
    """
    Automatically detects regions-boxes in image-files and outputs a csv that can be directly imported to VIA.

    Detecting boxes in image-files is done using `cv2.dilation`. This dilates recognized letters-contours to recognize
    wider structures. The retrieved bboxes are then shrinked back to their original size. This can be seen
    when drawing bboxes on images.

    To get the optimal output from `cv2.dilation`, the following parameters can be tweaked:
    - `dilation_kernel_size`: The size of the kernel used for dilation. The bigger the kernel, the more
        the dilation will spread. The kernel is a square.
    - `dilation_iterations`: The number of times the dilation is applied. The more iterations, the more
        the dilation will spread.
    The best way to chose is to test on a few images using `draw_images=True`.

    Once satisfactory regions can be output in the csv, please transfer the project to VIA2 :

    - [Download VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/)
    - Open a new VIA2 project and import your images
    - In the `project`-menu, chose import file/region attributes and import `default_via_attributes.json` in `ajmc/data`.
    - In the `annotation`-menu, chose import annotations from csv, and import the output csv annotations.
    - It is then recommended to save your project as a json (and to delete the csv).

    To automatically annotated some of the detected regions, please refer to `automatic_region_classification` in `ajmc/olr/_scripts`

    Args:
        img_dir: {image_dir}. Can be both `Path` or `str` as this function is an entrypoint.
        output_dir: Absolute path to the dir in which to output csv and images (if draw_images). Can be both
        `Path` or `str` as this function is an entrypoint.
        dilation_kernel_size: {dilation_kernel_size}
        dilation_iterations: {dilation_iterations}
        draw_images: Whether to draw the detected regions and output the image.
        artifact_size_threshold: {artifact_size_threshold}
        img_extension: {image_format}
    """

    # Dirs
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(sorted(list(img_dir.glob(f'*.{img_extension}'))), desc='Processing images '):
        via_csv_dict = detect_regions(img_path=img_path,
                                      dilation_kernel_size=dilation_kernel_size,
                                      dilation_iterations=dilation_iterations,
                                      draw_image=draw_images,
                                      img_output_dir=output_dir,
                                      via_csv_dict=variables.VIA_CSV_DICT_TEMPLATE,
                                      artifact_size_threshold=artifact_size_threshold)

    write_csv_manually('detected_annotations.csv', via_csv_dict, str(output_dir))
    logger.info(f'{len(via_csv_dict["filename"])} zones were automatically detected')

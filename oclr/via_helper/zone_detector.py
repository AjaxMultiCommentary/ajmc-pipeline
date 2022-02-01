import cv2
from typing import List
import os
import numpy as np
from oclr.utils import image_processing
from oclr.utils.geometry import Shape, is_rectangle_within_rectangle
from oclr.utils.image_processing import remove_artifacts_from_contours


def detect_zones(img_path: str,
                 output_dir: str,
                 dilation_kernel_size: int,
                 dilation_iterations: int,
                 artifact_size_threshold: int,
                 draw_rectangles: bool,
                 via_csv_dict: dict) -> dict:
    """Automatically detects regions of interest for every image in `IMG_DIR`, using a simple dilation process.
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
        contained_contours = [c for c in contours if is_rectangle_within_rectangle(c.bounding_rectangle, dc.bounding_rectangle)]

        if contained_contours:
            contained_stacked = Shape.from_numpy_array(np.concatenate([c.bounding_rectangle for c in contained_contours], axis=0))
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


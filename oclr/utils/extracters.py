"""
`extracters.py` stores all the function used to extract data from images, hocr, html or PAGE-xml formats.
"""
import copy
import os
import numpy as np
import cv2
from typing import List


def get_image_format(args: type) -> str :
    """Retrieve the image format of the first image in IMG_DIR. Returns a `.xxx`-like string"""
    for filename in os.listdir(args.IMG_DIR):
        if filename[-4:] in [".png", ".tif",".jpg", ".jp2"]:
            image_format = filename[-4:]
            break
    return image_format


def get_id(args: "ArgumentParser", segment: "Segment", data_type: str) -> str:
    """ Retrieves id from a lace or an ocrd segment.
    :param args: an argument parser
    :param segment: a segment
    :param data_type: the type of data (groundtruth or ocr)
    :return: ID of the segment
    """
    if args.ocr_engine == "ocrd" and data_type == "ocr":
        id = segment.getAttribute("id")
    else:
        id = segment["id"]
    return id


def get_coords(args, segment, data_type):
    """ Retrieves coords from a lace or an ocrd segment.
        Input :  a segment.
        Returns : coords as a list of (x,y)-coords-tuples,
        with the following order for rectangles : upper left, upper right, down right, down left """

    if args.ocr_engine == "ocrd" and data_type == "ocr":
        coords = segment.getElementsByTagName("pc:Coords")[0].getAttribute("points").split()

        # Deal with polygons (creates the minimal bounding rect of an ocrd-polygon).
        x_coords = [int(coord.split(",")[0]) for coord in coords]
        y_coords = [int(coord.split(",")[1]) for coord in coords]
        coords = [(min(x_coords), min(y_coords)), (max(x_coords), min(y_coords)),
                  (max(x_coords), max(y_coords)), (min(x_coords), max(y_coords)),]


    else :  # exception raised if lace segment
        coords = segment["title"].split()
        coords = [(int(coords[1]), int(coords[2])), (int(coords[3]), int(coords[2])),
                  (int(coords[3]), int(coords[4])), (int(coords[1]), int(coords[4]))]

        # Make sure the segment is at least one-pixel wide
        if coords[0][0] == coords[2][0]:
            coords[2] = (coords[2][0]+1, coords[2][1])
            coords[1] = (coords[1][0] + 1, coords[1][1])

        if coords[0][1] == coords[2][1]:
            coords[2] = (coords[2][0], coords[2][1]+1)
            coords[3] = (coords[3][0], coords[3][1]+1)

    return coords


def get_content(args, segment, data_type):
    """Retrieves the content of a segment"""
    if args.ocr_engine == "ocrd" and data_type == "ocr":
        content = segment.getElementsByTagName("pc:TextEquiv")[0].getElementsByTagName("pc:Unicode")[
            0].firstChild.nodeValue
    else:
        try:
            # Not implemented yet # TODO
            if args.evaluation_level == "line":
                content = " ".join([element.contents[0] for element in segment.findChildren("html:span", recursive=False)])
            elif args.evaluation_level == "word":
                content = segment.contents[0]
        except IndexError:
            content = ""

    return content



def get_segment_zonetype(args: "Argparse.ArgumentParser", segment: "Segment", overlap_matrix: "ndarray") -> str:
    """Get the zonetype of a segment ("commentary", "app_crit"...) by selecting the maximally overlap value.

    :return: The zone type ; "no_zone" if the segment does not belong to any zone.
    """

    array = overlap_matrix[0, segment.coords[0][1]:segment.coords[2][1], segment.coords[0][0]:segment.coords[2][0]]
    segment_zones = array.flatten().squeeze().tolist()
    unique_segment_zones = []

    # TODO [optimization] this could be optimized
    zone_counts = {zone_type:0 for zone_type in args.zone_types}
    for zone_type in args.zone_types:
        for pixel_zones in segment_zones:
            if zone_type in pixel_zones:
                zone_counts[zone_type]+=1

    for key in zone_counts.keys():
        if zone_counts[key] >= 0.3*len(segment_zones):
            unique_segment_zones.append(key)

    if unique_segment_zones == []:
        unique_segment_zones.append("no_zone")

    return unique_segment_zones


def get_corresponding_ocr_word(args: "ArgumentParser", gt_word: "Segment",
                               ocr: "OcrObject", overlap_matrix: "ndarray") -> "Segment":
    """Get the ocr-segment maximally overlapping a given groundtruth segment.

    :param ocr: The OcrObject containg all ocr-segments
    :return: The corresponding ocr-segment.
    """

    array = overlap_matrix[2, gt_word.coords[0][1]:gt_word.coords[2][1],
            gt_word.coords[0][0]:gt_word.coords[2][0]]
    uniques, counts = np.unique(array, return_counts=True)
    ocr_id = uniques[counts.argmax()]

    ocr_word = None

    if ocr_id == "":
        ocr_word = copy.copy(gt_word)
        ocr_word.content = ""

    else:
        for word in ocr.words:
            if word.id == ocr_id and not word.checked:
                ocr_word = copy.copy(word)
                word.checked = True

        if ocr_word is None:
            ocr_word = copy.copy(gt_word)
            ocr_word.content = ""


    return ocr_word


def find_included_contours(coords: List[tuple], image: "Image", mode : str):
    """Finds the contours included in a segment's surrounding box.

    :param mode: "height", "width“ or "height+width", determines how contours that are at the limit of the surrounding
    box (and cut by it) should be treated :
        - "height" : contours that would increase the height of the box are discarded
        - "width" : contours that would increase the width of the box are discarded
        - "height+width" : both are discarded
        - None : no contour is discarded, box remains the same
    :return: The contours included in the segment's surrounding box
    """

    array = image.i[3, coords[0][1]:coords[2][1], coords[0][0]:coords[2][0]]
    contours = [image.contours[unique] for unique in np.unique(array) if unique != -1]

    # Exludes contours that would increase the height of the surrounding box
    if mode is not None:
        contours_new = []

        for contour in contours:
            if mode == "height":
                if (np.max(contour[:, :, 1:2]) <= coords[2][1]) and (
                        np.min(contour[:, :, 1:2]) >= coords[0][1]):
                    contours_new.append(contour)

            elif mode == "width":
                if (np.max(contour[:, :, 0:1]) <= coords[2][0]) and (
                        np.min(contour[:, :, 0:1]) >= coords[0][0]):
                    contours_new.append(contour)

            elif mode == "height+width":
                if ((np.max(contour[:, :, 1:2]) <= coords[2][1]) and \
                    (np.min(contour[:, :, 1:2]) >= coords[0][1])) \
                        and \
                        ((np.max(contour[:, :, 0:1]) <= coords[2][0]) and (
                        np.min(contour[:, :, 0:1]) >= coords[0][0])):
                    contours_new.append(contour)

        return contours_new

    else:
        return contours



# TODO : Merge with annotation helper
def get_bounding_rectangles(contours: List["ndarray"], coords: List[tuple]) -> List[tuple]:
    """Concatenate provided contours-arrays and finds the smallest bounding rectangle."""

    try:
        all_contours = contours.pop(0)
        for contour in contours:
            all_contours = np.concatenate((all_contours,contour), axis=0)
        x, y, w, h = cv2.boundingRect(all_contours)
        rect_coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        return rect_coords

    except IndexError:  # If the segment is empty

        return coords


import codecs
from bs4 import BeautifulSoup
from xml.dom import minidom
import os
import cv2
from oclr.utils import extracters
import numpy as np
from typing import List, Tuple


class Registrable:
    """In case of later integration with dhSegment"""
    pass


class File(Registrable):
    """The default class for file objects.

    :param filename: a filename without extension
    """

    def __init__(self, filename: str):
        self.filename = filename



class Segment(Registrable):
    """The default class for segment-like elements ; The default constructor should always be called by format-specific alternative
    constructors (class-methods below).

    :param id: The ID of the segment
    :param coords: The coords of the segment
    :param zone_type: The type of region the segment is delimitating \
    (for zones) or included in (for words) (e.g. "commentary").
    :param data_type: The data type of the object ("groundtruth" or "ocr")
    :param content: The text of the segment, None for zone-like segments
    :param source: The original xml-like object (bs4 or dom object)
    """

    def __init__(self, content: str = None,
                 contours: List[tuple] = None,
                 coords: List[Tuple] = None,
                 data_type: str = None,
                 id: str = None,
                 raw_coords: List[tuple] = None,
                 source: object = None,
                 zone_type: str = None):
        self.content = content
        self.contours = contours
        self.coords = coords  #: coords reduced to the contours they include ; used for analysis
        self.data_type = data_type  #: the type of data, "ocr" or "groundtruth"
        self.id = id
        self.raw_coords = raw_coords  #: Coords used only for final html rendering.
        self.source = source
        self.zone_type = zone_type

        self.print_width = self.raw_coords[1][0] - self.raw_coords[0][0]
        self.print_height = self.raw_coords[2][1] - self.raw_coords[0][1]

        self.checked = False

    @classmethod
    def from_lace_svg(cls, zone, image: "Image", svg_height, svg_width):
        """This constructor creates a segment from a lace_svg zone"""

        x1 = int(round(float(zone.getAttribute(
            "x")) * image.width / svg_width))  # svg coordinates must be converted from viewport dimensions
        x2 = x1 + int(round(float(zone.getAttribute("width")) * image.width / svg_width))
        y1 = int(round(float(zone.getAttribute("y")) * image.height / svg_height))
        y2 = y1 + int(round(float(zone.getAttribute("height")) * image.height / svg_height))

        # Coords are list of (x,y)-point-tuples, going clockwise around the figure, consistent with PAGE-XML notation
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        id = zone.getAttribute('id')
        zone_type = zone.getAttribute("data-rectangle-type")

        return cls(id=id, raw_coords=coords, coords=coords, zone_type=zone_type, source=zone)

    @classmethod
    def from_via_region(cls, via_region: dict, image: "Image", id: str = None):
        """This constructor creates a segment from a via_region"""

        x1 = via_region["shape_attributes"]["x"]
        x2 = x1 + via_region["shape_attributes"]["width"]
        y1 = via_region["shape_attributes"]["y"]
        y2 = y1 + via_region["shape_attributes"]["height"]

        # Coords are list of (x,y)-point-tuples, going clockwise around the figure, consistent with PAGE-XML notation
        raw_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        zone_type = via_region["region_attributes"]["text"]

        # if zone_type != 'line_number_commentary': # avoid shrinking zones for lemmas
        #     contours = extracters.find_included_contours(raw_coords, image, mode="height+width")
        #     coords = extracters.get_bounding_rectangles(contours, raw_coords)
        #
        # else:
        #     coords = raw_coords
        # TODO : uncomment this for olr :)

        return cls(id=id, raw_coords=raw_coords, coords=raw_coords, zone_type=zone_type, source=via_region)

    @classmethod
    def from_ocr(cls, args: "argparse.ArgumentParser", image: "Image", segment, data_type: str) -> "Segment":
        """This constructor uses the functions from utils which accept both Lace and OCR-D."""

        content = extracters.get_content(args, segment, data_type)
        raw_coords = extracters.get_coords(args, segment, data_type)

        contours = extracters.find_included_contours(raw_coords, image, mode="height")
        coords = extracters.get_bounding_rectangles(contours, raw_coords)

        id = extracters.get_id(args, segment, data_type)

        return cls(content=content, contours=contours, coords=coords, data_type=data_type, id=id,
                   raw_coords=raw_coords, source=segment)


class OcrObject(File):
    """The default class for groundtruth and ocr-object ; Constructor builds instances depending on the engine and the
    data-type.

    :param args: an argparse parser
    :param filename: a filename without extension
    :param data_type: the data type of the object ("groundtruth" or "ocr")
    """

    def __init__(self, args: "argparse.ArgumentParser", image: "Image", filename: str, data_type: str):

        if data_type == "groundtruth":
            self.filename = filename
            self.file = codecs.open(os.path.join(args.GROUNDTRUTH_DIR, self.filename + ".html"), 'r')
            self.source = BeautifulSoup(self.file.read(), "html.parser")
            self.words_source = self.source.find_all("html:span", attrs={"class": "ocr_word"})
            self.words = []
            for source_word in self.words_source:
                # Appends only if contains characters
                word = Segment.from_ocr(args, image, source_word, "groundtruth")
                if word.content not in ["", " ", "  ", "   "]:
                    self.words.append(word)



        else:
            if args.ocr_engine == "lace":
                self.filename = filename
                self.file = codecs.open(os.path.join(args.OCR_DIR, self.filename + ".html"), 'r')
                self.source = BeautifulSoup(self.file.read(), "html.parser")
                self.words = self.source.find_all("html:span", attrs={"class": "ocr_word"})
                self.words = [Segment.from_ocr(args, image, ocr_word, "ocr") for ocr_word in self.words]

            else:  # for ocrd
                # Find the corresponding file :
                for filename_ in os.listdir(args.OCR_DIR):
                    if filename_.endswith(filename + ".xml"):
                        self.filename = filename_[:-4]

                self.source = minidom.parse(os.path.join(args.OCR_DIR, self.filename + ".xml"))
                self.words = self.source.getElementsByTagName("pc:Word")
                self.words = [Segment.from_ocr(args, image, ocr_word, "ocr") for ocr_word in self.words]


class ZonesMasks(File):
    """The default class for ZonesMasks. They can be extracted from Lace-svg's or from VIA-projects .json files"""

    csv_dict = {"filename": [],
                "file_size": [],
                "file_attributes": [],
                "region_count": [],
                "region_id": [],
                "region_shape_attributes": [],
                "region_attributes": []}

    def __init__(self, filename: str, file, linked_image, linked_image_name, svg_height, svg_width, zones):
        self.filename = filename
        self.file = file
        self.linked_image = linked_image
        self.linked_image_name = linked_image_name
        self.svg_height = svg_height
        self.svg_width = svg_width
        self.zones = zones

    @classmethod
    def from_lace_svg(cls, args: "argparse.ArgumentParser", image: "Image"):
        """Constructor.

        :param filename: a filename without extension
        :param image_format: image extension format with a pattern ".xxx"
        """

        svg = minidom.parse(os.path.join(args.SVG_DIR, image.filename + ".svg"))

        # Retrieve image and file-name
        linked_image = svg.getElementsByTagName("image")[0]
        linked_image_name = os.path.basename(linked_image.getAttribute("xlink:href"))[0:-4] + image.image_format

        # Retrieve svg viewports dimensions
        svg_height = float(svg.getElementsByTagName("svg")[0].getAttribute("height"))
        svg_width = float(svg.getElementsByTagName("svg")[0].getAttribute("width"))

        # Retrieve SVG zones
        zones = [r for r in svg.getElementsByTagName('rect') if r.getAttribute("data-rectangle-type") in
                 ["commentary", "page_number", "primary_text", "title", "app_crit", "translation"]]

        zones_shrinked = [Segment.from_lace_svg(zone, image, svg_height=svg_height, svg_width=svg_width) for zone in
                          zones]

        return cls(image.filename, svg, linked_image, linked_image_name, svg_height, svg_width, zones_shrinked)

    @classmethod
    def from_via_json(cls, image: "Image", via_project):
        """Instantiate a ZonesMasks-element which `zones`-attributes containes the list of regions in the given file"""

        for key in via_project["_via_img_metadata"].keys():
            if image.filename in key:
                zones = via_project["_via_img_metadata"][key]["regions"]
                break

        zones_shrinked = [Segment.from_via_region(zone, image) for zone in zones]

        return cls(filename=image.filename, file=None, linked_image=None, linked_image_name=None, svg_height=None,
                   svg_width=None, zones=zones_shrinked)

    def convert_lace_svg_to_via_csv_dict(self, args: "argparse.ArgumentParser"):
        """Convert svg-files from lace into VIA2 readable csv-files_dict, stored in ZonesMasks.csv_dict,
        a `'key':[listofvalues]`-like dictionnary containing all the imported rectangles for all the images"""

        linked_image_height = cv2.imread(os.path.join(args.IMG_DIR, self.linked_image_name)).shape[0]
        linked_image_width = cv2.imread(os.path.join(args.IMG_DIR, self.linked_image_name)).shape[1]

        for zone in self.zones:
            ZonesMasks.csv_dict["filename"].append(os.path.basename(self.linked_image_name))
            ZonesMasks.csv_dict["file_size"].append(os.stat(
                os.path.join(args.IMG_DIR,
                             self.linked_image_name)).st_size)
            ZonesMasks.csv_dict["file_attributes"].append({})
            ZonesMasks.csv_dict["region_count"].append(len(self.zones))
            ZonesMasks.csv_dict["region_id"].append(zone.getAttribute("data-zone-ordinal"))
            ZonesMasks.csv_dict["region_shape_attributes"].append({"name": "rect",
                                                                   "x": int(round(float(
                                                                       zone.getAttribute(
                                                                           "x")) * linked_image_width / self.svg_width)),
                                                                   "y": int(round(float(
                                                                       zone.getAttribute(
                                                                           "y")) * linked_image_height / self.svg_height)),
                                                                   "width": int(round(float(
                                                                       zone.getAttribute(
                                                                           "width")) * linked_image_width / self.svg_width)),
                                                                   "height": int(round(float(zone.getAttribute(
                                                                       "height")) * linked_image_height / self.svg_height)),
                                                                   })
            ZonesMasks.csv_dict["region_attributes"].append({"text": zone.getAttribute("data-zone-type")})

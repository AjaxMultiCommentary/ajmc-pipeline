"""Basic operations and objects for image processing."""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from lazy_objects.lazy_objects import lazy_property, lazy_init
from matplotlib import pyplot as plt
from PIL import Image as PILImage, ImageDraw, ImageFont
from skimage.util import random_noise

from ajmc.commons import variables
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.geometry import Shape
from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)


class AjmcImage:
    """Default class for ajmc images.

    Note:
          The center of `AjmcImage`-coordinates is the upper left corner, consistantly with cv2 and numpy. This implies
          that Y-coordinates are ascending towards the bottom of the image.
    """

    @lazy_init
    @docstring_formatter(**docstrings)
    def __init__(self,
                 id: Optional[str] = None,
                 path: Optional[Path] = None,
                 matrix: Optional[np.ndarray] = None,
                 word_range: Optional[Tuple[int, int]] = None):
        """Default constructor.

        Args:
            id: The id of the image
            path: {path} to the image.
            matrix: an np.ndarray containing the image. Overrides self.matrix if not None.
            word_range: {word_range}
        """


    @lazy_property
    def matrix(self) -> np.ndarray:
        """np.ndarray of the image image matrix. Its shape is (height, width, channels)."""
        return cv2.imread(str(self.path))

    @lazy_property
    def height(self) -> int:
        return self.matrix.shape[0]

    @lazy_property
    def width(self) -> int:
        return self.matrix.shape[1]

    @lazy_property
    def contours(self):
        return find_contours(self.matrix)

    def crop(self,
             box: variables.BoxType,
             margin: int = 0) -> 'AjmcImage':
        """Gets the slice of `self.matrix` corresponding to `box`.

        Args:
            box: The bbox delimiting the desired crop
            margin: The extra margin desired around `box`

        Returns:
             A new `AjmcImage` containing the desired crop.
        """
        cropped = self.matrix[box[0][1] - margin:box[1][1] + margin, box[0][0] - margin:box[1][0] + margin, :]

        return AjmcImage(matrix=cropped)

    def write(self, output_path: Path):
        cv2.imwrite(str(output_path), self.matrix)

    def show(self):
        cv2.imshow('image', self.matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def binarize(img_matrix: np.ndarray,
             inverted: bool = False):
    """Binarizes an `img_matrix` using cv2 and Otsu's method."""
    binarization_type = (cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) if inverted else (cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, type=binarization_type)[1]


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Converts an RGB tuple to BGR."""
    return rgb[2], rgb[1], rgb[0]


def draw_box(box: variables.BoxType,
             img_matrix: np.ndarray,
             stroke_color: Tuple[int, int, int] = (0, 0, 255),
             stroke_thickness: int = 1,
             fill_color: Optional[Tuple[int, int, int]] = None,
             fill_opacity: float = 1,
             text: str = None,
             text_size: float = .8,
             text_thickness: int = 2):
    """Draws a box on `img_matrix`.

    Args:
        box: A list of bboxes.
        img_matrix: The image matrix on which to draw the box.
        stroke_color: The color of the box contour.
        stroke_thickness: The thickness of the box contour.
        fill_color: The color of the box fill.
        fill_opacity: The opacity of the box fill.
        text: The text to be written on the box.
        text_size: The size of the text.
        text_thickness: The thickness of the text.

    Returns:
        np.ndarray: The modified `img_matrix`

    """

    if fill_color is not None:
        sub_img_matrix = img_matrix[box[0][1]:box[1][1], box[0][0]:box[1][0]]  # Creates the sub-image
        box_fill = sub_img_matrix.copy()  # Creates the fill to be added
        box_fill[:] = rgb_to_bgr(fill_color)  # Fills the fill with the desired color
        img_matrix[box[0][1]:box[1][1], box[0][0]:box[1][0]] = cv2.addWeighted(src1=sub_img_matrix,
                                                                               # Adds the fill to the image
                                                                               alpha=1 - fill_opacity,
                                                                               src2=box_fill,
                                                                               beta=fill_opacity,
                                                                               gamma=0)

    img_matrix = cv2.rectangle(img_matrix, pt1=box[0], pt2=box[1],
                               color=rgb_to_bgr(stroke_color),
                               thickness=stroke_thickness)

    if text is not None:
        # Start by getting the actual size of the text_box
        (text_width, text_height), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                       fontScale=text_size,
                                                       thickness=text_thickness)

        # Draw a rectangle around the text
        img_matrix = cv2.rectangle(img_matrix,
                                   pt1=(box[1][0] - text_width - 4, box[0][1] - text_height - 4),
                                   pt2=(box[1][0], box[0][1]),
                                   color=rgb_to_bgr(stroke_color),
                                   thickness=-1)  # -1 means that the rectangle will be filled

        img_matrix = cv2.putText(img_matrix, text,
                                 org=(box[1][0] - text_width, box[0][1] - 2),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=text_size,
                                 color=(255, 255, 255),
                                 thickness=text_thickness)

    return img_matrix


def draw_textcontainers(img_matrix: np.ndarray, output_path: Optional[Union[str, Path]] = None, *textcontainers):
    """Draws a list of `TextContainer`s on `img_matrix`."""

    # Get the set of textcontainer types
    for tc in textcontainers:
        if tc.type == 'region':
            img_matrix = draw_box(box=tc.bbox.bbox,
                                  img_matrix=img_matrix,
                                  stroke_color=variables.REGION_TYPES_TO_COLORS[tc.region_type],
                                  stroke_thickness=2,
                                  fill_color=variables.REGION_TYPES_TO_COLORS[tc.region_type],
                                  fill_opacity=.3,
                                  text=tc.region_type)

        elif tc.type in ['entity', 'sentence', 'hyphenation']:
            for i, bbox in enumerate(tc.bboxes):
                if i == len(
                        tc.bboxes) - 1:  # We write the region label text only if it's the last bbox to avoid overlap
                    img_matrix = draw_box(box=bbox.bbox,
                                          img_matrix=img_matrix,
                                          stroke_color=variables.TEXTCONTAINERS_TYPES_TO_COLORS[tc.type],
                                          stroke_thickness=2,
                                          fill_color=variables.TEXTCONTAINERS_TYPES_TO_COLORS[tc.type],
                                          fill_opacity=.3,
                                          text=tc.label if tc.type == 'entity' else tc.type)
                else:
                    img_matrix = draw_box(box=bbox.bbox,
                                          img_matrix=img_matrix,
                                          stroke_color=variables.TEXTCONTAINERS_TYPES_TO_COLORS[tc.type],
                                          stroke_thickness=2,
                                          fill_color=variables.TEXTCONTAINERS_TYPES_TO_COLORS[tc.type],
                                          fill_opacity=.3)


        else:
            img_matrix = draw_box(box=tc.bbox.bbox,
                                  img_matrix=img_matrix,
                                  stroke_color=variables.TEXTCONTAINERS_TYPES_TO_COLORS[tc.type],
                                  stroke_thickness=1,
                                  fill_color=None,
                                  text=tc.type.capitalize())

    if output_path is not None:
        cv2.imwrite(str(output_path), img_matrix)

    return img_matrix


def draw_reading_order(img_matrix: np.ndarray,
                       page: Union['OcrPage', 'CanonicalPage'],
                       output_path: Optional[Union[str, Path]] = None):
    # Compute word centers
    w_centers = [w.bbox.center for w in page.children.words]
    img_matrix = cv2.polylines(img=img_matrix,
                               pts=[np.array(w_centers, np.int32).reshape((-1, 1, 2))],
                               isClosed=False,
                               color=(255, 0, 0),
                               thickness=4)
    if output_path:
        cv2.imwrite(output_path, img_matrix)

    return img_matrix


def find_contours(img_matrix: np.ndarray,
                  binarize: bool = True) -> List[Shape]:
    """Finds contours using `cv2.findContours`, potentially binarizing the image first.

    Args:
        img_matrix (np.ndarray): The image matrix to find contours in.
        binarize (bool): Whether to binarize the image first.

    Returns:
        List[Shape]: A list of `Shape`s representing the contours.
    """

    # This has to be done in cv2. Using cv2.THRESH_BINARY_INV to avoid looking for the white background as a contour
    if binarize:
        gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    else:
        thresh = img_matrix

    # alternative: CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Discard single-point contours
    contours = [Shape.from_numpy_array(c) for c in contours if len(c) > 1]

    return contours


def draw_contours(img_matrix: np.ndarray,
                  contours: List[Shape],
                  outfile: Optional[Union[str, Path]] = None):
    """Draws the contours of an `img_matrix` on a white image."""
    white = np.zeros([img_matrix.matrix.shape[0], img_matrix.matrix.shape[1], 3], dtype=np.uint8)
    white.fill(255)

    for c in contours:
        color = (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))

        white = cv2.polylines(img=white,
                              pts=[np.array(c.points, np.int32).reshape((-1, 1, 2))],
                              isClosed=True,
                              color=color,
                              thickness=4)

        white = cv2.rectangle(white, pt1=c.bbox[0], pt2=c.bbox[1], color=color,
                              thickness=1)

    if outfile:
        cv2.imwrite(outfile, white)


def remove_artifacts_from_contours(contours: List[Shape],
                                   artifact_perimeter_threshold: float) -> List[Shape]:
    """Removes contours if the perimeter of their bounding box is inferior to `artifact_perimeter_threshold`"""

    contours_ = [c for c in contours if (2 * (c.width + c.height)) > artifact_perimeter_threshold]
    logger.info(f"""Removed {len(contours) - len(contours_)} artifacts""")

    return contours_


def resize_image(img: np.ndarray,
                 target_height) -> np.ndarray:
    """Resize image to target height while maintaining aspect ratio."""

    scale_percent = target_height / img.shape[0]  # percent of original size
    target_width = int(img.shape[1] * scale_percent)
    dim = target_width, target_height

    return cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_AREA)


def create_text_image(text: str,
                      font_path: Path,
                      padding: int,
                      image_height: int,
                      output_file: Optional[Path] = None) -> 'PIL.Image':
    """Draws text on a white image with given font, padding and image height."""
    # Todo come back here once tesseract experiments are done

    # Get the font size
    font_size = int(0.75*(image_height - 2 * padding))  # 0.75 for conversion from pixels to points

    # Get the font
    font = ImageFont.truetype(str(font_path), font_size)

    # Get the text size
    length = font.getlength(text)

    # Create the image
    image = PILImage.new('RGB', (int(length + 2 * padding), image_height), color='white')

    # Draw the text
    draw = ImageDraw.Draw(image)
    draw.text((padding, 0), text, font=font, fill='black')

    if output_file:
        image.save(output_file)

    return image

    # lines = {
    #     'modern_greek_line': 'Η Ελλάδα είναι μια χώρα στην Μεσόγειο, στην Ευρώπη',
    #     'polytonic_greek_line': 'μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος',
    #     'number_line': '1234567890',
    #     'english_line': 'The quick brown fox jumps',
    #     'mixed_line': '123. μῆνιν ἄειδε θεὰ — The quick brown fox jumps',
    # }
    #
    # output_dir = Path('/Users/sven/Desktop/test/')
    # for font in Path('data/greek_fonts').rglob('*.[ot]tf'):
    #     for type_, line in lines.items():
    #         create_text_image(text=line,
    #                           font_path=font,
    #                           padding=0,
    #                           image_height=100,
    #                           output_file=output_dir / f'{font.stem}_{type_}.png')


def align_rgb_values(img):
    # input is numpy array
    mean = np.mean(img, axis=2, keepdims=True)
    mean_img = np.tile(mean, (1,1,3))
    return np.array(mean_img, dtype='uint8')


def add_noise(img, noise_type, show_fig=True):
    if noise_type.lower() in ["s&p"]:
        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(img, mode='s&p',amount=0.4)
    elif noise_type.lower() in ["gaussian"]:
        noise_img = random_noise(img, mode='gaussian', clip=True, mean=0, var=0.2)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = align_rgb_values(255*noise_img)
    if show_fig:
        plt.imshow(noise_img)
        plt.axis("off")
    return noise_img

import argparse
import sys

parser = argparse.ArgumentParser()


def general_args(parser: "ArgumentParser") -> "ArgumentParser":
    """Adds general cli arguments to the parser.

    :param parser: an argparse parser
    :return: augmented argparse parser.
    """

    parser.add_argument("--img_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="Absolute path to the directory where the image-files are stored")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="Absolute path to the directory in which outputs are to be stored")

    parser.add_argument("--via_project",
                        default=None,
                        type=str,
                        required=False,
                        help="Absolute path to the .json via project.")

    return parser


def evaluator_args(parser: "ArgumentParser") -> "ArgumentParser":
    """Adds evaluator-specific cli arguments to the parser.

    :param parser: argparse parser
    :return: augmented argparse parser.
    """

    parser.add_argument(
        "--GROUNDTRUTH_DIR",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the directory in which groundtruth-files are stored")

    parser.add_argument(
        "--OCR_DIR",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the directory in which ocr-files are stored")

    parser.add_argument(
        "--evaluation_level",
        default="word",
        type=str,
        required=False,
        help="""The level at which evaluation should be performed ("line" or "word")""")

    parser.add_argument(
        "--PARENT_DIR",
        default=None,
        type=str,
        required=False,
        help="""If provided, all paths will be overwritten, all paths will be overwritten to test all ocr-outputs
        in the current folder. Specific folder structure needed""")

    parser.add_argument(
        "--fuzzy_eval", action="store_true",
        help="Whether to harmonise punctuation unicode before calculating the distance")

    return parser


def annotation_helper_args(parser: "ArgumentParser") -> "ArgumentParser":
    """Adds region_detection-specific cli arguments to the parser.

    :param parser: argparse parser
    :return: augmented argparse parser.
    """

    parser.add_argument("--dilation_kernel_size",
                        default=51,
                        type=int,
                        help="Dilation kernel size, preferably an odd number. "
                             "Tweak this parameter and `--dilation_iterations` "
                             "to improve automatic boxing.")

    parser.add_argument("--dilation_iterations",
                        default=1,
                        type=int,
                        help="Number of iterations in dilation, default 1")

    parser.add_argument("--artifact_perimeter_threshold",
                        default=15,
                        type=int,
                        help="Minimum perimeter (in pixels) of the contours to be kept. "
                             "Below this threshold contours are discarded."
                             "Recommended is around 0.01*image_width")

    parser.add_argument("--draw_rectangles",
                        action="store_true",
                        help="Whether to output images with both shrinked and dilated "
                             "rectangles. This is usefull if you want to have a look at "
                             "images, e.g. to test dilation parameters. ")

    return parser



def actualize_args(args: 'argparse.Namespace') -> 'argparse.Namespace':
    pass



if sys.argv[0] == "evaluator":
    parser = evaluator_args(general_args(parser))
    args = parser.parse_args()

elif sys.argv[0] == "region_detection":
    parser = annotation_helper_args(general_args(parser))
    args = parser.parse_args()


# For local testing
else:
    parser = evaluator_args(annotation_helper_args(general_args(parser)))

    args = parser.parse_args(["--output_dir",
                              "/Users/sven/Desktop/",
                              "--img_dir",
                              "/Users/sven/ajmc/data/images",
                              # "--GROUNDTRUTH_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/ocr/evaluation/groundtruth/html/",
                              # "--OCR_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/ocr/evaluation/groundtruth/html/",
                              # "--via_project", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/olr/via_project.json",
                              "--dilation_kernel_size", "29",
                              "--dilation_iterations", "2",
                              "--artifact_perimeter_threshold", "10",
                              "--draw_rectangles",
                              # "--merge_zones"
                              ])

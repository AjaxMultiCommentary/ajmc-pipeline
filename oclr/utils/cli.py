import argparse
import sys

parser = argparse.ArgumentParser()


def general_args(parser: "ArgumentParser") -> "ArgumentParser":
    """Adds general cli arguments to the parser.

    :param parser: an argparse parser
    :return: augmented argparse parser.
    """

    parser.add_argument(
        "--IMG_DIR",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the directory where the image-files are stored")

    parser.add_argument(
        "--SVG_DIR",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the directory where the svg-files are stored. Must be given if `--via_project` is not")

    parser.add_argument(
        "--OUTPUT_DIR",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the directory in which outputs are to be stored")

    parser.add_argument(
        "--via_project",
        default=None,
        type=str,
        required=False,
        help="Absolute path to the .json via project. Must be given if `--SVG_DIR` is not.")

    parser.add_argument(
        "--draw_rectangles", action="store_true", help="Whether to output images with both shrinked and dilated "
                                                       "rectangles. This is usefull if you want to have a look at "
                                                       "images, e.g. to test dilation parameters. ")

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
        "--fuzzy_eval", action="store_true", help="Whether to harmonise punctuation unicode before calculating the distance")



    return parser


def annotation_helper_args(parser: "ArgumentParser") -> "ArgumentParser":
    """Adds via_helper-specific cli arguments to the parser.

    :param parser: argparse parser
    :return: augmented argparse parser.
    """

    parser.add_argument(
        "--dilation_kernel_size",
        default=51,
        type=int,
        help="Dilation kernel size, preferably an odd number. Tweak this parameter and `--dilation_iterations` "
             "to improve automatic boxing.")

    parser.add_argument(
        "--dilation_iterations",
        default=1,
        type=int,
        help="Number of iterations in dilation, default 1")

    parser.add_argument(
        "--artifact_threshold",
        default=0.01,
        type=float,
        help="Size-threshold under which contours are to be considered as artifacts and removed, expressed as a "
             "percentage of image height. Default is 0.01")


    parser.add_argument(
        "--merge_zones", action="store_true", help="Whether to add automatically detected zones to Lace-zones before "
                                                   "exporting annotation file")

    return parser


if sys.argv[0] == "evaluator":
    parser = evaluator_args(general_args(parser))
    args = parser.parse_args()

elif sys.argv[0] == "via_helper":
    parser = annotation_helper_args(general_args(parser))
    args = parser.parse_args()


# For local testing
else:
    parser = evaluator_args(annotation_helper_args(general_args(parser)))

    # args = parser.parse_args(["--PARENT_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/",
    #                           "--evaluation_level", "word",  # "line" level not implemented yet # TODO
    #                           "--dilation_kernel_size", "27",
    #                           "--dilation_iterations", "2",
    #                           "--artifact_threshold", "0.020",
    #                           # "--fuzzy_eval"
    #                           # "--draw_rectangles"
    #                           # "--merge_zones"
    #                           ])

    args = parser.parse_args(["--OUTPUT_DIR", "/Users/sven/oclr/via_helper",
                              "--IMG_DIR", "/Users/sven/oclr/via_helper/data/test_png",
                              # #"--SVG_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/jebb/ocr/evaluation/groundtruth/svg",
                              # "--GROUNDTRUTH_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/ocr/evaluation/groundtruth/html/",
                              # "--OCR_DIR", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/ocr/evaluation/groundtruth/html/",
                              # "--via_project", "/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentary_data/lobeck/olr/via_project.json",
                              # "--evaluation_level", "word", # "line" level not implemented yet
                              "--dilation_kernel_size", "27",
                              "--dilation_iterations", "2",
                              "--artifact_threshold", "0.020",
                              "--draw_rectangles",
                              # "--merge_zones"
                              ])

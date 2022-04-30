import os
from olr.region_detection.utils import write_csv_manually, detect_regions
from commons.cli import args
from commons.variables import VIA_CSV_DICT_TEMPLATE



for filename in os.listdir(args.img_dir):
    if filename[-3:] in ["png", "jpg", "tif", "jp2"]:
        print("Processing image " + filename)

        detect_regions(img_path=os.path.join(args.img_dir, filename),
                       output_dir=args.output_dir,
                       dilation_kernel_size=args.dilation_kernel_size,
                       dilation_iterations=args.dilation_iterations,
                       artifact_size_threshold=args.artifact_size_threshold,
                       draw_rectangles=args.draw_rectangles,
                       via_csv_dict=VIA_CSV_DICT_TEMPLATE)

write_csv_manually("detected_annotations.csv", VIA_CSV_DICT_TEMPLATE, args.output_dir)
print("{} zones were automatically detected".format(len(VIA_CSV_DICT_TEMPLATE["filename"])))
